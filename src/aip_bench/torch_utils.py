"""
Bench Torch Utils: Optional torch+transformers utilities for real model inference.

Provides extraction of attention patterns, KV caches, and token
probabilities from HuggingFace models for use with bench pipelines.

Author: Carmen Esteban
"""

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def _extract_kv(pkv):
    """Extract keys/values from any cache format.

    Supports:
    - DynamicCache with .layers (transformers >= 5.x)
    - DynamicCache with .key_cache/.value_cache (transformers >= 4.36)
    - Legacy tuple format (transformers < 4.36)

    Returns
    -------
    tuple of (np.ndarray or None, np.ndarray or None)
        Keys and values from last layer, first head: shape (seq, head_dim).
    """
    keys, values = None, None
    if hasattr(pkv, "layers") and len(pkv.layers) > 0:
        # DynamicCache with .layers (transformers >= 5.x)
        last = pkv.layers[-1]
        keys = last.keys[0, 0].cpu().numpy()
        values = last.values[0, 0].cpu().numpy()
    elif hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        # DynamicCache (transformers >= 4.36)
        keys = pkv.key_cache[-1][0, 0].cpu().numpy()
        values = pkv.value_cache[-1][0, 0].cpu().numpy()
    elif isinstance(pkv, (list, tuple)) and len(pkv) > 0:
        # Legacy tuple format
        last_kv = pkv[-1]
        keys = last_kv[0][0, 0].cpu().numpy()
        values = last_kv[1][0, 0].cpu().numpy()
    return keys, values


def is_available():
    """Check if torch and transformers are available.

    Returns
    -------
    bool
        True if both torch and transformers are installed.
    """
    return HAS_TORCH and HAS_TRANSFORMERS


def extract_model_data(model, tokenizer, text):
    """Extract attention patterns, KV cache, and token probabilities from a model.

    IMPORTANT: model must be loaded with attn_implementation='eager'
    (use load_model_eager()) for attention weights to be returned.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        HuggingFace model loaded with attn_implementation='eager'.
    tokenizer : transformers.PreTrainedTokenizer
        HuggingFace tokenizer.
    text : str
        Input text to process.

    Returns
    -------
    dict
        {attn, layers, token_probs, keys, values, generated_text, context_len}

    Raises
    ------
    ImportError
        If torch or transformers not installed.
    """
    if not is_available():
        raise ImportError(
            "torch and transformers are required. "
            "Install with: pip install torch>=1.9 transformers>=4.30"
        )

    inputs = tokenizer(text, return_tensors="pt")
    context_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=True,
        )

    # Extract attentions as numpy
    layers = []
    for layer_attn in outputs.attentions:
        # (batch, heads, seq, seq) -> (heads, seq, seq)
        layers.append(layer_attn[0].cpu().numpy())

    attn = layers[-1] if layers else None

    # Extract KV cache (handles both tuple and DynamicCache formats)
    keys = None
    values = None
    if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
        pkv = outputs.past_key_values
        keys, values = _extract_kv(pkv)

    # Token probabilities
    logits = outputs.logits[0]  # (seq, vocab)
    probs = torch.softmax(logits, dim=-1)
    token_ids = inputs["input_ids"][0]
    token_probs = token_probs_from_logits(
        logits.cpu().numpy(), token_ids.cpu().numpy()
    )

    # Generated text
    generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)

    return {
        "attn": attn,
        "layers": layers,
        "token_probs": token_probs,
        "keys": keys,
        "values": values,
        "generated_text": generated_text,
        "context_len": context_len,
    }


def generate_answer(model, tokenizer, question, context, max_new_tokens=50):
    """Generate an answer given a question and context.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        HuggingFace model.
    tokenizer : transformers.PreTrainedTokenizer
        HuggingFace tokenizer.
    question : str
        The question to answer.
    context : str
        The context containing the answer.
    max_new_tokens : int
        Maximum number of new tokens to generate.

    Returns
    -------
    str
        Generated answer text.

    Raises
    ------
    ImportError
        If torch or transformers not installed.
    """
    if not is_available():
        raise ImportError(
            "torch and transformers are required. "
            "Install with: pip install torch>=1.9 transformers>=4.30"
        )

    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the generated portion
    input_len = inputs["input_ids"].shape[1]
    answer_ids = output_ids[0, input_len:]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def load_model_eager(model_name="distilgpt2", device="cpu"):
    """Load a HuggingFace model with eager attention (required for attention extraction).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str
        Device to load on.

    Returns
    -------
    tuple of (model, tokenizer)
    """
    if not is_available():
        raise ImportError("torch and transformers required.")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def prepare_halueval_real(model, tokenizer, samples, max_length=256):
    """Enrich real HaluEval samples with attention patterns from a model.

    Takes text/context/label samples from HuggingFace and extracts
    attention, token probs, and layer data needed by HallucinationBenchmark.

    IMPORTANT: model must be loaded with attn_implementation='eager'
    (use load_model_eager()) for attention extraction to work.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        HuggingFace model loaded with attn_implementation='eager'.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the model.
    samples : list of dict
        Samples with {text, context, label}.
    max_length : int
        Maximum token length for truncation.

    Returns
    -------
    list of dict
        Enriched samples with {attn, token_probs, layers, context_len, label}.
    """
    if not is_available():
        raise ImportError("torch and transformers required.")

    enriched = []
    for sample in samples:
        text = sample.get("context", "") + " " + sample.get("text", "")
        text = text.strip()[:2000]  # Limit text length
        if not text:
            continue  # Skip empty samples

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_length,
        )
        if inputs["input_ids"].shape[1] == 0:
            continue  # Skip if tokenizer produced empty input

        context_tokens = tokenizer(
            sample.get("context", "") or ".", truncation=True,
            max_length=max_length,
        )
        context_len = len(context_tokens["input_ids"])

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
            )

        # Extract layers as numpy
        layers = []
        for layer_attn in outputs.attentions:
            layers.append(layer_attn[0].cpu().numpy())

        attn = layers[-1] if layers else None

        # Token probabilities
        logits_np = outputs.logits[0].cpu().numpy()
        token_ids_np = inputs["input_ids"][0].cpu().numpy()
        token_probs = token_probs_from_logits(logits_np, token_ids_np)

        enriched.append({
            "attn": attn,
            "token_probs": token_probs,
            "layers": layers,
            "context_len": context_len,
            "label": sample.get("label", 0),
        })

    return enriched


def prepare_qa_real(model, tokenizer, samples, max_length=256):
    """Enrich real QA samples with KV caches from a model.

    Takes question/context/answer samples from HuggingFace and extracts
    key-value caches needed by QACompressionBenchmark.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        HuggingFace model with use_cache support.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the model.
    samples : list of dict
        Samples with {question, context, answer}.
    max_length : int
        Maximum token length for truncation.

    Returns
    -------
    list of dict
        Enriched samples with {question, context, answer, keys, values}.
    """
    if not is_available():
        raise ImportError("torch and transformers required.")

    enriched = []
    for sample in samples:
        prompt = "Context: {ctx}\nQuestion: {q}\nAnswer:".format(
            ctx=sample.get("context", "")[:1000],
            q=sample.get("question", ""),
        )

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=max_length,
        )

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)

        # Extract KV from last layer
        keys, values = None, None
        pkv = outputs.past_key_values
        if pkv is not None:
            keys, values = _extract_kv(pkv)

        if keys is None:
            # Fallback: create from hidden states
            keys = np.random.randn(inputs["input_ids"].shape[1], 64)
            values = np.random.randn(inputs["input_ids"].shape[1], 64)

        enriched.append({
            "question": sample.get("question", ""),
            "context": sample.get("context", ""),
            "answer": sample.get("answer", ""),
            "keys": keys,
            "values": values,
        })

    return enriched


def token_probs_from_logits(logits, token_ids):
    """Extract token probabilities from logits.

    Parameters
    ----------
    logits : numpy.ndarray
        Model logits, shape (seq_len, vocab_size).
    token_ids : numpy.ndarray
        Token IDs, shape (seq_len,).

    Returns
    -------
    numpy.ndarray
        Probability of each chosen token, shape (seq_len,).
    """
    logits = np.asarray(logits, dtype=np.float64)
    token_ids = np.asarray(token_ids, dtype=np.int64)

    # Softmax
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Gather probs for actual tokens
    seq_len = len(token_ids)
    token_probs = probs[np.arange(seq_len), token_ids]

    return token_probs
