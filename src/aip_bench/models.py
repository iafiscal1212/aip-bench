"""
Bench Models: Abstraction layer for different model backends.

Supports HuggingFace, OpenAI, Anthropic, and dummy (for testing).
Each backend is optional — loaded only when its dependencies are installed.

Usage:
    from aip_bench.models import load_model

    model = load_model("dummy")                    # testing
    model = load_model("hf:distilgpt2")            # local HF model
    model = load_model("openai:gpt-4o")            # OpenAI API
    model = load_model("anthropic:claude-sonnet-4-5-20250929")  # Anthropic API

    answer = model.generate("What is 2+2?")
    idx = model.classify("Q: Capital of France?", ["London", "Paris", "Berlin"])

Author: Carmen Esteban
"""

import hashlib


class BaseModel:
    """Abstract base class for model backends."""

    @property
    def name(self):
        raise NotImplementedError

    def generate(self, prompt, max_tokens=256, temperature=0.0):
        """Generate text given a prompt."""
        raise NotImplementedError

    def log_probs(self, prompt, completion=None):
        """Get log probabilities for tokens."""
        raise NotImplementedError

    def classify(self, prompt, choices):
        """Pick the most likely choice. Returns index."""
        raise NotImplementedError

    def prompt_hash(self, prompt):
        """SHA-256 hash of a prompt (for caching)."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]


class DummyModel(BaseModel):
    """Returns fixed responses. For testing without real models."""

    def __init__(self, name="dummy", default_response="the answer",
                 default_choice=0):
        self._name = name
        self.default_response = default_response
        self.default_choice = default_choice

    @property
    def name(self):
        return self._name

    def generate(self, prompt, max_tokens=256, temperature=0.0):
        return self.default_response

    def log_probs(self, prompt, completion=None):
        return [-0.5] * 10

    def classify(self, prompt, choices):
        return min(self.default_choice, len(choices) - 1)


class HuggingFaceModel(BaseModel):
    """HuggingFace transformers model backend.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g. 'distilgpt2', 'meta-llama/Llama-2-7b').
    device : str
        Device ('cpu', 'cuda', 'mps').
    max_length : int
        Max input length for tokenizer truncation.
    """

    def __init__(self, model_name="distilgpt2", device="cpu", max_length=512):
        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "torch and transformers required. "
                "Install: pip install 'aip-bench[bench-full]'"
            )

        self._name = model_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def name(self):
        return self._name

    def generate(self, prompt, max_tokens=256, temperature=0.0):
        import torch
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-7) if temperature > 0 else 1.0,
            )
        input_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(
            out[0, input_len:], skip_special_tokens=True
        ).strip()

    def log_probs(self, prompt, completion=None):
        import torch
        text = prompt + (completion or "")
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[0, :-1]
        token_ids = inputs["input_ids"][0, 1:]
        lp = torch.log_softmax(logits, dim=-1)
        token_lp = lp[range(len(token_ids)), token_ids]
        return token_lp.cpu().tolist()

    def classify(self, prompt, choices):
        import torch
        best_idx = 0
        best_score = float("-inf")
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[1]

        for i, choice in enumerate(choices):
            text = prompt + " " + str(choice)
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits[0]
            lp = torch.log_softmax(logits, dim=-1)
            token_ids = inputs["input_ids"][0]
            score = 0.0
            count = 0
            for j in range(prompt_len, len(token_ids)):
                if j > 0:
                    score += lp[j - 1, token_ids[j]].item()
                    count += 1
            if count > 0:
                score /= count
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx


class OpenAIModel(BaseModel):
    """OpenAI API model backend.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g. 'gpt-4o', 'gpt-3.5-turbo').
    api_key : str, optional
        API key. Falls back to OPENAI_API_KEY env var.
    """

    def __init__(self, model_name="gpt-4o", api_key=None):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai required. Install: pip install openai>=1.0"
            )
        self._name = model_name
        self.client = openai.OpenAI(api_key=api_key)

    @property
    def name(self):
        return self._name

    def generate(self, prompt, max_tokens=256, temperature=0.0):
        response = self.client.chat.completions.create(
            model=self._name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def log_probs(self, prompt, completion=None):
        response = self.client.chat.completions.create(
            model=self._name,
            messages=[{"role": "user", "content": prompt + (completion or "")}],
            max_tokens=1,
            logprobs=True,
        )
        content = response.choices[0].logprobs
        if content and content.content:
            return [t.logprob for t in content.content]
        return []

    def classify(self, prompt, choices):
        formatted = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
        )
        response = self.generate(
            f"{prompt}\n{formatted}\nAnswer with just the letter:",
            max_tokens=1,
        )
        for i in range(len(choices)):
            if chr(65 + i) in response.upper():
                return i
        return 0


class AnthropicModel(BaseModel):
    """Anthropic API model backend.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g. 'claude-sonnet-4-5-20250929').
    api_key : str, optional
        API key. Falls back to ANTHROPIC_API_KEY env var.
    """

    def __init__(self, model_name="claude-sonnet-4-5-20250929", api_key=None):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic required. Install: pip install anthropic>=0.18"
            )
        self._name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)

    @property
    def name(self):
        return self._name

    def generate(self, prompt, max_tokens=256, temperature=0.0):
        response = self.client.messages.create(
            model=self._name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def log_probs(self, prompt, completion=None):
        # Anthropic API does not expose log probs
        return []

    def classify(self, prompt, choices):
        formatted = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
        )
        response = self.generate(
            f"{prompt}\n{formatted}\nAnswer with just the letter:",
            max_tokens=1,
        )
        for i in range(len(choices)):
            if chr(65 + i) in response.upper():
                return i
        return 0


def load_model(spec):
    """Load a model from a spec string.

    Formats
    -------
    "dummy"                          DummyModel
    "dummy:response_text"            DummyModel with custom response
    "hf:model_name"                  HuggingFaceModel
    "hf:model_name:device"           HuggingFaceModel on specific device
    "openai:model_name"              OpenAIModel
    "anthropic:model_name"           AnthropicModel

    Parameters
    ----------
    spec : str
        Model specification string.

    Returns
    -------
    BaseModel
        Instantiated model.
    """
    parts = spec.split(":", maxsplit=2)
    backend = parts[0].lower()

    if backend == "dummy":
        response = parts[1] if len(parts) > 1 else "the answer"
        return DummyModel(default_response=response)
    elif backend == "hf":
        model_name = parts[1] if len(parts) > 1 else "distilgpt2"
        device = parts[2] if len(parts) > 2 else "cpu"
        return HuggingFaceModel(model_name=model_name, device=device)
    elif backend == "openai":
        model_name = parts[1] if len(parts) > 1 else "gpt-4o"
        return OpenAIModel(model_name=model_name)
    elif backend == "anthropic":
        model_name = parts[1] if len(parts) > 1 else "claude-sonnet-4-5-20250929"
        return AnthropicModel(model_name=model_name)
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            "Use: dummy, hf, openai, or anthropic."
        )
