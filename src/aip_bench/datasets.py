"""
Bench Datasets: HuggingFace loaders and synthetic data generators.

Synthetic generators produce controlled data for testing pipelines
without requiring downloads, GPU, or external dependencies.

Author: Carmen Esteban
"""

import numpy as np

try:
    import datasets as hf_datasets
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# ============================================================
# Structured KV cache generator
# ============================================================

def _structured_kv(rng, seq_len, head_dim, smoothness=0.8):
    """Generate a KV cache with realistic local correlation.

    Real transformer KV caches have:
    - High autocorrelation (adjacent tokens are very similar)
    - Low-rank structure (few dominant directions)
    - Smooth positional variation (gradual change)

    Parameters
    ----------
    rng : np.random.RandomState
        Random state.
    seq_len : int
        Sequence length.
    head_dim : int
        Head dimension.
    smoothness : float
        0.0 = pure random, 1.0 = maximally smooth/correlated.

    Returns
    -------
    np.ndarray
        Shape (seq_len, head_dim) with realistic structure.
    """
    # Low-rank basis: 4-8 dominant directions
    n_basis = min(max(4, head_dim // 8), 8)
    basis = rng.randn(n_basis, head_dim)
    basis /= np.linalg.norm(basis, axis=1, keepdims=True) + 1e-12

    # Smooth coefficients via AR(1) process (high autocorrelation)
    ar_coef = 0.9 + smoothness * 0.09  # 0.9 to 0.99
    coefs = np.zeros((seq_len, n_basis))
    coefs[0] = rng.randn(n_basis)
    for t in range(1, seq_len):
        coefs[t] = ar_coef * coefs[t - 1] + (1 - ar_coef) * rng.randn(n_basis)

    # Structured component: linear combination of basis vectors
    structured = coefs @ basis

    # Add positional encoding (sinusoidal, low frequency)
    positions = np.arange(seq_len, dtype=np.float64)[:, None]
    freqs = np.exp(np.linspace(-1, -5, head_dim))[None, :]
    pos_enc = np.sin(positions * freqs) * 0.2
    structured += pos_enc

    # Random component (small)
    noise = rng.randn(seq_len, head_dim) * 0.15

    # Blend structured and random
    result = smoothness * structured + (1 - smoothness) * noise
    return result


# ============================================================
# Dataset registry
# ============================================================

_REGISTRY = {
    "halueval_qa": {
        "path": "pminervini/HaluEval",
        "name": "qa_samples",
        "category": "hallucination",
        "description": "HaluEval QA hallucination detection samples",
    },
    "halueval_dialogue": {
        "path": "pminervini/HaluEval",
        "name": "dialogue_samples",
        "category": "hallucination",
        "description": "HaluEval dialogue hallucination detection samples",
    },
    "halueval_summarization": {
        "path": "pminervini/HaluEval",
        "name": "summarization_samples",
        "category": "hallucination",
        "description": "HaluEval summarization hallucination detection samples",
    },
    "squad_v2": {
        "path": "rajpurkar/squad_v2",
        "name": None,
        "category": "qa",
        "description": "SQuAD v2.0 question answering benchmark",
    },
    "hotpotqa": {
        "path": "hotpot_qa",
        "name": "distractor",
        "category": "qa",
        "description": "HotPotQA multi-hop question answering",
    },
    "truthfulqa": {
        "path": "truthfulqa/truthful_qa",
        "name": "multiple_choice",
        "category": "hallucination",
        "description": "TruthfulQA hallucination/truthfulness benchmark",
    },
    "mmlu": {
        "path": "cais/mmlu",
        "name": "all",
        "category": "knowledge",
        "description": "MMLU massive multitask language understanding",
    },
    "hellaswag": {
        "path": "Rowan/hellaswag",
        "name": None,
        "category": "reasoning",
        "description": "HellaSwag commonsense NLI",
    },
    "arc_challenge": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "category": "reasoning",
        "description": "ARC-Challenge science questions",
    },
    "winogrande": {
        "path": "allenai/winogrande",
        "name": "winogrande_xl",
        "category": "reasoning",
        "description": "WinoGrande commonsense reasoning",
    },
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "category": "math",
        "description": "GSM8K grade school math",
    },
    "boolq": {
        "path": "google/boolq",
        "name": None,
        "category": "qa",
        "description": "BoolQ yes/no reading comprehension",
    },
    "fever": {
        "path": "fever/fever",
        "name": "v1.0",
        "category": "hallucination",
        "description": "FEVER fact verification",
    },
    "natural_questions": {
        "path": "google-research-datasets/natural_questions",
        "name": "default",
        "category": "qa",
        "description": "Natural Questions open-domain QA",
    },
}


def list_datasets():
    """List available benchmark datasets.

    Returns
    -------
    dict
        Mapping of dataset name to {description, category}.
    """
    return {
        name: {"description": info["description"], "category": info["category"]}
        for name, info in _REGISTRY.items()
    }


def load_dataset(name, split="validation", max_samples=None):
    """Load a benchmark dataset from HuggingFace.

    Parameters
    ----------
    name : str
        Dataset name from the registry.
    split : str
        Dataset split to load.
    max_samples : int, optional
        Maximum number of samples to return.

    Returns
    -------
    list of dict
        Standardized samples:
        - hallucination: {text, context, label, category}
        - qa: {question, context, answer, id}

    Raises
    ------
    ImportError
        If the `datasets` library is not installed.
    ValueError
        If the dataset name is not in the registry.
    """
    if not HAS_DATASETS:
        raise ImportError(
            "The 'datasets' library is required for loading real datasets. "
            "Install with: pip install datasets>=2.14"
        )

    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name!r}. "
            f"Available: {list(_REGISTRY.keys())}"
        )

    info = _REGISTRY[name]
    kwargs = {"path": info["path"], "split": split}
    if info["name"] is not None:
        kwargs["name"] = info["name"]

    ds = hf_datasets.load_dataset(**kwargs)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    category = info["category"]
    samples = []

    for item in ds:
        if category == "hallucination":
            samples.append(_normalize_hallucination_sample(item, name))
        elif category == "qa":
            samples.append(_normalize_qa_sample(item, name))

    return samples


def _normalize_hallucination_sample(item, dataset_name):
    """Convert a hallucination sample to standard format."""
    if dataset_name == "truthfulqa":
        # TruthfulQA: question + mc1_targets or mc2_targets
        text = item.get("best_answer", "")
        context = item.get("question", "")
        # In TruthfulQA multiple_choice, label indicates correctness
        label_raw = item.get("label", 0)
        label = 1 - int(label_raw) if isinstance(label_raw, (int, float)) else 0
    else:
        text = item.get("answer", item.get("response", item.get("text", "")))
        context = item.get("knowledge", item.get("context", item.get("question", "")))
        label_raw = item.get("hallucination", item.get("label", ""))
        label = 1 if str(label_raw).lower() in ("yes", "1", "true") else 0
    return {
        "text": str(text),
        "context": str(context),
        "label": label,
        "category": dataset_name,
    }


def _normalize_qa_sample(item, dataset_name):
    """Convert a QA sample to standard format."""
    question = item.get("question", "")
    context = item.get("context", "")
    if dataset_name == "squad_v2":
        answers = item.get("answers", {})
        answer = answers.get("text", [""])[0] if answers.get("text") else ""
    elif dataset_name == "hotpotqa":
        answer = item.get("answer", "")
        # Combine supporting facts as context
        if not context:
            sents = item.get("context", {})
            if isinstance(sents, dict):
                titles = sents.get("title", [])
                sentences = sents.get("sentences", [])
                parts = []
                for t, s_list in zip(titles, sentences):
                    parts.append(f"{t}: {''.join(s_list)}")
                context = " ".join(parts)
    elif dataset_name == "truthfulqa":
        answer = item.get("best_answer", item.get("answer", ""))
        context = item.get("question", "")
    else:
        answer = item.get("answer", "")

    return {
        "question": str(question),
        "context": str(context),
        "answer": str(answer),
        "id": str(item.get("id", "")),
    }


# ============================================================
# Synthetic data generators
# ============================================================

class SyntheticHaluEval:
    """Synthetic hallucination evaluation dataset.

    Generates samples with controlled attention patterns and token
    probabilities for testing hallucination detection pipelines.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    separation : float
        Separability between normal and hallucinated classes.
        0.0 -> AUROC ~0.5 (random), 1.0 -> AUROC ~1.0 (perfect).
    num_heads : int
        Number of attention heads per sample.
    seq_len : int
        Sequence length for attention matrices.
    num_layers : int
        Number of layers for cross-layer consistency.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_samples=200, separation=0.8, num_heads=4,
                 seq_len=32, num_layers=3, seed=42):
        self.n_samples = n_samples
        self.separation = separation
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.seed = seed
        self._samples = None

    def _generate(self):
        """Generate all samples."""
        rng = np.random.RandomState(self.seed)
        samples = []

        for i in range(self.n_samples):
            label = i % 2  # Alternate 0/1 for balanced classes

            if label == 0:
                sample = self._generate_normal(rng)
            else:
                sample = self._generate_hallucinated(rng)

            sample["label"] = label
            samples.append(sample)

        self._samples = samples

    def _generate_normal(self, rng):
        """Normal sample: focused attention, high probs, consistent layers."""
        sep = self.separation

        # Focused attention: 1-3 targets per query, dirichlet concentrated
        attn = np.zeros((self.num_heads, self.seq_len, self.seq_len))
        for h in range(self.num_heads):
            for q in range(self.seq_len):
                n_targets = rng.randint(1, 4)
                targets = rng.choice(self.seq_len, size=n_targets, replace=False)
                # Concentrated Dirichlet
                alpha = np.ones(n_targets) * (0.1 + sep * 9.9)
                weights = rng.dirichlet(alpha)
                attn[h, q, targets] = weights

        # High token probs
        token_probs = rng.beta(2 + sep * 6, 2, size=self.seq_len)

        # Consistent layers (copy with small noise)
        layers = [attn.copy()]
        for _ in range(self.num_layers - 1):
            noise_scale = 0.01 + (1 - sep) * 0.2
            noise = rng.randn(*attn.shape) * noise_scale
            layer = np.clip(attn + noise, 0, None)
            # Re-normalize
            layer_sum = layer.sum(axis=-1, keepdims=True)
            layer_sum = np.maximum(layer_sum, 1e-12)
            layer = layer / layer_sum
            layers.append(layer)

        # High lookback (context = first half)
        context_len = self.seq_len // 2

        return {
            "attn": attn,
            "token_probs": token_probs,
            "layers": layers,
            "context_len": context_len,
        }

    def _generate_hallucinated(self, rng):
        """Hallucinated sample: uniform attn, low probs, inconsistent layers."""
        sep = self.separation

        # Uniform-ish attention
        attn = np.ones((self.num_heads, self.seq_len, self.seq_len))
        noise = rng.randn(self.num_heads, self.seq_len, self.seq_len) * (1 - sep) * 0.5
        attn = attn + noise
        attn = np.clip(attn, 0.01, None)
        attn = attn / attn.sum(axis=-1, keepdims=True)

        # Low token probs
        token_probs = rng.beta(2, 2 + sep * 6, size=self.seq_len)

        # Independent layers (not consistent)
        layers = []
        for _ in range(self.num_layers):
            layer_attn = np.ones((self.num_heads, self.seq_len, self.seq_len))
            noise = rng.randn(self.num_heads, self.seq_len, self.seq_len) * 0.5
            layer_attn = np.clip(layer_attn + noise, 0.01, None)
            layer_attn = layer_attn / layer_attn.sum(axis=-1, keepdims=True)
            layers.append(layer_attn)

        # Low lookback
        context_len = self.seq_len // 2

        return {
            "attn": attn,
            "token_probs": token_probs,
            "layers": layers,
            "context_len": context_len,
        }

    @property
    def samples(self):
        if self._samples is None:
            self._generate()
        return self._samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)


class SyntheticOckBench:
    """Synthetic reasoning efficiency benchmark dataset.

    Generates problems with attention patterns and KV caches
    to test inference efficiency pipelines.

    Parameters
    ----------
    n_problems : int
        Number of problems to generate.
    num_heads : int
        Number of attention heads.
    seq_len : int
        Sequence length.
    head_dim : int
        Head dimension for KV cache.
    base_accuracy : float
        Base accuracy before optimization.
    token_budget : int
        Token budget for efficiency calculation.
    seed : int
        Random seed.
    """

    def __init__(self, n_problems=100, num_heads=8, seq_len=64,
                 head_dim=32, base_accuracy=0.85, token_budget=1024,
                 seed=42):
        self.n_problems = n_problems
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.base_accuracy = base_accuracy
        self.token_budget = token_budget
        self.seed = seed
        self._problems = None

    def _generate(self):
        rng = np.random.RandomState(self.seed)
        problems = []

        for i in range(self.n_problems):
            # Random softmax attention
            raw = rng.randn(self.num_heads, self.seq_len, self.seq_len)
            exp = np.exp(raw - np.max(raw, axis=-1, keepdims=True))
            attn = exp / np.sum(exp, axis=-1, keepdims=True)

            # Structured KV cache (local correlation + smooth basis)
            keys = _structured_kv(
                rng, self.seq_len, self.head_dim, smoothness=0.8
            )
            values = _structured_kv(
                rng, self.seq_len, self.head_dim, smoothness=0.8
            )

            # Whether this problem is answered correctly
            correct = rng.random() < self.base_accuracy

            # Tokens used (varies per problem)
            tokens_used = rng.randint(
                int(self.token_budget * 0.3),
                int(self.token_budget * 0.9)
            )

            problems.append({
                "attn": attn,
                "keys": keys,
                "values": values,
                "correct": bool(correct),
                "tokens_used": int(tokens_used),
            })

        self._problems = problems

    @property
    def problems(self):
        if self._problems is None:
            self._generate()
        return self._problems

    def __len__(self):
        return self.n_problems

    def __getitem__(self, idx):
        return self.problems[idx]

    def __iter__(self):
        return iter(self.problems)


class SyntheticQA:
    """Synthetic QA dataset for KV cache compression evaluation.

    Generates question-context-answer triples with associated
    key-value caches for testing compression impact on QA quality.

    Parameters
    ----------
    n_samples : int
        Number of QA samples.
    seq_len : int
        Sequence length for KV cache.
    head_dim : int
        Head dimension.
    seed : int
        Random seed.
    """

    def __init__(self, n_samples=100, seq_len=128, head_dim=64, seed=42):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.seed = seed
        self._samples = None

    def _generate(self):
        rng = np.random.RandomState(self.seed)
        samples = []

        # Vocabulary of answer words
        words = [
            "the", "cat", "sat", "on", "mat", "dog", "ran", "fast",
            "big", "red", "car", "drove", "north", "south", "east",
            "west", "tree", "grew", "tall", "river", "flowed", "down",
            "mountain", "peak", "high", "above", "clouds", "rain",
            "sun", "moon", "star", "bright", "dark", "cold", "warm",
        ]

        for i in range(self.n_samples):
            # Generate plausible answer (2-5 words)
            n_words = rng.randint(2, 6)
            answer_words = [words[rng.randint(len(words))] for _ in range(n_words)]
            answer = " ".join(answer_words)

            # Context contains answer + surrounding words
            n_context = rng.randint(15, 30)
            ctx_words = [words[rng.randint(len(words))] for _ in range(n_context)]
            insert_pos = rng.randint(0, len(ctx_words))
            ctx_words[insert_pos:insert_pos] = answer_words
            context = " ".join(ctx_words)

            # Question
            question = f"What is sample {i}?"

            # Structured KV cache (local correlation + smooth basis)
            keys = _structured_kv(
                rng, self.seq_len, self.head_dim, smoothness=0.85
            )
            values = _structured_kv(
                rng, self.seq_len, self.head_dim, smoothness=0.85
            )

            samples.append({
                "question": question,
                "context": context,
                "answer": answer,
                "keys": keys,
                "values": values,
            })

        self._samples = samples

    @property
    def samples(self):
        if self._samples is None:
            self._generate()
        return self._samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)
