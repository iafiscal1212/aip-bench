"""
Bench Prompts: Few-shot prompt templates per benchmark dataset.

Each template has:
- system: Task instruction
- question: Format for the current item
- shot: Format for few-shot examples

Usage:
    from aip_bench.prompts import format_prompt

    prompt = format_prompt("mmlu", item, n_shots=5, examples=train_examples)

Author: Carmen Esteban
"""

_TEMPLATES = {
    "mmlu": {
        "system": "The following is a multiple choice question about {subject}.\n",
        "question": "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:",
        "shot": "Q: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: {answer}\n\n",
    },
    "hellaswag": {
        "system": "Complete the sentence with the most likely continuation.\n",
        "question": "{context}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:",
        "shot": "{context} {answer}\n\n",
    },
    "arc_challenge": {
        "system": "Answer the following science question.\n",
        "question": "{question}\n{choices_text}\nAnswer:",
        "shot": "Q: {question}\n{choices_text}\nAnswer: {answer}\n\n",
    },
    "winogrande": {
        "system": "Complete the sentence by choosing the correct option.\n",
        "question": "{sentence}\nA. {option1}\nB. {option2}\nAnswer:",
        "shot": "{sentence} Answer: {answer}\n\n",
    },
    "gsm8k": {
        "system": "Solve the following math problem step by step.\n",
        "question": "Question: {question}\nAnswer:",
        "shot": "Question: {question}\nAnswer: {answer}\n\n",
    },
    "boolq": {
        "system": "Answer the following yes/no question based on the passage.\n",
        "question": "Passage: {passage}\nQuestion: {question}\nAnswer (yes/no):",
        "shot": "Passage: {passage}\nQuestion: {question}\nAnswer: {answer}\n\n",
    },
    "fever": {
        "system": "Determine if the claim is SUPPORTS, REFUTES, or NOT ENOUGH INFO.\n",
        "question": "Claim: {claim}\nEvidence: {evidence}\nVerdict:",
        "shot": "Claim: {claim}\nEvidence: {evidence}\nVerdict: {verdict}\n\n",
    },
    "natural_questions": {
        "system": "Answer the question based on the context.\n",
        "question": "Context: {context}\nQuestion: {question}\nAnswer:",
        "shot": "Context: {context}\nQuestion: {question}\nAnswer: {answer}\n\n",
    },
    "truthfulqa": {
        "system": "Answer the following question truthfully.\n",
        "question": "{question}\nAnswer:",
        "shot": "Q: {question}\nA: {answer}\n\n",
    },
    "squad_v2": {
        "system": "Answer the question based on the context. "
                  "If unanswerable, say 'unanswerable'.\n",
        "question": "Context: {context}\nQuestion: {question}\nAnswer:",
        "shot": "Context: {context}\nQuestion: {question}\nAnswer: {answer}\n\n",
    },
    "hotpotqa": {
        "system": "Answer the following multi-hop question.\n",
        "question": "Context: {context}\nQuestion: {question}\nAnswer:",
        "shot": "Context: {context}\nQuestion: {question}\nAnswer: {answer}\n\n",
    },
    "halueval": {
        "system": "Determine if the following response contains hallucinations.\n",
        "question": "Context: {context}\nResponse: {text}\n"
                    "Does the response contain hallucinations? (yes/no):",
        "shot": "Context: {context}\nResponse: {text}\nAnswer: {answer}\n\n",
    },
}


def list_templates():
    """List available prompt templates.

    Returns
    -------
    list of str
        Template names.
    """
    return sorted(_TEMPLATES.keys())


def get_template(task):
    """Get prompt template for a task.

    Parameters
    ----------
    task : str
        Task name.

    Returns
    -------
    dict
        Template with 'system', 'question', 'shot' keys.
    """
    if task not in _TEMPLATES:
        raise ValueError(
            f"No template for {task!r}. Available: {list_templates()}"
        )
    return dict(_TEMPLATES[task])


def format_prompt(task, item, n_shots=0, examples=None):
    """Format a prompt with optional few-shot examples.

    Parameters
    ----------
    task : str
        Task name (e.g. 'mmlu', 'gsm8k').
    item : dict
        Current item to format (keys depend on task).
    n_shots : int
        Number of few-shot examples to prepend.
    examples : list of dict, optional
        Few-shot examples.

    Returns
    -------
    str
        Formatted prompt ready for model input.
    """
    template = get_template(task)
    parts = []

    # System instruction
    try:
        parts.append(template["system"].format_map(_SafeDict(item)))
    except (KeyError, IndexError):
        parts.append(template["system"])

    # Few-shot examples
    if n_shots > 0 and examples:
        for ex in examples[:n_shots]:
            try:
                parts.append(template["shot"].format_map(_SafeDict(ex)))
            except (KeyError, IndexError):
                pass

    # Current question
    try:
        parts.append(template["question"].format_map(_SafeDict(item)))
    except (KeyError, IndexError):
        parts.append(template["question"])

    return "\n".join(parts)


class _SafeDict(dict):
    """Dict that returns '{key}' for missing keys instead of raising."""
    def __missing__(self, key):
        return "{" + key + "}"
