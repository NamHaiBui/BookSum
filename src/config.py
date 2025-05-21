import numpy as np

# Model and Tokenizer settings
MODEL_NAME = "OscarBui/GemmaSummerizer1.0" 
MAX_SEQ_LENGTH = 4192
LORA_RANK = 4

# System Prompts and XML Tags
REASONING_START_TAG = "<verbatim_support>"
REASONING_END_TAG = "</verbatim_support>"
SOLUTION_START_TAG = "<summary>"
SOLUTION_END_TAG = "</summary>"

SYSTEM_PROMPT = f"""
You are an AI assistant specializing in **evidence-based summarization**. Given a text:
1.  **Extract Evidence:** Identify and extract the most crucial verbatim snippets. These snippets must directly support the main points you will include in your summary. Copy them exactly.
2.  **Summarize:** Write a concise summary capturing the core information of the text. Every key point in your summary *must* be traceable to one or more of the verbatim snippets you extracted.

**Output Format:** Adhere strictly to the following structure, providing *only* the requested content within the specified tags. Do not include any other text.

{SOLUTION_START_TAG}
[Concise summary based solely on the text]
{SOLUTION_END_TAG}
{REASONING_START_TAG}
[List of exact verbatim snippets supporting the summary]
{REASONING_END_TAG}
"""

XML_COT_FORMAT = f"""\
{SOLUTION_START_TAG}
{{summary}}
{SOLUTION_END_TAG}
{REASONING_START_TAG}
{{verbatim_support}}
{REASONING_END_TAG}
"""


FP_TYPE = np.float16
TORCH_FP_TYPE = "float16"

# Reward Configuration (V8 - Float16 Attempt)
REWARD_CONFIG = {
    # --- General ---
    "print_details": False,
    "apply_raw_reward_clip": True,
    "reward_clip_min": -200.0,
    "reward_clip_max": 200.0,

    # --- Weights ---
    "w_format": 4.0,
    "w_fidelity": 3.0,
    "w_nli_support": 6.0,
    "w_semantic_support": 5.0, # Used if NLI fallback or explicit semantic
    "w_relevance": 2.0,
    "w_reference": 5.0,
    "w_conciseness": 1.5,
    "w_coherence": 1.0,

    # --- NLI Params ---
    "nli_batch_size": 64,
    "nli_support_threshold": 0.7,
    "nli_reward_scale": 5.0,
    "nli_penalty_scale": 5.0,

    # --- Semantic Support Backup Config ---
    "use_nli_support": True, # Master switch for NLI vs Semantic for main support
    "semantic_support_threshold": 0.65,
    "semantic_reward_scale": 4.0,
    "semantic_penalty_scale": 4.0,

    # --- Other Function Params ---
    "min_format_tokens": 3,
    "strict_fidelity": True,
    "min_snippet_len": 10,
    "semantic_batch_size": 128, # For semantic model embeddings
    "relevance_threshold": 0.55,
    "ref_w_semantic": 0.6,
    "ref_w_rouge": 0.4,
    "conciseness_max_reward": 1.5,
    "conciseness_sigma_factor": 0.5,
    "conciseness_oversize_factor": 1.8,
    "conciseness_penalty_scale": 4.0,
    "ideal_coherence_sim": 0.55,
    "coherence_tolerance": 0.15,
    "coherence_max_reward": 1.0,
    "coherence_min_reward": -1.0,

    # Internal flag, do not change manually
    "_printed_nli_fallback_warning": False,
}

# NLI and Semantic Model Names
NLI_MODEL_NAME = "roberta-large-mnli"
SEMANTIC_MODEL_NAME = 'all-MiniLM-L6-v2'

# Epsilon for float16 calculations
EPS = np.finfo(FP_TYPE).eps