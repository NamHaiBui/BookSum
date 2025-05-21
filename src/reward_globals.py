
import os
import re
import torch
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer as rouge_scorer_lib

from config import (
    FP_TYPE, TORCH_FP_TYPE, EPS,
    REWARD_CONFIG, NLI_MODEL_NAME, SEMANTIC_MODEL_NAME,
    REASONING_START_TAG, REASONING_END_TAG, SOLUTION_START_TAG, SOLUTION_END_TAG
)

# --- Global Variable Initialization ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Reward globals: Initializing resources on device: {DEVICE}")
print(f"Reward globals: Using numeric type {FP_TYPE} for calculations.")

# Hugging Face Token (set as an environment variable for security)
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


# --- Load Models (NLI and Semantic) ---
nli_tokenizer = None
nli_model = None
semantic_model = None

try:
    print(f"Reward globals: Loading NLI model: {NLI_MODEL_NAME}...")
    if HUGGINGFACE_TOKEN:
        nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME, token=HUGGINGFACE_TOKEN)
        # Attempt to load in specified TORCH_FP_TYPE if supported
        nli_model_dtype = torch.float16 if TORCH_FP_TYPE == "float16" else (torch.bfloat16 if TORCH_FP_TYPE == "bfloat16" else None)

        nli_model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL_NAME,
            token=HUGGINGFACE_TOKEN,
            torch_dtype=nli_model_dtype
        ).to(DEVICE)
        nli_model.eval()
        print(f"Reward globals: NLI model loaded (dtype: {nli_model.dtype if nli_model else 'N/A'}).")
    else:
        print("Reward globals: Warning - HF_TOKEN not found. NLI model loading might fail for private models or hit rate limits.")
        # Fallback or error if no token and model is gated/private
        nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        nli_model_dtype = torch.float16 if TORCH_FP_TYPE == "float16" else (torch.bfloat16 if TORCH_FP_TYPE == "bfloat16" else None)
        nli_model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL_NAME, torch_dtype=nli_model_dtype
        ).to(DEVICE)
        nli_model.eval()
        print(f"Reward globals: NLI model loaded without token (dtype: {nli_model.dtype if nli_model else 'N/A'}).")

except Exception as e:
    print(f"Reward globals: ERROR loading NLI model: {e}. NLI-dependent rewards will be impacted.")
    nli_model = None
    nli_tokenizer = None

try:
    print(f"Reward globals: Loading Semantic model: {SEMANTIC_MODEL_NAME}...")
    semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME, device=DEVICE)
    # SentenceTransformer typically handles precision. If explicit half precision is needed:
    # if TORCH_FP_TYPE == "float16" and DEVICE == "cuda":
    #     semantic_model.half()
    print("Reward globals: Semantic model loaded.")
except Exception as e:
    print(f"Reward globals: ERROR loading Semantic model: {e}. Semantic-dependent rewards will be impacted.")
    semantic_model = None

# --- Utility Functions ---
def cosine_similarity_global(embeds1: np.ndarray, embeds2: np.ndarray) -> np.ndarray:
    embeds1 = np.asarray(embeds1, dtype=FP_TYPE)
    embeds2 = np.asarray(embeds2, dtype=FP_TYPE)
    if embeds1.ndim == 1: embeds1 = embeds1.reshape(1, -1)
    if embeds2.ndim == 1: embeds2 = embeds2.reshape(1, -1)
    sim_matrix = np.dot(embeds1, embeds2.T)
    return np.clip(sim_matrix, FP_TYPE(-1.0), FP_TYPE(1.0))

try:
    print("Reward globals: Initializing ROUGE scorer...")
    rouge_scorer_global = rouge_scorer_lib.RougeScorer(['rougeL'], use_stemmer=True)
    print("Reward globals: ROUGE scorer initialized.")
except Exception as e:
    print(f"Reward globals: ERROR initializing ROUGE scorer: {e}")
    rouge_scorer_global = None

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Reward globals: Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)
sent_tokenize_global = sent_tokenize
print("Reward globals: Sentence tokenizer function defined (using NLTK).")


# --- Helper Functions for Reward Calculation ---
def extract_xml_sections(text: str) -> tuple[str, str]:
    if not isinstance(text, str): return "", ""
    summary_match = re.search(rf'{SOLUTION_START_TAG}(.*?){SOLUTION_END_TAG}', text, re.DOTALL | re.IGNORECASE)
    reasoning_match = re.search(rf'{REASONING_START_TAG}(.*?){REASONING_END_TAG}', text, re.DOTALL | re.IGNORECASE)
    summary = summary_match.group(1).strip() if summary_match else ""
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    return reasoning, summary

def safe_get_content(data_item: list | dict | str | None,
                     index_in_list: int = 0, # Used if data_item is a list of turns
                     role_key: str = 'role', # For chat format
                     content_key: str = 'content', # For chat format
                     target_role: str = 'assistant' # For chat format, to get assistant's response
                    ) -> str | None:
    """
    Safely access content from various TRL DPO/PPO/Reward input formats.
    Handles:
    - Direct strings (completions)
    - Dicts like {'response': ...}, {'text': ...}, {'content': ...} (completions)
    - Lists of chat dicts e.g., [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}]
      - For prompts, it typically gets the last user message. (target_role='user', index_in_list=-1)
      - For completions, it gets the assistant message. (target_role='assistant')
    """
    if isinstance(data_item, str):
        return data_item

    if isinstance(data_item, dict): # e.g. for completions like {"response": "..."}
        return data_item.get("response", data_item.get("text", data_item.get(content_key)))

    if isinstance(data_item, list): # Likely a list of chat turns
        if not data_item: return None

        if target_role: # Try to find content by role
            relevant_turns = [turn.get(content_key) for turn in data_item if isinstance(turn, dict) and turn.get(role_key) == target_role and turn.get(content_key)]
            if relevant_turns:
                return relevant_turns[index_in_list] if abs(index_in_list) < len(relevant_turns) else relevant_turns[-1] # Default to last if index is bad

        # Fallback if no target_role or role not found, try by index (e.g. for prompts being the last turn)
        if abs(index_in_list) < len(data_item):
            item_at_index = data_item[index_in_list]
            if isinstance(item_at_index, dict):
                return item_at_index.get(content_key)
            if isinstance(item_at_index, str): # list of strings
                return item_at_index
    return None

# Ensure REWARD_CONFIG is accessible
current_reward_config = REWARD_CONFIG