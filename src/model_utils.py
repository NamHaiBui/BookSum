# model_utils.py
from unsloth import FastModel
from config import MAX_SEQ_LENGTH, LORA_RANK, MODEL_NAME as DEFAULT_MODEL_NAME, TORCH_FP_TYPE

def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL_NAME,
                             load_in_4bit: bool = False,
                             load_in_8bit: bool = False,
                             use_full_finetuning: bool = False,
                             peft_random_state: int = 3407):
    """Loads the model and tokenizer with specified PEFT configurations."""
    print(f"Loading model: {model_name}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"LoRA rank: {LORA_RANK}")
    print(f"Load in 4-bit: {load_in_4bit}, Load in 8-bit: {load_in_8bit}")
    print(f"Full finetuning: {use_full_finetuning}")

    # Determine torch_dtype based on TORCH_FP_TYPE for memory efficiency
    # FastModel might handle this internally, but being explicit can be good.
    # Unsloth typically defaults to bfloat16 if available and not 4-bit/8-bit
    # Forcing float16 if TORCH_FP_TYPE is "float16" might be desired
    dtype_arg = None
    if not load_in_4bit and not load_in_8bit:
        if TORCH_FP_TYPE == "float16":
            dtype_arg = TORCH_FP_TYPE
        # Add elif for "bfloat16" if needed
        # else: Unsloth default (often bfloat16)

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=dtype_arg, # Pass determined dtype
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        # token = "YOUR_HF_TOKEN" # Optional: if model is private
    )

    if not use_full_finetuning:
        print("Applying PEFT (LoRA) configuration...")
        model = FastModel.get_peft_model(
            model,
            r=LORA_RANK,
            lora_alpha=LORA_RANK, # Typically same as r
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth", # Recommended by Unsloth
            random_state=peft_random_state,
            target_modules = [ "q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj",], # Common for Gemma-like models
            # finetune_vision_layers, finetune_language_layers etc. are more for multi-modal
            # For LLMs, target_modules is more direct for LoRA.
            # The original script had:
            # finetune_vision_layers=False, finetune_language_layers=True,
            # finetune_attention_modules=True, finetune_mlp_modules=True,
            # which are less common PEFT args. `target_modules` is preferred for Unsloth.
        )
    else:
        print("Full finetuning enabled. Skipping PEFT adapter setup.")

    return model, tokenizer

if __name__ == '__main__':
    # Example usage:
    print("Testing model and tokenizer loading...")
    # Set load_in_4bit to True if you have limited VRAM and want to test 4-bit loading
    # Note: 4-bit loading requires bitsandbytes
    model, tokenizer = load_model_and_tokenizer(load_in_4bit=False)
    if model and tokenizer:
        print("Model and tokenizer loaded successfully.")
        print("Model class:", type(model))
        print("Tokenizer class:", type(tokenizer))
    else:
        print("Failed to load model and tokenizer.")