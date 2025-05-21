
import torch
from transformers import TextStreamer
import argparse
import os

from model_utils import load_model_and_tokenizer # Using the refactored loading
from config import SYSTEM_PROMPT, MAX_SEQ_LENGTH, MODEL_NAME as DEFAULT_MODEL_NAME
from data_utils import get_book_sum_ds # To get a sample chapter if needed

def run_inference(args):
    # Load the fine-tuned model (PEFT adapters merged or base model if full finetune)
    # If PEFT adapters were saved, model_name should point to the base model,
    # and then LoRA weights should be loaded on top.
    # Unsloth's FastModel.from_pretrained can often load adapters automatically if they are in the same directory
    # or if the model_name points to a Hub repo with adapters.

    # For simplicity, let's assume args.model_path points to a model ready for inference
    # (either base model if loading adapters separately, or merged model, or fully finetuned model)
    print(f"Loading model from: {args.model_path}")

    # If model_path contains PEFT adapters, FastModel should handle it.
    # If it's a fully merged model, it will load that.
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_path, # This path should contain the model and tokenizer
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        # full_finetuning might not be relevant here if loading a saved model
        # PEFT adapters are usually auto-detected by Unsloth if in the right place
    )
    model.eval() # Set to evaluation mode

    if args.chapter_text:
        chapter_text = args.chapter_text
    elif args.sample_from_dataset:
        print("Fetching a sample chapter from the dataset...")
        try:
            # Get a sample from the test set for inference
            sample_dataset = get_book_sum_ds("test", chapter_length_filter=0) # No filter for this sample
            if len(sample_dataset) > 0:
                chapter_text = sample_dataset[0]["chapter"]
                print("Using the first chapter from the test dataset.")
            else:
                print("Could not get a sample from the dataset. Please provide --chapter_text.")
                return
        except Exception as e:
            print(f"Error loading sample from dataset: {e}. Please provide --chapter_text.")
            return
    else:
        print("No input text provided. Use --chapter_text or --sample_from_dataset.")
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, # System prompt might be implicitly handled by some models if fine-tuned with it
        {"role": "user", "content": SYSTEM_PROMPT + chapter_text}, # Re-iterate for clarity, or just chapter_text
    ]

    # Apply chat template
    # Ensure the tokenizer has a chat_template, otherwise, manual formatting is needed.
    try:
        text_input = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True, # Important for instruction-following models
            tokenize=False,
        )
    except Exception as e:
        print(f"Error applying chat template: {e}. This tokenizer might not have one.")
        print("Falling back to simple concatenation (might be suboptimal).")
        text_input = SYSTEM_PROMPT + "\n" + chapter_text # Basic fallback

    print("\n--- Input Text for Generation ---")
    print(text_input)
    print("---------------------------------\n")

    inputs = tokenizer(text_input, return_tensors="pt").to(model.device if hasattr(model, 'device') else "cuda")

    streamer = TextStreamer(tokenizer, skip_prompt=True)

    print("--- Generated Output ---")
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=True, # Important for temperature, top_p, top_k to have effect
            streamer=streamer,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, # Handle missing pad_token
        )
    print("\n------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained summarizer model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model directory (or Hugging Face model name if it contains adapters/merged model).")
    parser.add_argument("--chapter_text", type=str, default=None, help="Text of the chapter to summarize.")
    parser.add_argument("--sample_from_dataset", action="store_true", help="Use a sample chapter from the BookSum test set.")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top_p.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit for inference.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit for inference.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for private models for loading.")


    args = parser.parse_args()

    if args.hf_token:
        os.environ['HF_TOKEN'] = args.hf_token

    run_inference(args)