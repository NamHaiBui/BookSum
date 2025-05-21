
import os
import argparse
import torch
from accelerate import Accelerator
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import wandb # For logging, ensure user is logged in via CLI or env var

from config import (
    MODEL_NAME as DEFAULT_MODEL_NAME,
    MAX_SEQ_LENGTH,
    REWARD_CONFIG, TORCH_FP_TYPE
)
from data_utils import get_book_sum_ds
from model_utils import load_model_and_tokenizer
from reward_functions import combined_reward_function_v8 # This is our main reward interface
from reward_globals import HUGGINGFACE_TOKEN # To check if token is set for model loading

def main(args):
    # Initialize Accelerator
    # Mixed precision can be set here for overall training if desired,
    # Unsloth handles its own precision for the base model layers.
    # GRPOTrainer will also leverage Accelerate.
    accelerator = Accelerator(mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None)
    accelerator.print(f"Accelerator initialized with state: {accelerator.state}")
    accelerator.print(f"Using device: {accelerator.device}")
    accelerator.print(f"Number of processes: {accelerator.num_processes}")
    accelerator.print(f"Distributed type: {accelerator.distributed_type}")

    # Set Hugging Face token from args if provided, otherwise use reward_globals.HUGGINGFACE_TOKEN
    # This ensures the token is available for model loading if needed.
    # The model_utils.load_model_and_tokenizer can be adapted to accept a token argument.
    # For now, we assume FastModel.from_pretrained will pick up HF_TOKEN env var if set,
    # or if the model is public.
    if args.hf_token:
        os.environ['HF_TOKEN'] = args.hf_token
    elif not HUGGINGFACE_TOKEN and not args.model_name.startswith("OscarBui"): # Assuming OscarBui models are public
        accelerator.print("Warning: Hugging Face token not provided and model might be private. Set via --hf_token or HF_TOKEN env var.")


    # Load model and tokenizer
    # model_utils.load_model_and_tokenizer can be updated to accept accelerator.device
    # However, GRPOTrainer and Unsloth typically handle device placement well with Accelerate.
    # For 4-bit/8-bit, Unsloth manages this. For full precision, Accelerate will move the model.
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_full_finetuning=args.full_finetuning
    )
    accelerator.print("Model and tokenizer loaded.")

    # Load datasets
    accelerator.print("Loading datasets...")
    train_dataset = get_book_sum_ds("train", chapter_length_filter=args.max_chapter_len)
    eval_dataset = get_book_sum_ds("test", chapter_length_filter=args.max_chapter_len) # Use same filter for eval
    accelerator.print(f"Train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}")

    # GRPO Configuration
    # Update REWARD_CONFIG verbosity for main process during training
    if accelerator.is_main_process:
        REWARD_CONFIG['print_details'] = args.print_reward_details

    grpo_config = GRPOConfig(
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim=args.optimizer, #"adamw_torch_fused" if using torch >= 2.0 and CUDA, else "adamw_torch"
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size, # This is per_device
        gradient_accumulation_steps=args.gradient_accumulation,
        num_generations=args.num_generations, # K in GRPO paper (number of completions per prompt)
        max_prompt_length=args.max_prompt_length, # Max length for prompts
        max_completion_length=args.max_completion_length, # Max length for generated completions
        num_train_epochs=args.epochs if args.max_steps == -1 else None, # Use epochs if max_steps not set
        max_steps=args.max_steps if args.max_steps != -1 else -1, # Use max_steps if set
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_grad_norm=0.1, # From original script
        report_to="wandb" if args.use_wandb else None,
        output_dir=args.output_dir,
        seed=args.seed,
        gradient_checkpointing=True, # Recommended by Unsloth for PEFT
        # GRPOTrainer specific or common TRL args:
        # fp16 = (accelerator.mixed_precision == "fp16"), # TRL might pick this from accelerator
        # bf16 = (accelerator.mixed_precision == "bf16"),
        remove_unused_columns=False, # Important for GRPO as it needs 'answer' etc.
        # Check TRL/GRPO docs for more specific Accelerate integration args if needed
    )
    accelerator.print("GRPOConfig initialized.")
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        accelerator.print("Weights & Biases initialized.")


    # Instantiate GRPOTrainer
    # GRPOTrainer is designed to work with Accelerate.
    # It will handle model and data preparation using the accelerator instance.
    trainer = GRPOTrainer(
        model=model, # Already PEFT model if not full_finetuning
        tokenizer=tokenizer,
        reward_func=combined_reward_function_v8, # Single reward function that combines all aspects
                                                 # TRL's GRPO trainer expects a single function
                                                 # that returns a list of scalar rewards.
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # Optional, if you want to run evals
        # accelerator=accelerator, # GRPOTrainer should automatically use the global accelerator
    )
    accelerator.print("GRPOTrainer initialized.")

    # Start training
    accelerator.print("Starting training...")
    trainer.train()
    accelerator.print("Training finished.")

    # Save final model
    if accelerator.is_main_process:
        final_save_path = os.path.join(args.output_dir, "final_model")
        trainer.save_model(final_save_path)
        # If using PEFT, Unsloth models might have a specific way to save merged weights
        # For PEFT, `model.save_pretrained(final_save_path)` saves adapters.
        # To save a merged model:
        if not args.full_finetuning: # PEFT case
            try:
                accelerator.print("Attempting to merge and save final PEFT model...")
                # Ensure model is on CPU and in full precision for merging if issues arise
                # model.cpu().float().save_pretrained_merged(final_save_path, tokenizer, save_method = "merged_16bit")
                # Or use the model directly if merging on GPU is fine
                merged_model_path = os.path.join(args.output_dir, "final_merged_model_16bit")

                # Detach from accelerator for merging if needed, or ensure it's on one device
                # unwrapped_model = accelerator.unwrap_model(model) # Get base model if wrapped

                # For Unsloth PEFT models, saving the adapter is usually sufficient,
                # or use its specific merging methods if available and compatible.
                # The original script used: model.save_pretrained("GemmaSummerizer1.0", tokenizer)
                # For merged:
                # model.save_pretrained_merged("output_merged_16bit", tokenizer, save_method = "merged_16bit")
                # model.push_to_hub_merged(...)
                # Let's stick to saving adapters via trainer.save_model for simplicity here.
                # Merging can be a separate step.
                accelerator.print(f"PEFT Adapters saved to {final_save_path} by trainer.save_model().")
                accelerator.print("For a merged model, load adapters and merge separately or use Unsloth's merging utilities.")

            except Exception as e:
                accelerator.print(f"Error during final model merging/saving: {e}")
        else: # Full finetuning case
            accelerator.print(f"Full model saved to {final_save_path} by trainer.save_model().")

        accelerator.print(f"Final model artifacts saved to {final_save_path}")

    if args.use_wandb and accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a summarizer model using GRPO.")

    # Model and Data
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Pretrained model name or path.")
    parser.add_argument("--output_dir", type=str, default="outputs_grpo", help="Directory to save checkpoints and logs.")
    parser.add_argument("--max_chapter_len", type=int, default=3500, help="Max chapter length for filtering data.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit.")
    parser.add_argument("--full_finetuning", action="store_true", help="Enable full finetuning instead of PEFT.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for private models.")

    # Training Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--optimizer", type=str, default="adamw_torch_fused", choices=["adamw_torch", "adamw_torch_fused", "adamw_bnb_8bit"], help="Optimizer.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size.") # Reduced default for memory
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient accumulation steps.") # Increased default
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (if max_steps is -1).")
    parser.add_argument("--max_steps", type=int, default=-1, help="Total number of training steps. Overrides epochs. -1 for epoch-based.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument("--max_prompt_length", type=int, default=MAX_SEQ_LENGTH - 1024, help="Maximum prompt length for tokenizer during training.") # Adjusted
    parser.add_argument("--max_completion_length", type=int, default=1024, help="Maximum completion length for generation during training.") # Adjusted

    # GRPO Specific
    parser.add_argument("--num_generations", type=int, default=4, help="Number of completions per prompt (K in GRPO).")

    # Logging and Saving
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit the total amount of checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--print_reward_details", action="store_true", help="Print detailed reward calculations during training (main process only).")

    # Accelerator
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision training.")

    # W&B
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="grpo_summarizer", help="W&B project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (defaults to auto-generated).")


    args = parser.parse_args()

    # Some basic validation or adjustments
    if args.load_in_4bit or args.load_in_8bit:
        if args.optimizer == "adamw_torch_fused":
            print("Warning: Fused AdamW might not be optimal with 4/8bit. Consider 'adamw_torch' or 'adamw_bnb_8bit'.")
        if args.mixed_precision not in ["no", "fp16"]: # bf16 with 4/8bit can be tricky
             print(f"Warning: Using {args.mixed_precision} with 4/8bit. 'fp16' or 'no' might be more stable if issues arise.")


    # Correcting TORCH_FP_TYPE based on mixed_precision for model loading consistency.
    # This assumes model_utils.py will use the global TORCH_FP_TYPE from config.
    # It's better if model_utils.load_model_and_tokenizer takes dtype explicitly.
    # For now, we'll just print a reminder.
    print(f"Note: Global TORCH_FP_TYPE is '{TORCH_FP_TYPE}'. Model loading will try to use this if not 4/8 bit.")
    print(f"Accelerator mixed_precision is '{args.mixed_precision}'. Ensure these are compatible for your setup.")


    main(args)