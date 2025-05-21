from datasets import load_dataset, Dataset
from config import SYSTEM_PROMPT

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return text
    return text.split("####")[1].strip()

def get_book_sum_ds(split_name: str, chapter_length_filter: int = 3500) -> Dataset:
    """Loads and preprocesses the booksum dataset for a given split."""
    try:
        data = load_dataset("nschantz21/booksum-randomized")[split_name]
    except Exception as e:
        print(f"Failed to load dataset split '{split_name}': {e}")
        raise

    # Filter by chapter length
    if chapter_length_filter > 0:
        new_ft_data = data.filter(lambda x: x['chapter_length'] is not None and int(x['chapter_length']) <= chapter_length_filter)
    else:
        new_ft_data = data

    def format_example(example):
        # Ensure 'chapter' and 'summary_text' are present
        if 'chapter' not in example or 'summary_text' not in example:
            # Potentially skip or handle missing data for robustness
            return {'prompt': [], 'answer': None, 'chapter': example.get('chapter')}

        prompt_content = SYSTEM_PROMPT + str(example['chapter']) # Ensure chapter is string
        return {
            'prompt': [{'role': 'user', 'content': prompt_content}],
            'answer': extract_hash_answer(str(example['summary_text'])), # Ensure summary is string
            'chapter': str(example['chapter'])
        }

    new_ft_data = new_ft_data.map(format_example, batched=False) # Process individual examples

    # Remove examples where formatting might have failed (e.g. None answer)
    new_ft_data = new_ft_data.filter(lambda x: x['answer'] is not None and x['prompt'] is not None)

    # Select relevant columns
    # Ensure 'chapter' column exists, if not, it might have been removed or not created correctly
    columns_to_keep = ['prompt', 'answer']
    if 'chapter' in new_ft_data.column_names:
        columns_to_keep.append('chapter')
    new_ft_data = new_ft_data.select_columns(columns_to_keep)

    return new_ft_data

if __name__ == '__main__':
    # Example usage:
    print("Loading train dataset...")
    train_ds = get_book_sum_ds("train")
    if train_ds:
        print(f"Train dataset loaded. Number of examples: {len(train_ds)}")
        print("First train example prompt:", train_ds[0]['prompt'])
        print("First train example answer:", train_ds[0]['answer'])

    print("\nLoading test dataset...")
    eval_ds = get_book_sum_ds("test")
    if eval_ds:
        print(f"Test dataset loaded. Number of examples: {len(eval_ds)}")
        print("First test example prompt:", eval_ds[0]['prompt'])
        print("First test example answer:", eval_ds[0]['answer'])