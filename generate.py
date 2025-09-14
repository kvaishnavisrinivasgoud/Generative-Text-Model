import sys
from transformers import pipeline

def generate_text(prompt):
    """
    Generates text using the fine-tuned GPT-2 model.
    """
    model_path = './fine_tuned_gpt2'
    try:
        generator = pipeline('text-generation', model=model_path, tokenizer=model_path)
    except OSError:
        print(f"Error: Model not found at {model_path}.")
        print("Please run train.py first to fine-tune and save the model.")
        sys.exit(1)

    print(f"Generating text for prompt: '{prompt}'")
    
    generated_texts = generator(
        prompt,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.85,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    
    print("\n--- Generated Paragraph ---")
    print(generated_texts[0]['generated_text'])

if __name__ == '__main__':
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
        generate_text(user_prompt)
    else:
        print("Usage: python generate.py \"<your prompt here>\"")