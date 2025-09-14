import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def train_model():
    # --- 1. Load Model & Tokenizer ---
    print("Loading pre-trained model and tokenizer...")
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Add a padding token to the tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    # --- 2. Prepare Dataset ---
    print("Preparing dataset...")
    train_file_path = './data.txt'
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file_path,
        block_size=128
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # --- 3. Define Training Arguments ---
    print("Setting up training arguments...")
    output_dir = './fine_tuned_gpt2'
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    # --- 4. Train the Model ---
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    print("Starting fine-tuning... 🚀")
    trainer.train()
    
    # --- 5. Save the Final Model ---
    print("Saving the fine-tuned model.")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete! Model saved to '{}'".format(output_dir))

if __name__ == '__main__':
    train_model()