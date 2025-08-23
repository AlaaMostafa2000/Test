import sys
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

model_name = sys.argv[1]
dataset_path = sys.argv[2]
output_dir = sys.argv[3]

dataset = load_dataset('text', data_files={'train': dataset_path})

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch['text'])

tokenized = dataset.map(tokenize, batched=True, remove_columns=['text'])

model = AutoModelForCausalLM.from_pretrained(model_name)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    data_collator=collator,
)

trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
