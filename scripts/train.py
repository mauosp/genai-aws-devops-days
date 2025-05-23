import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str)
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--train_batch_size', type=int)
parser.add_argument('--learning_rate', type=float)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

dataset = load_dataset("json", data_files={"train": args.dataset_path})["train"]
dataset = dataset.map(lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=512), batched=True)

training_args = TrainingArguments(
    output_dir="/opt/ml/model",
    per_device_train_batch_size=args.train_batch_size,
    num_train_epochs=args.epochs,
    learning_rate=args.learning_rate,
    evaluation_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()