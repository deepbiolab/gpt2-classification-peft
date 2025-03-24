import os
import torch
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
import wandb

# Global configurations
SEED = 42
MODEL_NAME = "gpt2"
ID2LABEL = {0: "not_similar", 1: "similar"}
LABEL2ID = {"not_similar": 0, "similar": 1}

# Set random seeds uniformly
def set_seeds():
    torch.manual_seed(SEED)
    if torch.mps.is_available():
        torch.mps.manual_seed(SEED)
    set_seed(SEED)

def setup_wandb():
    os.environ["WANDB_WATCH"] = "false"
    config = {
        "model_name": MODEL_NAME,
        "learning_rate": 2e-5,
        "epochs": 3,
        "batch_size": 8,
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "seed": SEED,
    }
    wandb.init(
        project="medical-qa-peft",
        name="gpt2-lora-experiment",
        config=config,
        settings=wandb.Settings(console="off"),
    )
    return config

def load_and_prepare_dataset():
    dataset = load_dataset("medical_questions_pairs")
    if "validation" not in dataset:
        train_testvalid = dataset["train"].train_test_split(test_size=0.2, seed=SEED)
        test_valid = train_testvalid["test"].train_test_split(test_size=0.5, seed=SEED)
        dataset = {
            "train": train_testvalid["train"],
            "validation": test_valid["train"],
            "test": test_valid["test"],
        }
    return dataset

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    metrics = {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
    wandb.log(metrics)
    return metrics

def get_training_args(config):
    return TrainingArguments(
        output_dir="./checkpoints",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        logging_dir="./logs",
        logging_steps=10,
    )

def preprocess_data(dataset, tokenizer):
    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["question_1"],
            examples["question_2"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    return {
        split: dataset[split].map(
            preprocess_function,
            batched=True,
            remove_columns=dataset[split].column_names,
        )
        for split in dataset.keys()
    }

def evaluate_model(model, trainer, model_name=""):
    """Evaluate model performance"""
    print(f"\nEvaluating {model_name}...")
    metrics = trainer.evaluate()
    print(f"{model_name} metrics:", metrics)
    return metrics

def main():
    # Initialize settings
    set_seeds()
    config = setup_wandb()
    
    # Load dataset
    dataset = load_and_prepare_dataset()
    
    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Preprocess data
    encoded_dataset = preprocess_data(dataset, tokenizer)
    
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        pad_token_id=tokenizer.eos_token_id,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Get training arguments
    training_args = get_training_args(wandb.config)
    
    # Create trainer for base model evaluation
    base_trainer = Trainer(
        model=base_model,
        args=training_args,
        eval_dataset=encoded_dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # Evaluate base model
    base_metrics = evaluate_model(base_model, base_trainer, "Base Model")
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=["c_attn", "c_proj"],
        fan_in_fan_out=True,
    )

    # Create PEFT model
    peft_model = get_peft_model(base_model, peft_config)
    print("Trainable parameters:", peft_model.print_trainable_parameters())

    # Create trainer for PEFT model
    peft_trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    # Train PEFT model
    print("Training PEFT model...")
    peft_trainer.train()
    
    # Evaluate PEFT model
    peft_metrics = evaluate_model(peft_model, peft_trainer, "PEFT Model")

    # Compare and record performance differences
    print("\nPerformance Comparison:")
    print(f"Base Model Accuracy: {base_metrics['eval_accuracy']:.4f}")
    print(f"PEFT Model Accuracy: {peft_metrics['eval_accuracy']:.4f}")
    print(f"Improvement: {(peft_metrics['eval_accuracy'] - base_metrics['eval_accuracy'])*100:.2f}%")

    # Log comparison results to wandb
    comparison_data = [
        ["Base Model", base_metrics["eval_accuracy"]],
        ["PEFT Model", peft_metrics["eval_accuracy"]]
    ]
    wandb.log({
        "model_comparison": wandb.plot.bar(
            wandb.Table(data=comparison_data, columns=["Model Type", "Accuracy"]),
            "Model Type",
            "Accuracy",
            title="Model Accuracy Comparison"
        )
    })

    # Save model
    peft_model.save_pretrained("./peft_model")

    # Test set prediction and results saving
    test_results = peft_trainer.predict(encoded_dataset["test"])
    
    test_df = pd.DataFrame({
        "question_1": [item["question_1"] for item in dataset["test"]],
        "question_2": [item["question_2"] for item in dataset["test"]],
        "predictions": test_results.predictions.argmax(axis=1),
        "true_labels": test_results.label_ids,
    })
    
    test_df["prediction_text"] = test_df["predictions"].map(ID2LABEL)
    test_df["true_label_text"] = test_df["true_labels"].map(ID2LABEL)
    test_df["is_correct"] = test_df["predictions"] == test_df["true_labels"]
    
    os.makedirs("test_results", exist_ok=True)
    test_df.to_csv("test_results/predictions.csv", index=False)
    print("\nTest results saved to predictions.csv")

    wandb.finish()

if __name__ == "__main__":
    main()