import time
import evaluate
import optuna
import torch
from transformers import Trainer, TrainingArguments
import transformers
import numpy as np
import os
import pandas as pd
import pickle
from nlp import Dataset

ALBERT_MODEL = 'albert-base-v2'    # https://huggingface.co/albert/albert-base-v2
tokenizer = transformers.AlbertTokenizer.from_pretrained(ALBERT_MODEL)


def encode_batch(batch):
    return tokenizer(batch['text_combined'], padding=True, truncation=True, max_length=512)


# precompute the train and test datasets
if os.path.exists('train_dataset.pkl') and os.path.exists('test_dataset.pkl'):
    with open('train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open('test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
else:
    df = pd.read_csv('phishing_email.csv')[['text_combined', 'label']].dropna()
    dataset = Dataset.from_pandas(df)
    data_dct = dataset.train_test_split(train_size=0.7)
    train_dataset = data_dct['train'].map(encode_batch, batched=True)
    test_dataset = data_dct['test'].map(encode_batch, batched=True)
    with open('train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open('test_dataset.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)

train_dataset.set_format(type='torch', columns=[
                         'input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=[
                        'input_ids', 'attention_mask', 'label'])

epoch_factor = 0.25
total_train_batches = int(len(train_dataset) * epoch_factor)

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "SAFE", 1: "PHISHING"}
label2id = {"PHISHING": 1, "SAFE": 0}

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-7, 5e-4)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [1, 2, 4, 8, 12])

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        f"albert/{ALBERT_MODEL}", num_labels=2, id2label=id2label, label2id=label2id, attn_implementation="sdpa"
    )
    torch.cuda.empty_cache()
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=False,
        auto_find_batch_size=False,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=epoch_factor,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1e-8},
        logging_dir="./logs",
        logging_steps=int(total_train_batches / batch_size / 10),
        eval_strategy="steps",
        save_strategy="no", 
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    for step, _ in enumerate(trainer.get_train_dataloader()):
        trainer.train(resume_from_checkpoint=None)
        if step % training_args.logging_steps == 0:
            metrics = trainer.evaluate()
            trial.report(metrics["eval_accuracy"], step=step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    metrics = trainer.evaluate()
    return metrics["eval_accuracy"]

tic = time.time()
study = optuna.create_study(direction="maximize", study_name="albert-phishing-detector", storage="sqlite:///albert-phishing-detector.db", load_if_exists=True)
# run for maximum of 32 hours
study.optimize(objective, n_trials=100, timeout=16*60*60)
toc = time.time()

print("Time taken:", toc - tic)
print("Best trial:", study.best_trial)
print("Best parameters:", study.best_params)


best_params = study.best_params
torch.cuda.empty_cache()
training_args = TrainingArguments(
    output_dir="./best_results",
    overwrite_output_dir=True,
    learning_rate=best_params['learning_rate'],
    weight_decay=best_params['weight_decay'],
    per_device_train_batch_size=best_params['batch_size'],
    num_train_epochs = 1,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr": 1e-8},
    logging_dir="./logs",
    logging_steps=int(total_train_batches / best_params['batch_size'] / 50),
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    bf16=True,
    report_to="tensorboard"
)

model = transformers.AutoModelForSequenceClassification.from_pretrained(
    f"albert/{ALBERT_MODEL}", num_labels=2, id2label=id2label, label2id=label2id, attn_implementation="sdpa")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("best-albert-phishing-detector")



# train my own guess

torch.cuda.empty_cache()
training_args = TrainingArguments(
    output_dir="./own",
    overwrite_output_dir=True,
    learning_rate=1e-5,
    weight_decay=1e-8,
    auto_find_batch_size=True,
    num_train_epochs = 1,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr": 1e-8},
    logging_dir="./logs",
    logging_steps=int(total_train_batches / best_params['batch_size'] / 50),
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    bf16=True,
    report_to="tensorboard",
)

model = transformers.AutoModelForSequenceClassification.from_pretrained(
    f"albert/{ALBERT_MODEL}", num_labels=2, id2label=id2label, label2id=label2id, attn_implementation="sdpa")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("best-albert-phishing-detector-guessed-values")