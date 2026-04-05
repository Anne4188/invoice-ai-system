# Responsible for training the LoRA adapter (output to data/models/...)


"""
Why write train_lora.py?

Because I want to turn weak_labels_clean into a reproducible training pipeline:
- Generate training data (weak supervision)
- Reduce reliance on LLMs (LoRA)
- Quantitatively evaluate improvements (eval)

Its responsibilities are only three things:
- Read weak_labels_clean/*.jsonl (each line contains an input and target)
- Fine-tune a seq2seq model such as FLAN-T5-small with LoRA
- Output the LoRA adapter to data/models/lora_merchant_norm/
"""


# src/training/train_lora.py
from __future__ import annotations

import inspect
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, TaskType


class JsonlSeq2SeqDataset(Dataset):
    def __init__(self, path: str):
        self.rows: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        obj = self.rows[idx]
        return {
            "input": obj["input"],
            "target": obj["target"],
        }


@dataclass
class TokenizedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def make_preprocess_fn(tokenizer, max_source_len: int, max_target_len: int):
    def preprocess(ex):
        src = ex["input"]
        tgt = ex["target"]

        model_inputs = tokenizer(
            src,
            max_length=max_source_len,
            truncation=True,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                tgt,
                max_length=max_target_len,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--base_model", default="google/flan-t5-small")
    parser.add_argument("--output_dir", default="data/models/lora_merchant_norm")

    parser.add_argument("--max_source_len", type=int, default=512)
    parser.add_argument("--max_target_len", type=int, default=32)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")  
    parser.add_argument("--bf16", action="store_true")  
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[train_lora] base_model={args.base_model}")
    print(f"[train_lora] train={args.train}")
    print(f"[train_lora] val={args.val}")
    print(f"[train_lora] output_dir={args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    # LoRA configuration (seq2seq)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds_raw = JsonlSeq2SeqDataset(args.train)
    val_ds_raw = JsonlSeq2SeqDataset(args.val)

    preprocess = make_preprocess_fn(tokenizer, args.max_source_len, args.max_target_len)

    
    def tokenize_dataset(ds):
        feats = []
        for i in range(len(ds)):
            feats.append(preprocess(ds[i]))
        return feats

    train_feats = tokenize_dataset(train_ds_raw)
    val_feats = tokenize_dataset(val_ds_raw)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "evaluation_strategy" if "evaluation_strategy" in ta_params else "eval_strategy"

    training_args_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=[],
    )

    
    training_args_kwargs[eval_key] = "epoch"

    training_args = TrainingArguments(**training_args_kwargs)



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_feats,
        eval_dataset=val_feats,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save only the LoRA adapter (used by lora_normalizer.py)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"[train_lora] saved adapter -> {args.output_dir}")
    print("[train_lora] done.")


if __name__ == "__main__":
    main()

