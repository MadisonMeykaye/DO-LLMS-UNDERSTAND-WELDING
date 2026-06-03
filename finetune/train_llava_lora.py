from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def apply_chat_template(processor, messages, add_generation_prompt=False) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except TypeError:
        text = processor.apply_chat_template(messages)
        if add_generation_prompt and not text.rstrip().endswith("[/INST]"):
            return text
        return text


class WeldLlavaDataset(Dataset):
    def __init__(self, jsonl_path: Path, limit: int | None = None):
        self.rows = load_jsonl(jsonl_path)
        if limit is not None:
            self.rows = self.rows[:limit]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


@dataclass
class LlavaSFTCollator:
    processor: object

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        images = []
        full_texts = []
        prompt_texts = []
        sample_weights = []

        for ex in examples:
            with Image.open(ex["image_path"]) as image:
                images.append(image.convert("RGB"))
            full_texts.append(
                apply_chat_template(
                    self.processor,
                    ex["messages"],
                    add_generation_prompt=False,
                )
            )
            prompt_texts.append(
                apply_chat_template(
                    self.processor,
                    ex["messages"][:-1],
                    add_generation_prompt=True,
                )
            )
            sample_weights.append(float(ex.get("sample_weight", 1.0)))

        model_inputs = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = model_inputs["input_ids"].clone()
        token_weights = torch.ones_like(labels, dtype=torch.float32)

        for i in range(labels.shape[0]):
            prompt_len = int(prompt_inputs["attention_mask"][i].sum().item())
            labels[i, :prompt_len] = -100
            token_weights[i, :prompt_len] = 0.0
            token_weights[i, prompt_len:] *= sample_weights[i]

        labels[model_inputs["attention_mask"] == 0] = -100
        token_weights[model_inputs["attention_mask"] == 0] = 0.0

        model_inputs["labels"] = labels
        model_inputs["token_weights"] = token_weights
        return model_inputs


class WeightedLlavaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        token_weights = inputs.pop("token_weights")
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_weights = token_weights[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        loss = loss * shift_weights.view(-1)

        denom = shift_weights.view(-1).sum().clamp_min(1.0)
        loss = loss.sum() / denom

        return (loss, outputs) if return_outputs else loss


def make_training_args(args):
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "optim": args.optim,
        "fp16": not use_bf16,
        "bf16": use_bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": args.lr_scheduler_type,
        "remove_unused_columns": False,
        "report_to": "none",
    }
    if args.eval_jsonl and Path(args.eval_jsonl).exists():
        sig = inspect.signature(TrainingArguments.__init__)
        if "eval_strategy" in sig.parameters:
            kwargs["eval_strategy"] = "steps"
        else:
            kwargs["evaluation_strategy"] = "steps"
        kwargs["eval_steps"] = args.eval_steps
    return TrainingArguments(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-eval", type=int)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable ratio: {100 * trainable / total:.4f}%")

    train_dataset = WeldLlavaDataset(Path(args.train_jsonl), args.limit_train)
    eval_dataset = None
    if args.eval_jsonl and Path(args.eval_jsonl).exists():
        eval_dataset = WeldLlavaDataset(Path(args.eval_jsonl), args.limit_eval)
        if len(eval_dataset) == 0:
            eval_dataset = None

    trainer = WeightedLlavaTrainer(
        model=model,
        args=make_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=LlavaSFTCollator(processor),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
