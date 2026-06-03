from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "scripts"))

from util import string2seed  # noqa: E402


CONTEXTS = {
    "RV & Marine": ("AcceptableRVMarine", "NarrativeRVMarine"),
    "Aeronautical": ("AcceptableAeronautical", "NarrativeAeronautical"),
    "Farming": ("AcceptableFarming", "NarrativeFarming"),
}

SYSTEM_PROMPT = (
    "You are an assistant whose task is to inspect images of welding joints "
    "and decide whether the weld is acceptable for a given application "
    "context."
)

BINARY_PROMPT = (
    "Is this weld acceptable for {context}?\n"
    "Answer only:\n"
    "acceptable\n"
    "or\n"
    "unacceptable"
)


def load_json(path: Path):
    with path.open("rt", encoding="utf-8") as f:
        return json.load(f)


def apply_chat_template(processor, messages, add_generation_prompt=True) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except TypeError:
        return processor.apply_chat_template(messages)


def image_path_for_guid(data_dir: Path, guid: str) -> Path:
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        path = data_dir / "pics" / f"{guid}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image found for {guid}")


def make_messages(context: str):
    prompt = f"{SYSTEM_PROMPT}\n\n{BINARY_PROMPT.format(context=context)}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        }
    ]


def parse_bool(text: str) -> bool:
    lowered = text.lower().strip()

    if lowered == "acceptable":
        return True
    if lowered == "unacceptable":
        return False

    if re.search(r"\bunacceptable\b", lowered):
        return False
    if re.search(r"\bacceptable\b", lowered):
        return True

    raise ValueError(f"Could not parse acceptability from: {text}")


def normalize_guid_items(items):
    if not items:
        return []
    if isinstance(items[0], str):
        return [{"guid": item} for item in items]
    return items


def resolve_guids_path(args) -> Path:
    if args.guids:
        return Path(args.guids)

    candidates = [
        Path(args.data_dir) / "private-guids.json",
        Path(args.data_dir) / "guids_private.json",
        Path(args.data_dir) / "test-guids.json",
        Path(args.data_dir) / "guids.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate a GUID list to evaluate.")


def load_model(model_name: str, adapter_dir: str | None):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(adapter_dir or model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return processor, model


@torch.inference_mode()
def generate_one(
    processor,
    model,
    image: Image.Image,
    context: str,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt = apply_chat_template(processor, make_messages(context), True)
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to("cuda")

    prompt_len = inputs["input_ids"].shape[1]
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": 1,
    }
    if temperature > 0:
        generate_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        generate_kwargs.update(do_sample=False)

    output = model.generate(**inputs, **generate_kwargs)
    return processor.decode(output[0][prompt_len:], skip_special_tokens=True).strip()


@torch.inference_mode()
def score_true_false(processor, model, image: Image.Image, context: str) -> float:
    """
    Optional diagnostic: positive logit margin for 'true' vs 'false'.
    Larger values mean stronger preference for acceptable=true.
    """
    prompt = apply_chat_template(processor, make_messages(context), True)
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to("cuda")

    outputs = model(**inputs)
    logits = outputs.logits[0, -1]

    true_ids = processor.tokenizer(" true", add_special_tokens=False).input_ids
    false_ids = processor.tokenizer(" false", add_special_tokens=False).input_ids
    if not true_ids or not false_ids:
        return float("nan")

    true_logit = logits[true_ids[-1]].item()
    false_logit = logits[false_ids[-1]].item()
    return true_logit - false_logit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--guids")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-logit-diff", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    guids_path = resolve_guids_path(args)
    guid_items = normalize_guid_items(load_json(guids_path))
    if args.limit is not None:
        guid_items = guid_items[: args.limit]

    processor, model = load_model(args.model, args.adapter_dir)

    for item in guid_items:
        guid = item["guid"]
        out_file = outdir / f"{guid}.json"
        if out_file.exists() and not args.overwrite:
            print(f"skip {guid}")
            continue

        with Image.open(image_path_for_guid(data_dir, guid)) as im:
            image = im.convert("RGB")
            runs = []
            for run_n in range(args.num_runs):
                row = {"guid": guid, "run_n": run_n}
                for context, (label_key, narrative_key) in CONTEXTS.items():
                    seed = string2seed(
                        f"LORA_EVAL_{guid}_{context}_RUN_{run_n}"
                    )
                    raw = generate_one(
                        processor,
                        model,
                        image,
                        context,
                        seed,
                        args.max_new_tokens,
                        args.temperature,
                        args.top_p,
                    )
                    print(f"[{guid}] [{context}] -> {raw}")
                    try:
                        row[label_key] = parse_bool(raw)
                    except ValueError:
                        print(f"parse failed for {guid} {context}: {raw}")
                        row[label_key] = False
                    row[narrative_key] = raw

                    if args.save_logit_diff:
                        try:
                            row[f"{label_key}_logit_diff"] = score_true_false(
                                processor, model, image, context
                            )
                        except Exception as exc:
                            print(f"logit diff failed for {guid} {context}: {exc}")
                            row[f"{label_key}_logit_diff"] = None
                runs.append(row)

        with out_file.open("wt", encoding="utf-8") as f:
            json.dump(runs, f, indent=4, ensure_ascii=False)
        print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
