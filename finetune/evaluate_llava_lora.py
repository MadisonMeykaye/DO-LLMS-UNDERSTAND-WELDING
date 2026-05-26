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
    "Given this image of a weld, decide whether this joint is acceptable in "
    "the context of {context}. Return only a valid JSON object with a single "
    'boolean key called "acceptable".'
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
    matches = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for match in matches:
        try:
            obj = json.loads(match)
        except json.JSONDecodeError:
            continue
        for key in ["acceptable", "is_acceptable", "Acceptable"]:
            if key in obj:
                return bool(obj[key])

    lowered = text.lower()
    if re.search(r"\bnot acceptable\b|\bunacceptable\b|\bfalse\b|\bno\b", lowered):
        return False
    if re.search(r"\bacceptable\b|\btrue\b|\byes\b", lowered):
        return True
    raise ValueError(f"Could not parse acceptability from: {text}")


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
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p,
        num_beams=1,
    )
    return processor.decode(output[0][prompt_len:], skip_special_tokens=True).strip()


def normalize_guid_items(items):
    if not items:
        return []
    if isinstance(items[0], str):
        return [{"guid": item} for item in items]
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--guids", default="data/guids.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    guid_items = normalize_guid_items(load_json(Path(args.guids)))
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
                    try:
                        row[label_key] = parse_bool(raw)
                    except ValueError:
                        print(f"parse failed for {guid} {context}: {raw}")
                        row[label_key] = False
                    row[narrative_key] = raw
                runs.append(row)

        with out_file.open("wt", encoding="utf-8") as f:
            json.dump(runs, f, indent=4)
        print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
