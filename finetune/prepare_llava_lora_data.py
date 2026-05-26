from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


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


def dump_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_path_for_guid(data_dir: Path, guid: str) -> str:
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        path = data_dir / "pics" / f"{guid}{ext}"
        if path.exists():
            return path.as_posix()
    raise FileNotFoundError(f"No image file found for GUID {guid}")


def class_for_guid(all_guids: list[dict], guid: str) -> str:
    for item in all_guids:
        if item["guid"] == guid:
            return item["class"]
    raise KeyError(f"GUID {guid} not found in data/guids.json")


def make_example(data_dir: Path, all_guids: list[dict], guid: str, context: str):
    data = load_json(data_dir / "data" / f"{guid}.json")
    label_key, narrative_key = CONTEXTS[context]
    acceptable = bool(data[label_key])
    prompt = f"{SYSTEM_PROMPT}\n\n{BINARY_PROMPT.format(context=context)}"
    answer = json.dumps({"acceptable": acceptable})

    return {
        "guid": guid,
        "class": class_for_guid(all_guids, guid),
        "context": context,
        "label_key": label_key,
        "narrative_key": narrative_key,
        "image_path": image_path_for_guid(data_dir, guid),
        "acceptable": acceptable,
        "answer": answer,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
    }


def split_private_guids(private_guids: list[str], eval_ratio: float, seed: int):
    guids = list(private_guids)
    random.Random(seed).shuffle(guids)
    n_eval = max(1, round(len(guids) * eval_ratio)) if eval_ratio > 0 else 0
    eval_guids = sorted(guids[:n_eval])
    train_guids = sorted(guids[n_eval:])
    return train_guids, eval_guids


def guid_records(all_guids: list[dict], selected: list[str]) -> list[dict]:
    selected_set = set(selected)
    return [item for item in all_guids if item["guid"] in selected_set]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--outdir", default="finetune/data/llava-realworld")
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-on-all-private",
        action="store_true",
        help="Use every private/Real-World sample for training.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    all_guids = load_json(data_dir / "guids.json")
    all_guid_set = {item["guid"] for item in all_guids}
    raw_private_guids = load_json(data_dir / "guids_private.json")
    raw_web_guids = load_json(data_dir / "guids_web.json")
    private_guids = [guid for guid in raw_private_guids if guid in all_guid_set]
    web_guids = [guid for guid in raw_web_guids if guid in all_guid_set]
    skipped_private = sorted(set(raw_private_guids) - all_guid_set)
    skipped_web = sorted(set(raw_web_guids) - all_guid_set)

    if args.train_on_all_private:
        train_guids = sorted(private_guids)
        dev_guids = []
    else:
        train_guids, dev_guids = split_private_guids(
            private_guids, args.eval_ratio, args.seed
        )

    def build_rows(guids: list[str]) -> list[dict]:
        rows = []
        for guid in guids:
            for context in CONTEXTS:
                rows.append(make_example(data_dir, all_guids, guid, context))
        return rows

    train_rows = build_rows(train_guids)
    dev_rows = build_rows(dev_guids)
    web_rows = build_rows(sorted(web_guids))

    write_jsonl(outdir / "train.jsonl", train_rows)
    write_jsonl(outdir / "dev.jsonl", dev_rows)
    write_jsonl(outdir / "web.jsonl", web_rows)
    dump_json(outdir / "train-guids.json", guid_records(all_guids, train_guids))
    dump_json(outdir / "dev-guids.json", guid_records(all_guids, dev_guids))
    dump_json(outdir / "web-guids.json", guid_records(all_guids, web_guids))
    dump_json(outdir / "private-guids.json", guid_records(all_guids, private_guids))
    dump_json(outdir / "all-guids.json", all_guids)

    print(f"train examples: {len(train_rows)} from {len(train_guids)} GUIDs")
    print(f"dev examples: {len(dev_rows)} from {len(dev_guids)} GUIDs")
    print(f"web examples: {len(web_rows)} from {len(web_guids)} GUIDs")
    if skipped_private:
        print(f"skipped {len(skipped_private)} private GUIDs not in data/guids.json")
    if skipped_web:
        print(f"skipped {len(skipped_web)} web GUIDs not in data/guids.json")
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
