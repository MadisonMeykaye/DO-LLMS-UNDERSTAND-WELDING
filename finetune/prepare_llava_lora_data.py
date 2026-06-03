from __future__ import annotations

import argparse
import json
import random
from collections import Counter
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
    "Is this weld acceptable for {context}?\n"
    'Answer ONLY with:\n'
    '{{"acceptable": true}}\n'
    "or\n"
    '{{"acceptable": false}}'
)


def load_json(path: Path):
    with path.open("rt", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


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
    answer = json.dumps({"acceptable": acceptable}, ensure_ascii=False)

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


def split_guids(guids: list[str], eval_ratio: float, seed: int):
    guids = list(guids)
    random.Random(seed).shuffle(guids)
    n_eval = max(1, round(len(guids) * eval_ratio)) if eval_ratio > 0 else 0
    eval_guids = sorted(guids[:n_eval])
    train_guids = sorted(guids[n_eval:])
    return train_guids, eval_guids


def guid_records(all_guids: list[dict], selected: list[str]) -> list[dict]:
    selected_set = set(selected)
    return [item for item in all_guids if item["guid"] in selected_set]


def add_soft_weights(rows: list[dict], beta: float = 0.25):
    """
    更温和的 soft weighting：
    - 不做 1:1
    - 不强推 positive
    - 重点缓解 minority collapse
    """

    from collections import Counter

    ctx_cnt = Counter(r["context"] for r in rows)
    cls_cnt = Counter((r["context"], r["acceptable"]) for r in rows)

    for r in rows:
        context = r["context"]
        y = r["acceptable"]

        total = ctx_cnt[context]
        n = cls_cnt[(context, y)]

        # inverse-frequency
        w = (total / n) ** beta

        # 关键：
        # positive 不允许权重太大
        if y is True:
            w = min(w, 1.5)

        # negative 可以稍微高一点
        else:
            w = min(w, 2.5)

        r["sample_weight"] = float(max(1.0, w))

    return rows


def build_rows(data_dir: Path, all_guids: list[dict], guids: list[str]) -> list[dict]:
    rows = []
    for guid in guids:
        for context in CONTEXTS:
            rows.append(make_example(data_dir, all_guids, guid, context))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--outdir",
        default="finetune/data/llava-realworld",
        help="Output directory for jsonl/json split files.",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-source",
        choices=["web", "private"],
        default="web",
        help="Source split used for training.",
    )
    parser.add_argument(
        "--test-source",
        choices=["web", "private"],
        default="private",
        help="Source split used for held-out testing.",
    )
    parser.add_argument(
        "--soft-weight-beta",
        type=float,
        default=0.5,
        help="Inverse-frequency smoothing exponent for sample weights.",
    )
    parser.add_argument(
        "--soft-weight-max",
        type=float,
        default=3.0,
        help="Maximum per-sample weight after clipping.",
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

    if args.train_source == "web":
        train_pool = sorted(web_guids)
        train_guids, dev_guids = split_guids(train_pool, args.eval_ratio, args.seed)
        test_guids = sorted(private_guids) if args.test_source == "private" else []
    else:
        train_pool = sorted(private_guids)
        train_guids, dev_guids = split_guids(train_pool, args.eval_ratio, args.seed)
        test_guids = sorted(web_guids) if args.test_source == "web" else []

    train_rows = add_soft_weights(
        build_rows(data_dir, all_guids, train_guids),
        beta=args.soft_weight_beta,
        #max_w=args.soft_weight_max,
    )
    dev_rows = build_rows(data_dir, all_guids, dev_guids)
    test_rows = build_rows(data_dir, all_guids, test_guids)

    # Compatibility aliases for older Makefile / scripts.
    write_jsonl(outdir / "train.jsonl", train_rows)
    write_jsonl(outdir / "dev.jsonl", dev_rows)
    write_jsonl(outdir / "test.jsonl", test_rows)
    write_jsonl(outdir / "web.jsonl", train_rows)
    write_jsonl(outdir / "private.jsonl", test_rows)

    dump_json(outdir / "train-guids.json", guid_records(all_guids, train_guids))
    dump_json(outdir / "dev-guids.json", guid_records(all_guids, dev_guids))
    dump_json(outdir / "test-guids.json", guid_records(all_guids, test_guids))
    dump_json(outdir / "web-guids.json", guid_records(all_guids, web_guids))
    dump_json(outdir / "private-guids.json", guid_records(all_guids, private_guids))
    dump_json(outdir / "all-guids.json", all_guids)

    print(f"train examples: {len(train_rows)} from {len(train_guids)} GUIDs")
    print(f"dev examples:   {len(dev_rows)} from {len(dev_guids)} GUIDs")
    print(f"test examples:  {len(test_rows)} from {len(test_guids)} GUIDs")

    train_counter = Counter((row["context"], row["acceptable"]) for row in train_rows)
    print("\nTrain distribution:")
    for k, v in sorted(train_counter.items()):
        print(k, v)

    if train_rows:
        weight_values = [float(r.get("sample_weight", 1.0)) for r in train_rows]
        print(
            "\nSample weight range:",
            f"min={min(weight_values):.3f}",
            f"max={max(weight_values):.3f}",
        )

    if skipped_private:
        print(f"skipped {len(skipped_private)} private GUIDs not in data/guids.json")
    if skipped_web:
        print(f"skipped {len(skipped_web)} web GUIDs not in data/guids.json")
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
