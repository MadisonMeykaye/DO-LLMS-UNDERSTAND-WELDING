import json
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def default_report_path(outdir, suffix):
    out_path = Path(outdir)
    return out_path.parent / f"{out_path.name}-{suffix}"


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def run_with_retries(cmd, retries, sleep):
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt >= retries:
                raise
            print(f"Retry {attempt} failed, sleeping {sleep}s...")
            time.sleep(sleep)


def load_guids(path, limit):
    with open(path, "rt", encoding="utf-8") as guid_file:
        guids = json.load(guid_file)
    guids = [item["guid"] if isinstance(item, dict) else item for item in guids]
    if limit is not None:
        guids = guids[:limit]
    return guids


parser = ArgumentParser()
parser.add_argument("--model", default="gpt-4o")
parser.add_argument("--guids", default="data/guids.json")
parser.add_argument("--cotdata", default="results/cot_data.json")
parser.add_argument("--outdir", required=True)
parser.add_argument("--k", default=5, type=int)
parser.add_argument(
    "--selection-strategy",
    default="similarity",
    choices=["similarity", "diverse", "balanced", "balanced-diverse"],
)
parser.add_argument(
    "--reference-scope",
    default="same-dataset",
    choices=["same-dataset", "all"],
)
parser.add_argument("--mmr-lambda", default=0.7, type=float)
parser.add_argument("--sleep", default=1, type=float)
parser.add_argument("--retries", default=3, type=int)
parser.add_argument("--limit", type=int)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--class-report")
parser.add_argument("--no-class-report", action="store_true")
parser.add_argument("--make-embeddings", action="store_true")
parser.add_argument("--embeddings-dir")
parser.add_argument("--truth-embeddings", default="results/data-embeddings")
parser.add_argument("--dist-report")
args = parser.parse_args()


guids_path = resolve_path(args.guids)
cotdata_path = resolve_path(args.cotdata)
outdir = resolve_path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

guids = load_guids(guids_path, args.limit)
print(
    "Running WeldPrompt variant:",
    f"strategy={args.selection_strategy}",
    f"k={args.k}",
    f"scope={args.reference_scope}",
    f"n={len(guids)}",
)

for guid in guids:
    outfile = outdir / f"{guid}.json"
    if outfile.exists() and not args.overwrite:
        print(f"Skipping existing response {guid}")
        continue

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "generate_medprompt_response.py"),
        "--guid",
        guid,
        "--model",
        args.model,
        "--out",
        str(outfile),
        "--guids",
        str(guids_path),
        "--cotdata",
        str(cotdata_path),
        "--k",
        str(args.k),
        "--selection-strategy",
        args.selection_strategy,
        "--reference-scope",
        args.reference_scope,
        "--mmr-lambda",
        str(args.mmr_lambda),
    ]
    print(f"Generating {guid}")
    run_with_retries(cmd, args.retries, args.sleep)
    time.sleep(args.sleep)

if not args.no_class_report:
    class_report = (
        resolve_path(args.class_report)
        if args.class_report
        else default_report_path(outdir, "class-report.xlsx")
    )
    class_report.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "test_classification.py"),
        "--guids",
        str(guids_path),
        "--truth",
        str(REPO_ROOT / "data/data"),
        "--pred",
        str(outdir),
        "--out",
        str(class_report),
    ]
    print(f"Writing classification report {class_report}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

if args.make_embeddings:
    embeddings_dir = (
        resolve_path(args.embeddings_dir)
        if args.embeddings_dir
        else default_report_path(outdir, "embeddings")
    )
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    for guid in guids:
        embedding_file = embeddings_dir / f"{guid}.json"
        if embedding_file.exists() and not args.overwrite:
            print(f"Skipping existing embedding {guid}")
            continue
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "create_embeddings.py"),
            "--input",
            str(outdir / f"{guid}.json"),
            "--out",
            str(embedding_file),
        ]
        print(f"Embedding {guid}")
        run_with_retries(cmd, args.retries, args.sleep)
        time.sleep(args.sleep)

    dist_report = (
        resolve_path(args.dist_report)
        if args.dist_report
        else default_report_path(outdir, "dist-report.xlsx")
    )
    dist_report.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "test_distances.py"),
        "--guids",
        str(guids_path),
        "--truth",
        str(resolve_path(args.truth_embeddings)),
        "--pred",
        str(embeddings_dir),
        "--out",
        str(dist_report),
    ]
    print(f"Writing distance report {dist_report}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
