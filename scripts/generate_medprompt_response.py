import json
from argparse import ArgumentParser
from copy import deepcopy

from apiclient import run_chat_messages
from config import CONFIG
from read_data import load_embedding_cot_data_pair, load_image_data_pair
from scipy import spatial
from util import obj2base85json, string2seed
from formats import Acceptability

if __name__ != "__main__":
    raise NotImplementedError("cannot be used as a module")

parser = ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--guid", required=True)
parser.add_argument("--cotdata", required=True)
parser.add_argument("--guids", required=True)
parser.add_argument("--model", default="gpt-4o")
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
parser.add_argument(
    "--mmr-lambda",
    default=0.7,
    type=float,
    help="Similarity-vs-diversity weight used by diverse selectors.",
)
args = parser.parse_args()

if args.k <= 0:
    raise ValueError("--k must be a positive integer")
if args.mmr_lambda < 0 or args.mmr_lambda > 1:
    raise ValueError("--mmr-lambda must be between 0 and 1")


def make_init_messages(context):
    messages = [
        {
            "role": "system",
            "content": CONFIG["medprompt_responses"][
                "generation_system_prompt"
            ].format(context=context),
        },
    ]
    return messages


def make_cot_injection_message(context, cot_data):
    formatted_cots = "\n".join(
        [
            "\n".join(
                [
                    (c[0] + "\n" + "Acceptable: ")
                    + ("Yes" if c[1] else "No")
                    for c in d["cots"]
                ]
            )
            for d in cot_data
        ]
    )
    image_b64s = [load_image_data_pair(d["guid"])[0] for d in cot_data]

    msg_text = (
        CONFIG["medprompt_responses"][
            "medprompt_cot_injection_prompt"
        ].format(context=context)
        + "\n\n"
        + formatted_cots
    )

    img_token_list = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{imageb64}"},
        }
        for imageb64 in image_b64s
    ]

    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": msg_text,
            }
        ]
        + img_token_list,
    }


def make_picture_cot_message(imageb64, context):
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": CONFIG["medprompt_responses"][
                    "cot_generation_prompt"
                ].format(context=context),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{imageb64}"},
            },
        ],
    }


def make_binary_response_message(context):
    return {
        "role": "user",
        "content": CONFIG["medprompt_responses"][
            "binary_generation_prompt"
        ].format(context=context),
    }


def make_json_repair_message():
    return {
        "role": "user",
        "content": (
            'Your previous answer did not match the required schema. '
            'Return only one JSON object with exactly one boolean key: '
            '{"acceptable": true} or {"acceptable": false}.'
        ),
    }


def extract_json_objects(text):
    decoder = json.JSONDecoder()
    objects = []
    start = 0
    while True:
        start = text.find("{", start)
        if start == -1:
            return objects
        try:
            obj, end = decoder.raw_decode(text[start:])
            objects.append(obj)
            start += end
        except json.JSONDecodeError:
            start += 1


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"yes", "true", "acceptable", "accepted"}:
            return True
        if value in {
            "no",
            "false",
            "unacceptable",
            "not acceptable",
            "rejected",
        }:
            return False
    raise ValueError(f"Cannot coerce value to bool: {value}")


def parse_acceptability_response(raw):
    for obj in extract_json_objects(raw):
        if not isinstance(obj, dict):
            continue
        for key in [
            "acceptable",
            "is_acceptable",
            "acceptability",
            "isAcceptable",
            "answer",
            "classification",
            "label",
        ]:
            if key in obj:
                return Acceptability(acceptable=coerce_bool(obj[key]))
    raise ValueError(f"No acceptable boolean found in response: {raw}")


def run_binary_response(messages, model, seed):
    repair_messages = list(messages)
    last_error = None
    for attempt in range(3):
        raw_response = run_chat_messages(
            messages=repair_messages, seed=seed, model=model
        )
        try:
            return raw_response, parse_acceptability_response(raw_response)
        except Exception as exc:
            last_error = exc
            print("FAILED JSON DECODE")
            print(raw_response)
            if attempt == 2:
                break
            repair_messages = (
                repair_messages
                + [{"role": "assistant", "content": raw_response}]
                + [make_json_repair_message()]
            )
    raise last_error


def cosine_sim(a, b):
    return 1 - spatial.distance.cosine(a, b)


def point_label(point):
    votes = [cot[1] for cot in point["cots"]]
    return sum(votes) >= len(votes) / 2


def get_candidates(ds, ctx, cotdata, reference_scope):
    if reference_scope == "same-dataset":
        return list(cotdata.get(ctx, {}).get(ds, []))

    candidates = []
    for dataset_points in cotdata.get(ctx, {}).values():
        candidates.extend(dataset_points)
    return candidates


def score_candidates(emb, candidates):
    scored = []
    for point in candidates:
        c_point = deepcopy(point)
        c_point["_query_similarity"] = cosine_sim(emb, point["embedding"])
        c_point["_label"] = point_label(point)
        scored.append(c_point)
    return sorted(
        scored, key=lambda point: point["_query_similarity"], reverse=True
    )


def mmr_score(emb, point, selected, mmr_lambda):
    if len(selected) == 0:
        return point["_query_similarity"]

    max_selected_sim = max(
        cosine_sim(point["embedding"], selected_point["embedding"])
        for selected_point in selected
    )
    return (
        mmr_lambda * point["_query_similarity"]
        - (1 - mmr_lambda) * max_selected_sim
    )


def pop_best_mmr(emb, candidates, selected, mmr_lambda):
    best_i = None
    best_score = float("-Inf")
    for i, point in enumerate(candidates):
        score = mmr_score(emb, point, selected, mmr_lambda)
        if score > best_score:
            best_i = i
            best_score = score

    assert best_i is not None
    return candidates.pop(best_i)


def best_available_label(candidates, selected):
    labels = {point["_label"] for point in candidates}
    if len(labels) < 2:
        return None

    counts = {
        True: sum(1 for point in selected if point["_label"] is True),
        False: sum(1 for point in selected if point["_label"] is False),
    }
    if counts[True] == counts[False]:
        return candidates[0]["_label"]
    return True if counts[True] < counts[False] else False


def select_balanced(candidates, k):
    candidates = list(candidates)
    selected = []

    while candidates and len(selected) < k:
        target_label = best_available_label(candidates, selected)
        if target_label is None:
            selected.append(candidates.pop(0))
            continue

        best_i = next(
            i
            for i, point in enumerate(candidates)
            if point["_label"] == target_label
        )
        selected.append(candidates.pop(best_i))

    return selected


def select_balanced_diverse(emb, candidates, k, mmr_lambda):
    candidates = list(candidates)
    selected = []

    while candidates and len(selected) < k:
        target_label = best_available_label(candidates, selected)
        if target_label is None:
            selected.append(
                pop_best_mmr(emb, candidates, selected, mmr_lambda)
            )
            continue

        label_candidates = [
            (i, point)
            for i, point in enumerate(candidates)
            if point["_label"] == target_label
        ]
        best_i = max(
            label_candidates,
            key=lambda item: mmr_score(emb, item[1], selected, mmr_lambda),
        )[0]
        selected.append(candidates.pop(best_i))

    return selected


def clean_selection_metadata(points):
    cleaned = []
    for point in points:
        c_point = deepcopy(point)
        c_point.pop("_label", None)
        c_point.pop("_query_similarity", None)
        cleaned.append(c_point)
    return cleaned


def describe_selection(points):
    return [
        {
            "guid": point["guid"],
            "acceptable": bool(point["_label"]),
            "similarity": point["_query_similarity"],
        }
        for point in points
    ]


def find_k_cots(
    emb,
    ds,
    ctx,
    cotdata,
    k=5,
    selection_strategy="similarity",
    reference_scope="same-dataset",
    mmr_lambda=0.7,
):
    candidates = get_candidates(ds, ctx, cotdata, reference_scope)
    candidates = score_candidates(emb, candidates)
    if not candidates:
        raise ValueError(f"No candidates found for context={ctx}, ds={ds}")

    if len(candidates) < k:
        print(
            f"Warn! Requested k={k}, but only {len(candidates)} "
            f"candidates are available for context={ctx}, ds={ds}"
        )

    if selection_strategy == "similarity":
        selected = candidates[:k]
    elif selection_strategy == "diverse":
        c_candidates = list(candidates)
        selected = []
        while c_candidates and len(selected) < k:
            selected.append(
                pop_best_mmr(emb, c_candidates, selected, mmr_lambda)
            )
    elif selection_strategy == "balanced":
        selected = select_balanced(candidates, k)
    elif selection_strategy == "balanced-diverse":
        selected = select_balanced_diverse(emb, candidates, k, mmr_lambda)
    else:
        raise ValueError(f"Unknown selection strategy {selection_strategy}")

    return clean_selection_metadata(selected), describe_selection(selected)


def find_guid_dataset(guid, guids):
    for guid_ds in guids:
        c_guid = guid_ds["guid"]
        dataset = guid_ds["class"]
        if c_guid == guid:
            return dataset
    raise ValueError(f"GUID {guid} was not found")


def drop_guid(guid, ctx, data):
    for ds_name, ds in data[ctx].items():
        for i in range(len(ds)):
            if ds[i]["guid"] == guid:
                ds.pop(i)
                return
    raise ValueError(f"GUID {guid} was not found")


with open(args.guids, "rt", encoding="utf-8") as gf:
    guids = json.load(gf)

with open(args.cotdata, "rt", encoding="utf-8") as cotfile:
    cot_data = json.load(cotfile)

imageb64, data = load_image_data_pair(args.guid)
_, input_emb = load_embedding_cot_data_pair(args.guid)
input_ds = find_guid_dataset(args.guid, guids)

ctxs = list(CONFIG["zero_shot_responses"]["contexts"].keys())
for ctx in ctxs:
    try:
        drop_guid(args.guid, ctx, cot_data)
    except ValueError:
        print("Warn! Error dropping GUID", args.guid)

all_responses_out = []

for run_n in range(CONFIG["medprompt_responses"]["n_responses"]):
    out_dict = {
        "seed_strs": [],
        "seed_nums": [],
        "message_dumps": [],
        "selected_examples": [],
        "selection_params": {
            "k": args.k,
            "selection_strategy": args.selection_strategy,
            "reference_scope": args.reference_scope,
            "mmr_lambda": args.mmr_lambda,
        },
        "guid": args.guid,
        "run_n": run_n,
    }
    for context_name, (binary_column, narrative_column) in CONFIG[
        "zero_shot_responses"
    ]["contexts"].items():
        c_cot_data = deepcopy(cot_data)
        emb_guids, selection_metadata = find_k_cots(
            input_emb,
            input_ds,
            context_name,
            c_cot_data,
            k=args.k,
            selection_strategy=args.selection_strategy,
            reference_scope=args.reference_scope,
            mmr_lambda=args.mmr_lambda,
        )
        # Assert that there's no data leak
        exact_emb_guids = set([g["guid"] for g in emb_guids])
        assert args.guid not in exact_emb_guids
        out_dict["selected_examples"].append(
            {context_name: selection_metadata}
        )

        # Generate seed
        seed_str = CONFIG["medprompt_responses"]["generation_seed"].format(
            guid=args.guid, run=run_n, context=context_name
        )
        seed = string2seed(seed_str)

        # Inject MedPrompt CoT
        messages = make_init_messages(context_name)
        messages.append(
            make_cot_injection_message(context_name, emb_guids)
        )
        messages.append(
            {
                "role": "assistant",
                "content": run_chat_messages(
                    messages=messages, seed=seed, model=args.model
                ),
            }
        )

        # Get CoT Response

        messages.append(make_picture_cot_message(imageb64, context_name))
        cot_response = run_chat_messages(
            messages=messages, seed=seed, model=args.model
        )
        messages.append({"role": "assistant", "content": cot_response})

        # Get Binary Response
        messages.append(make_binary_response_message(context_name))
        raw_binary_response, parsed_binary_response = run_binary_response(
            messages=messages, seed=seed, model=args.model
        )
        messages.append(
            {"role": "assistant", "content": raw_binary_response}
        )
        binary_response = parsed_binary_response.acceptable

        out_dict["seed_strs"].append({context_name: seed_str})
        out_dict["seed_nums"].append({context_name: seed})

        out_dict[binary_column] = binary_response
        out_dict[narrative_column] = cot_response

        out_dict["message_dumps"].append(
            {context_name: obj2base85json(messages)}
        )

    all_responses_out.append(out_dict)

with open(args.out, "wt", encoding="utf-8") as outfile:
    json.dump(all_responses_out, outfile, indent=4)
