import time
import re
import threading

from config import CONFIG
from openai import OpenAI
from openai import RateLimitError
import json as _json
from urllib import request as _urlrequest

client = OpenAI(api_key=CONFIG["secrets"]["openai_key"])

external_model_api = CONFIG["remotemodels"]["api_url"]
external_client = OpenAI(api_key="-", base_url=external_model_api)


def run_openai_chat_messages(
    messages, model="gpt-4o", seed=None, json=False
):
    local_client = client
    if not model.lower().startswith("gpt"):
        local_client = external_client

    chat_funtion = local_client.chat.completions.create
    parsing = False

    if isinstance(json, bool):
        fmt = {"type": "json_object"} if json else None
    else:
        fmt = json
        chat_funtion = local_client.beta.chat.completions.parse
        parsing = True

    retry = 0
    max_retry = 20

    while True:
        try:
            rate_limit()
            response = chat_funtion(
                model=model,
                messages=messages,
                seed=seed,
                response_format=fmt,
            )
            break

        except RateLimitError as e:
            retry += 1

            msg = str(e)
            wait = 30

            # 解析 API 返回的 retry 时间
            m = re.search(r"try again in ([0-9.]+)s", msg)
            if m:
                wait = float(m.group(1)) + 1
            else:
                # 指数退避
                wait = min(30 * (2 ** (retry - 1)), 120)

            print(f"Rate limit hit, sleeping {wait:.1f}s...")

            time.sleep(wait)

            if retry >= max_retry:
                raise RuntimeError("Too many rate limit retries")

    choice = response.choices[0]

    assert choice.finish_reason == "stop", f"Finish reason: {choice.finish_reason}"

    msg = choice.message

    if parsing:
        return msg.content, msg.parsed

    return msg.content

def _remote_endpoint():
    base = CONFIG["remotemodels"]["api_url"].rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    if base.endswith("/api/generate"):
        return base
    return base + "/api/generate"

def _extract_json_object(text):
    objects = _extract_json_objects(text)
    return objects[0] if objects else None


def _extract_json_objects(text):
    if text is None:
        return []
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:].strip()
    decoder = _json.JSONDecoder()
    objects = []
    start = 0
    while True:
        start = t.find("{", start)
        if start == -1:
            break
        try:
            _, end = decoder.raw_decode(t[start:])
            objects.append(t[start:start + end])
            start = start + end
        except _json.JSONDecodeError:
            start += 1
    if len(objects) == 0:
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            objects.append(t[start:end + 1])
    return objects


def _coerce_json_for_schema(data, parse_class):
    field_names = set(getattr(parse_class, "model_fields", {}).keys())
    if "acceptable" not in field_names or not isinstance(data, dict):
        return data

    for key in [
        "acceptable",
        "is_acceptable",
        "acceptability",
        "isAcceptable",
        "answer",
        "classification",
        "label",
    ]:
        if key in data:
            value = data[key]
            if isinstance(value, str):
                value_l = value.strip().lower()
                if value_l in {"yes", "true", "acceptable", "accepted"}:
                    value = True
                elif value_l in {
                    "no",
                    "false",
                    "unacceptable",
                    "not acceptable",
                    "rejected",
                }:
                    value = False
            return {"acceptable": value}

    return data


def run_remote_chat_messages(messages, model, seed=None, json=False):
    # LLaVA server only needs messages + params
    if isinstance(json, bool):
        json_flag = json
        parse_class = None
    else:
        json_flag = True
        parse_class = json

    payload = {
        "messages": messages,
        "params": {
            "model": model,
            "json": json_flag,
        },
    }
    data = _json.dumps(payload).encode("utf-8")
    req = _urlrequest.Request(
        _remote_endpoint(),
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with _urlrequest.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
    resp_obj = _json.loads(body)
    raw = resp_obj.get("content", "")

    if parse_class is not None:
        parse_errors = []
        for candidate in [raw] + _extract_json_objects(raw):
            try:
                data = _json.loads(candidate)
                data = _coerce_json_for_schema(data, parse_class)
                parsed = parse_class(**data)
                return raw, parsed
            except Exception as exc:
                parse_errors.append(str(exc))
        raise ValueError(
            "Invalid JSON from model: "
            f"{raw}\nParse errors: {' | '.join(parse_errors)}"
        )

    return raw



def run_chat_messages(messages, model="gpt-4o", seed=None, json=False):
    if model.lower().startswith("gpt"):
        return run_openai_chat_messages(messages, model, seed, json)
    return run_remote_chat_messages(messages, model, seed, json)



# From: https://platform.openai.com/docs/guides/embeddings/use-cases
def generate_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ").strip()

    retry = 0

    while True:
        try:
            emb = (
                client.embeddings.create(
                    input=[text],
                    model=model
                )
                .data[0]
                .embedding
            )
            return emb

        except RateLimitError as e:

            retry += 1
            msg = str(e)
            wait = 30

            m = re.search(r"try again in ([0-9.]+)s", msg)
            if m:
                wait = float(m.  group(1)) + 1
            else:
                wait = min(30 * (2 ** (retry - 1)), 120)

            print(f"Embedding rate limit hit, sleeping {wait:.1f}s...")

            time.sleep(wait) 

            if retry >= 20:
                raise RuntimeError("Too many embedding retries")  
            


# 👉 控制全局请求频率（核心参数）
MIN_INTERVAL = 0.5   # 每0.5秒最多1次（≈2 QPS）

_lock = threading.Lock()
_last_time = 0


def rate_limit():
    global _last_time
    with _lock:
        now = time.time()
        wait = MIN_INTERVAL - (now - _last_time)
        if wait > 0:
            time.sleep(wait)
        _last_time = time.time()
