from argparse import ArgumentParser

from flask import Flask, jsonify, request
from hugging_face_vlm import HuggingFaceChatModel
from server_class import OpenAIMessageDecoder

parser = ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument(
    "--adapter-dir",
    help="Optional PEFT/LoRA adapter directory to load on top of --model.",
)
parser.add_argument(
    "--noprompt",
    action="store_true",
    help="Deprecated; prompt removal is now the default.",
)
parser.add_argument(
    "--keep-prompt",
    action="store_true",
    help="Return the full prompt plus generated text. Do not use for experiments.",
)
parser.add_argument("--port", default=56873, type=int)
ARGS = parser.parse_args()

MODEL = HuggingFaceChatModel.get_class(ARGS.model)(
    ARGS.model,
    adapter_dir=ARGS.adapter_dir,
    delete_prompt=not ARGS.keep_prompt,
)
DECODER = OpenAIMessageDecoder()

app = Flask(__name__)


@app.route("/api/generate", methods=["POST"])
def generate():
    content = request.get_json()
    if ARGS.model != content["params"]["model"]:
        raise AssertionError(
            "wrong model used, expected {}".format(ARGS.model)
        )
    msgs, imgs = DECODER.preprocess_chat(content["messages"])
    output = MODEL.run_messages(msgs, imgs, json=content["params"]["json"])
    return jsonify({"content": output})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=ARGS.port, debug=False, use_reloader=False)
