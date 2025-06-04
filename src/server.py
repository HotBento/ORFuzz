import torch
from flask import Flask, request, jsonify
from eval import QwenAnswerEvaluator, QwenToxicEvaluator
import threading
import queue
import time
import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers import QuantoConfig
import argparse
from config import ServerConfig

app = Flask(__name__)

def process_requests():
    while True:
        try:
            request_id, evaluator_type, data = request_queue.get()
            if evaluator_type == "answer":
                inputs = data["inputs"]
                outputs = data["outputs"]
                th = data["th"]
                results, predictions = qwen_answer_evaluator.evaluate(inputs, outputs, th=th)
            elif evaluator_type == "toxic":
                inputs = data["inputs"]
                th = data["th"]
                results, predictions = qwen_toxic_evaluator.evaluate(inputs, th=th)
            else:
                results, predictions = None, None

            response_dict[request_id] = {"results": results, "predictions": predictions}
        except Exception as e:
            response_dict[request_id] = {"error": str(e)}
        finally:
            request_queue.task_done()



@app.route('/evaluate/answer', methods=['POST'])
def evaluate_answer():
    try:
        data = request.json
        inputs = data.get("inputs", [])
        outputs = data.get("outputs", [])
        th = data.get("th", 0.5)

        if not inputs or not outputs:
            return jsonify({"error": "inputs and outputs are required"}), 400

        request_id = f"{next(request_counter)}_answer"
        request_queue.put((request_id, "answer", {"inputs": inputs, "outputs": outputs, "th": th}))

        while request_id not in response_dict:
            time.sleep(0.1)

        return jsonify(response_dict.pop(request_id))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate/toxic', methods=['POST'])
def evaluate_toxic():
    try:
        data = request.json
        inputs = data.get("inputs", [])
        th = data.get("th", 0.5)

        if not inputs:
            return jsonify({"error": "inputs are required"}), 400

        request_id = f"{next(request_counter)}_toxic"
        request_queue.put((request_id, "toxic", {"inputs": inputs, "th": th}))

        while request_id not in response_dict:
            time.sleep(0.1)

        return jsonify(response_dict.pop(request_id))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    config = ServerConfig.get_config()

    request_counter = itertools.count()

    quantization_config = QuantoConfig(weights="int8")
    answer_model:PreTrainedModel = AutoModelForCausalLM.from_pretrained("model_finetuning/lora/qwen_answer", torch_dtype="auto",quantization_config=quantization_config, device_map="cuda:0", attn_implementation="eager").eval()
    toxic_model:PreTrainedModel = AutoModelForCausalLM.from_pretrained("model_finetuning/lora/qwen_answer", torch_dtype="auto",quantization_config=quantization_config, device_map="cuda:0", attn_implementation="eager").eval()
    eval_tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained(config.model_path, padding_side="left")

    qwen_answer_evaluator = QwenAnswerEvaluator()
    qwen_answer_evaluator.model = answer_model
    qwen_answer_evaluator.tokenizer = eval_tokenizer
    qwen_toxic_evaluator = QwenToxicEvaluator()
    qwen_toxic_evaluator.model = toxic_model
    qwen_toxic_evaluator.tokenizer = eval_tokenizer


    request_queue = queue.Queue()
    response_dict = {}
    threading.Thread(target=process_requests, daemon=True).start()
    app.run(host=config.host, port=config.port)