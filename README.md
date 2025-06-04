# ORFuzz
Official implement for paper "ORFUZZ: Fuzzing the "Other Side" of LLM Safety – Testing Over-Refusal".
Here we provide an example of how to use ORFuzz to generate over-refusal prompts.

## Example Usage

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Setup Judge Model Server
```bash
CUDA_VISIBLE_DEVICES="0" python src/server.py --host 127.0.0.1 --port {REPLACE_WITH_PORT}
```

### 3. Generate Over-Refusal Prompts
If you have more than one GPU, you can change the `CUDA_VISIBLE_DEVICES` environment variable to specify which GPU to use.
You need to complete the configurations of the gen_model in `config/keys`.
In this case, we use Deepseek as the model provider, so you need to complete the file `config/keys/deepseek_keys.yaml`.
We recommend using other providers if you fail to use Deepseek. In that case, you need to change the argument `--gen_model` to the corresponding model provider in `src/ORFuzz.py` and complete the corresponding keys file in `config/keys`.
For more configurations, please refer to the `src/config/config.py` directory or run the command with `--help`.
```bash
CUDA_VISIBLE_DEVICES="0" python src/ORFuzz.py --stream --server 127.0.0.1:{REPLACE_WITH_PORT} --gen_model deepseek --model_list llama3
```

## Released Resources
User study results: [here](ORFuzz_dataset/user_study.csv)

ORFuzz dataset: [here](ORFuzz_dataset/ORFuzz.csv)

Judge models: [here](model_finetuning)
