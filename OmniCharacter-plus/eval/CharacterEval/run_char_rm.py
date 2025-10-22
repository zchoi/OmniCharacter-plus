import os
import json
import torch
from tqdm import tqdm
from multiprocessing import Process
from BaichuanCharRM.modeling_baichuan import BaichuanCharRM
from BaichuanCharRM.tokenization_baichuan import BaichuanTokenizer

max_seq_length = 4096
reward_model_path = 'BaichuanCharRM/'
num_gpus = 8  # 自动检测 GPU 数量

with open("data/character_profiles.json", "r") as f:
    character_profile = json.load(f)
with open("results/generation_trans.jsonl", "r", encoding='utf-8') as f:
    records = json.load(f)

def chunk_data(data, n_chunks):
    chunk_size = (len(data) + n_chunks - 1) // n_chunks
    return [data[i*chunk_size : (i+1)*chunk_size] for i in range(n_chunks)]

def format_input(example, character_profile):
    input_text = "<RoleInfo>\n\n" \
        + str(character_profile[example['role']]) + "\n\n<Context>\n\n" \
        + example['context'] + "\n\n<Response>\n\n" \
        + example['model_output'] + "\n\n<Dimension>\n\n" + example["metric_zh"]
    return input_text

def run_eval_worker(gpu_id, data_chunk, character_profile, reward_model_path, max_seq_length, output_path):
    
    tokenizer = BaichuanTokenizer.from_pretrained(reward_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = BaichuanCharRM.from_pretrained(reward_model_path, torch_dtype=torch.bfloat16).cuda()
    model.eval()

    results = []
    for record in tqdm(data_chunk, desc=f"GPU {gpu_id}"):
        input_text = format_input(record, character_profile)
        input_ids = tokenizer.encode(input_text, add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[-max_seq_length:]
        input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
        with torch.no_grad():
            score = model(input_ids=input_ids)[1].item() * 4 + 1
        record[record['metric_en']] = score
        results.append(record)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    data_chunks = chunk_data(records, num_gpus)

    processes = []
    for i in range(num_gpus):
        out_path = f"results/eval_gpu{i}.jsonl"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
        p = Process(target=run_eval_worker, args=(
            i, data_chunks[i], character_profile, reward_model_path, max_seq_length, out_path
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    final = []
    for i in range(num_gpus):
        path = f"results/eval_gpu{i}.jsonl"
        with open(path, 'r', encoding='utf-8') as f:
            final += json.load(f)
        os.remove(path)

    with open("results/evaluation.jsonl", "w", encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=4)
