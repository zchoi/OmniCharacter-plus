import os
import json
import torch
import argparse
from tqdm import tqdm
from multiprocessing import Process, current_process
from transformers import AutoTokenizer
from openomni.model.builder import load_pretrained_qwen_model
from openomni.utils import disable_torch_init
import warnings
warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
torch.multiprocessing.set_start_method('spawn', force=True)

def load_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_qwen_model(
        model_path, args.model_base, is_lora=args.is_lora
    )

    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    return tokenizer, model, context_len

def concat_messages(conversations, role, system):
    history = []
    first_query = system
    if conversations[0]['from'] == role:
        first_response = f"好的！现在我来扮演{role}。" + "我首先发话：" + conversations[0]['value']
    else:
        first_response = f"好的！现在我来扮演{role}。"

    history.append({"role": "system", "content": first_query})
    history.append({"role": "assistant", "content": first_response})

    for i in range(len(conversations)):
        if conversations[i]['from'] == role:
            if i == 0:
                continue
            else:
                assert conversations[i-1]['from'] != role
                query = f"{conversations[i-1]['from']}：" + conversations[i-1]['value']
                response = f"{conversations[i]['from']}：" + conversations[i]['value']
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})
    assert conversations[-1]['from'] != role

    query = f"{conversations[-1]['from']}：" + conversations[-1]['value']
    return history, query

def make_inputs(context):
    dialogues = context.split('\n')
    inputs = []
    for dial in dialogues:
        role = dial.split("：")[0]
        dial = "：".join(dial.split("：")[1:])
        inputs.append({"from": role, "value": dial})
    return inputs

def get_response_chatglm(data, args, tokenizer, model, context_len, role_informations):
    context = data['context']
    role = data['role']

    role_information = role_informations[role]
    role_system = f'''{role_information}
现在请你扮演一个角色扮演专家。请你根据上述信息扮演{role}进行对话。
'''
    messages, query = concat_messages(make_inputs(context), role, role_system)
    messages.append({"role": "user", "content": query})

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to("cuda", non_blocking=True)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            speech=None,
            speech_lengths=None,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            streaming_unit_gen=False,
            faster_infer=True
        )
        output_ids, _ = outputs
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    data["model_output"] = response
    return data

def run_worker(gpu_id, model_idx, args, data_chunk, role_informations):

    tokenizer, model, context_len = load_model(args)
    result = []

    pbar = tqdm(data_chunk, desc=f"GPU {gpu_id} Model {model_idx}", position=model_idx, leave=True)
    for item in pbar:
        output = get_response_chatglm(item, args, tokenizer, model, context_len, role_informations)
        result.append(output)

    out_path = f"results/generation_gpu{gpu_id}_m{model_idx}.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def chunk_data(datas, total_chunks):
    chunk_size = (len(datas) + total_chunks - 1) // total_chunks
    return [datas[i * chunk_size: (i + 1) * chunk_size] for i in range(total_chunks)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/zhanghaonan/code/acl_extension/OpenOmni/checkpoints/omnicharacter++_stage1_5e-4_no_audio_2epoches/checkpoint-29504")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_qwen2")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--is_lora", action="store_true", default=False)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--models-per-gpu", type=int, default=1)
    args = parser.parse_args()

    with open('data/test_data.jsonl', 'r', encoding='utf-8') as f:
        datas = json.load(f)
    with open('data/character_profiles.json', 'r', encoding='utf-8') as f:
        role_informations = json.load(f)

    total_workers = args.num_gpus * args.models_per_gpu
    data_chunks = chunk_data(datas, total_workers)

    processes = []
    for i in range(total_workers):
        gpu_id = i % args.num_gpus
        model_idx = i // args.num_gpus
        # print(gpu_id, model_idx)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id + 1)
        torch.cuda.set_device(gpu_id)
        p = Process(target=run_worker, args=(gpu_id, model_idx, args, data_chunks[i], role_informations))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    merged = []
    for i in range(total_workers):
        gpu_id = i % args.num_gpus
        model_idx = i // args.num_gpus
        path = f"results/generation_gpu{gpu_id}_m{model_idx}.jsonl"
        with open(path, 'r', encoding='utf-8') as f:
            merged += json.load(f)
        os.remove(path)

    with open("results/generation.jsonl", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)

