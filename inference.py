import argparse
import torch
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import whisper

from openomni.constants import SPEECH_TOKEN_INDEX
from openomni.model.builder import load_pretrained_qwen_model
from openomni.utils import disable_torch_init
import time

def eval_model(args):

    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_qwen_model(model_path, args.model_base, is_lora=args.is_lora)

    use_speech = False

    tokenizer.add_tokens(["<speech>"], special_tokens=True)
    tokenizer.chat_template="{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    speech_token_index = tokenizer.convert_tokens_to_ids("<speech>")

    speech_file = "./assets/question.wav"

    if use_speech:
        question_prompt = "<speech>\n Please answer the questions in the user's input speech"
    else:
        question_prompt = "hello"

    input_id = []
    system_message = "You are a helpful language, vision and speech assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language or speech."
    input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message},
                                                {"role" : "user", "content" : question_prompt}],
                                                add_generation_prompt=True)
    
    for idx, encode_id in enumerate(input_id):
        if encode_id == speech_token_index:
            input_id[idx] = SPEECH_TOKEN_INDEX

    input_ids = torch.tensor([input_id], dtype=torch.long)
    input_ids = input_ids.to(device='cuda', non_blocking=True)

    speech = whisper.load_audio(os.path.join('',speech_file))
    if args.input_type == "raw":
        speech = torch.from_numpy(speech)
    elif args.input_type == "mel":
        speech = whisper.pad_or_trim(speech)
        speech_tensor = whisper.log_mel_spectrogram(speech, n_mels=args.mel_size).permute(1, 0)

    speech_length = torch.LongTensor([speech_tensor.shape[0]])
    speech_tensor = speech_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True).unsqueeze(0)
    speech_length = speech_length.to(device='cuda', non_blocking=True)
    with torch.inference_mode():
        time1=time.time()
        outputs = model.generate(
            input_ids,
            speech = None,
            speech_lengths = None,
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
        time2=time.time()
        output_ids, output_units = outputs

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if args.s2s and args.speech_generator_type=="ar":
        output_units=output_units

    print(f"H-{time2-time1}-{idx}\t{outputs}")
    if args.s2s:
        print(f"U-{idx}\t{output_units}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/omnicharacter++_stage1/checkpoint-last")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_qwen2")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--input_type", type=str, default="mel")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--s2s", action="store_true", default=True)
    parser.add_argument("--speech_generator_type", action="store_true", default="ar")
    parser.add_argument("--is_lora", action="store_true", default=False)
    parser.add_argument("--square_eval", type=bool, default=True)
    args = parser.parse_args()

    eval_model(args)
