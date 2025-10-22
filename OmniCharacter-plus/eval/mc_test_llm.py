import json
from data_root.api_call import *
import random
import string
import re
import json
import os
from tqdm import tqdm
import random

from concurrent.futures import ThreadPoolExecutor, as_completed


def evaluate_model_accuracy_with_api(
    test_data,
    provider="Doubao-pro-32k",
    temperature=0.2,
    max_tokens=1024,
    task_type="eval_dialogue",
    max_workers=8,
    max_sample=1000,
    save_eval_results=False
):

    print(f"📂 Found {len(test_data)} test sampels")

    results = []

    def eval_worker(data):
        
        # try:
        correct_answer = data["correct_answer"].strip().upper()  # A/B/C/D

        data.pop("correct_answer")
        multichoice = data["multichoice"]  # List[str]

        options = []
        for item in multichoice:
            m = re.match(r"([A-D])\.\s*(.*)", item)
            if not m:
                print(f"⚠️ Unable to parse option format: {item}")
                continue
            label, text = m.group(1), m.group(2)
            options.append((label, text))

        correct_text = None
        for label, text in options:
            if label == correct_answer:
                correct_text = text
                break
        if correct_text is None:
            print(f"❌ Correct answer text not found, file id: {data['id']}")
            return None, None, None

        all_correct = True
        for _ in range(4):

            random.shuffle(options)

            new_labels = list(string.ascii_uppercase[:len(options)])  # ['A','B','C','D']
            new_multichoice = {label: text for label, (_, text) in zip(new_labels, options)}

            new_correct_label = None
            for label, text in new_multichoice.items():
                if text == correct_text:
                    new_correct_label = label
                    break
            if new_correct_label is None:
                print(f"❌ New correct answer not found, file id: {data['id']}")
                all_correct = False
                break
            
            options_text = "\n".join([f"{label}. {text}" for label, text in new_multichoice.items()])
            data["multichoice"] = options_text

            dialogue = ''.join(f"{s['from']}: {s['value']}\n" for s in data["conversations"])

            full_prompt = f"""You will play as {data['gpt']}. Your profile is: {data['system']}

            Select the best answer to the following question and dialogue contexts.

            Dialogue contexts: {dialogue}.
            
            Respond with only the letter (A, B, C, or D) of the correct option.

            [Question]: {data['question']}

            [Multi-Choices]: {data['multichoice']}

            The best answer is:"""

            _, model_output = call_model(
                prompt=full_prompt,
                provider=provider,
                temperature=temperature,
                max_tokens=max_tokens,
                task_type=task_type,
                output_root=None,
                eval_mode=True
            )
            if model_output is None:
                model_answer = None
            else:
                matches = re.search(r'[ABCD]', model_output.strip())

                model_answer = None if matches is None else matches[0]

            # model_answer = random.sample(['A', 'B', 'C', 'D'], k=1)[0] # random evaluation

            if model_answer != new_correct_label:
                all_correct = False
                break

        type_current = data['type']
        return all_correct, type_current, data

        # except Exception as e:
        #     print(f"❌ Error evaluating {data['id']}: {e}")
        #     return None, None

    type = {
        "Negotiation_correct": 0,
        "Negotiation_total": 0,
        "Exchange_correct": 0,
        "Exchange_total": 0,
        "Free-talk_correct": 0,
        "Free-talk_total": 0,
        "Expert-domain_correct": 0,
        "Expert-domain_total": 0,
        "Instruction-giving_correct": 0,
        "Instruction-giving_total": 0,
        "Persuasion_correct": 0,
        "Persuasion_total": 0,
        "Conflict-resolution_correct": 0,
        "Conflict-resolution_total": 0,
        "Planning_correct": 0,
        "Planning_total": 0,
    }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(eval_worker, sample) for idx, sample in enumerate(test_data[:max_sample])]
        
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing", ncols=100):
            # try:
            result, type_current, sample = f.result()

            if result is not None:
                results.append(result)
            if result is True:
                type[f"{type_current}_correct"] += 1
                type[f"{type_current}_total"] += 1
            elif result is not None:
                type[f"{type_current}_total"] += 1

            # except Exception as e:
            #     print(f"❌ Exception in thread: {e}")
    
    total = len(results) # + erro
    correct = sum(results)
    accuracy = correct / total if total else 0.0

    result = {}
    valid_accuracies = []
    result["model"] = provider
    for key in type:
        if key.endswith("_correct"):
            base = key.replace("_correct", "")
            correct = type.get(f"{base}_correct", 0)
            total = type.get(f"{base}_total", 0)
            if total > 0:
                acc = round(correct / total, 4)
                result[base] = acc
                valid_accuracies.append(acc)
            else:
                result[base] = None  

    if valid_accuracies:
        avg_accuracy = round(sum(valid_accuracies) / len(valid_accuracies), 4)
    else:
        avg_accuracy = None

    result["average_accuracy"] = avg_accuracy

    if save_eval_results:
        output_dir = "hard_test_mc_output"
        os.makedirs(output_dir, exist_ok=True)
        model_name = provider.replace("/", "_")
        output_file = os.path.join(output_dir, f"{model_name}.json")

        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)

        print(f"Per-category and average accuracy saved to: {output_file}")

    print(result)
    return accuracy

if __name__ == "__main__":
    
    test_data_path = "/home/zhanghaonan/code/acl_extension/data_root/merged_dialogues_four_test.json" 
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_data_path = "/home/zhanghaonan/code/acl_extension/data_root/merged_dialogues_three_test.json" 
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data.extend(json.load(f))

    test_data_path = "/home/zhanghaonan/code/acl_extension/data_root/merged_dialogues_two_test.json" 
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data.extend(json.load(f))

    evaluate_model_accuracy_with_api(
        test_data,
        provider="gemini-2.5-pro-preview-05-06", # gpt-3.5-turbo
        temperature=0.,
        max_tokens=4096,
        task_type="dialogue_eval",
        max_workers=256,
        max_sample=999999,
        save_eval_results=False
    )