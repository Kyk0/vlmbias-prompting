import re
import json
from tqdm import tqdm
from config import RAW_DIR, EXTRACTED_DIR


def extract_from_brackets(text):
    matches = re.findall(r'\{([^}]+)\}', text)
    if matches:
        return matches[-1].strip()
    return None


def extract_number_fallback(text):
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        return numbers[-1]
    return None


def strip_thinking(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def extract_answer(raw_response, is_thinking_model=False):
    text = raw_response
    if is_thinking_model:
        text = strip_thinking(text)

    answer = extract_from_brackets(text)
    method = "brackets"

    if answer is None:
        answer = extract_number_fallback(text)
        method = "fallback_number"

    if answer is None:
        method = "failed"

    return answer, method


def process_raw_results(model_key, is_thinking=False):
    raw_path = RAW_DIR / f"{model_key}.jsonl"
    out_path = EXTRACTED_DIR / f"{model_key}.jsonl"

    if not raw_path.exists():
        return

    records = []
    with open(raw_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    correct = 0
    failed = 0
    with open(out_path, "w") as f:
        for record in tqdm(records, desc=f"Extracting {model_key}"):
            answer, method = extract_answer(record["raw_response"], is_thinking)
            record["extracted_answer"] = answer
            record["extraction_method"] = method
            record["is_correct"] = (
                answer is not None
                and answer.strip() == record["ground_truth"].strip()
            )
            record["is_bias_aligned_error"] = (
                answer is not None
                and answer.strip() != record["ground_truth"].strip()
                and answer.strip() == record["expected_bias"].strip()
            )
            if record["is_correct"]:
                correct += 1
            if method == "failed":
                failed += 1
            f.write(json.dumps(record) + "\n")

    print(f"  {model_key}: {correct}/{len(records)} correct, {failed} failed extractions")
    return out_path
