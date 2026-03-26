import io
import gc
import json
import base64
import time
from datetime import datetime, timezone

import torch
from tqdm import tqdm

from config import (
    MODELS,
    FALLBACK_MODEL,
    RAW_DIR,
    BATCH_SIZE,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_P,
    SINGLE_TURN_CONDITIONS,
    TWO_TURN_CONDITIONS,
    PROMPT_CONDITIONS,
)
from prompts import build_prompt


def pil_to_data_uri(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def make_single_turn_messages(image_uri, prompt_text):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def make_two_turn_messages(image_uri, turn1_prompt, turn1_response, turn2_prompt):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": turn1_prompt},
            ],
        },
        {
            "role": "assistant",
            "content": turn1_response,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": turn2_prompt},
            ],
        },
    ]


def load_checkpoint(output_path):
    done = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    key = (record["sample_id"], record["prompt_condition"])
                    done.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def save_result(output_path, record):
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_model(model_key):
    from vllm import LLM

    model_cfg = MODELS[model_key]
    print(f"Loading {model_cfg['hf_id']}...")

    try:
        llm = LLM(
            model=model_cfg["hf_id"],
            trust_remote_code=True,
            max_model_len=model_cfg["max_model_len"],
            gpu_memory_utilization=0.90,
            dtype="auto",
        )
        return llm, model_cfg
    except Exception as e:
        print(f"  Failed: {e}")
        if model_key in FALLBACK_MODEL:
            fb = FALLBACK_MODEL[model_key]
            print(f"  Trying fallback: {fb['hf_id']}")
            llm = LLM(
                model=fb["hf_id"],
                trust_remote_code=True,
                max_model_len=fb["max_model_len"],
                gpu_memory_utilization=0.90,
                dtype="auto",
            )
            return llm, fb
        raise


def unload_model(llm):
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(5)


def run_batch_single_turn(llm, model_cfg, batch_items, condition, ds):
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=SAMPLING_TEMPERATURE,
        top_p=SAMPLING_TOP_P,
        max_tokens=model_cfg["max_tokens"],
    )

    conversations = []
    metadata = []

    for idx in batch_items:
        sample = ds[idx]
        image_uri = pil_to_data_uri(sample["image"])
        prompt_text = build_prompt(condition, sample["prompt"], turn=1)
        messages = make_single_turn_messages(image_uri, prompt_text)
        conversations.append(messages)
        metadata.append({
            "sample_id": sample["ID"],
            "topic": sample["topic"],
            "sub_topic": sample["sub_topic"],
            "pixel": sample["pixel"],
            "type_of_question": sample["type_of_question"],
            "ground_truth": sample["ground_truth"],
            "expected_bias": sample["expected_bias"],
            "prompt_condition": condition,
            "full_prompt": prompt_text,
        })

    outputs = llm.chat(messages=conversations, sampling_params=params)

    results = []
    for i, output in enumerate(outputs):
        response_text = output.outputs[0].text
        record = metadata[i].copy()
        record["raw_response"] = response_text
        record["turn"] = 1
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        results.append(record)

    return results


def run_batch_two_turn(llm, model_cfg, batch_items, condition, ds):
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=SAMPLING_TEMPERATURE,
        top_p=SAMPLING_TOP_P,
        max_tokens=model_cfg["max_tokens"],
    )

    turn1_conversations = []
    sample_data = []

    for idx in batch_items:
        sample = ds[idx]
        image_uri = pil_to_data_uri(sample["image"])
        turn1_prompt = build_prompt(condition, sample["prompt"], turn=1)
        messages = make_single_turn_messages(image_uri, turn1_prompt)
        turn1_conversations.append(messages)
        sample_data.append({
            "idx": idx,
            "sample": sample,
            "image_uri": image_uri,
            "turn1_prompt": turn1_prompt,
        })

    turn1_outputs = llm.chat(messages=turn1_conversations, sampling_params=params)

    turn2_conversations = []
    metadata = []

    for i, output in enumerate(turn1_outputs):
        turn1_response = output.outputs[0].text
        sd = sample_data[i]
        sample = sd["sample"]

        turn2_prompt = build_prompt(
            condition, sample["prompt"], turn=2, previous_response=turn1_response
        )

        messages = make_two_turn_messages(
            sd["image_uri"], sd["turn1_prompt"], turn1_response, turn2_prompt
        )
        turn2_conversations.append(messages)
        metadata.append({
            "sample_id": sample["ID"],
            "topic": sample["topic"],
            "sub_topic": sample["sub_topic"],
            "pixel": sample["pixel"],
            "type_of_question": sample["type_of_question"],
            "ground_truth": sample["ground_truth"],
            "expected_bias": sample["expected_bias"],
            "prompt_condition": condition,
            "full_prompt": turn2_prompt,
            "turn1_response": turn1_response,
        })

    turn2_outputs = llm.chat(messages=turn2_conversations, sampling_params=params)

    results = []
    for i, output in enumerate(turn2_outputs):
        response_text = output.outputs[0].text
        record = metadata[i].copy()
        record["raw_response"] = response_text
        record["turn"] = 2
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        results.append(record)

    return results


def run_model(model_key, ds, conditions=None, sample_mode=False):
    conditions = conditions or PROMPT_CONDITIONS
    output_path = RAW_DIR / f"{model_key}.jsonl"
    done = load_checkpoint(output_path)

    llm, model_cfg = load_model(model_key)

    total_indices = list(range(len(ds)))
    if sample_mode:
        total_indices = total_indices[:1]

    for condition in conditions:
        pending = [
            i for i in total_indices
            if (ds[i]["ID"], condition) not in done
        ]

        if not pending:
            continue

        is_two_turn = condition in TWO_TURN_CONDITIONS
        runner = run_batch_two_turn if is_two_turn else run_batch_single_turn
        skipped = len(total_indices) - len(pending)
        desc = f"  {model_key}/{condition}"
        if skipped:
            desc += f" (resuming, {skipped} done)"

        for batch_start in tqdm(
            range(0, len(pending), BATCH_SIZE),
            desc=desc,
            unit="batch",
        ):
            batch = pending[batch_start : batch_start + BATCH_SIZE]
            try:
                results = runner(llm, model_cfg, batch, condition, ds)
                for record in results:
                    save_result(output_path, record)
                    done.add((record["sample_id"], record["prompt_condition"]))
            except Exception as e:
                print(f"\n  ERROR batch {batch_start}: {e}")
                continue

    unload_model(llm)
    return output_path
