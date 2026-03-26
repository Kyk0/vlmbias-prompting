import argparse
import time
from config import MODELS, PROMPT_CONDITIONS
from data import load_dataset, validate_dataset
from extract import process_raw_results
from evaluate import run_full_evaluation


def cmd_sample(args):
    print("Sample run: 1 image, all prompts")
    ds = load_dataset()
    sample = ds[0]
    print(f"  Sample: {sample['ID']} | {sample['topic']} | gt={sample['ground_truth']} bias={sample['expected_bias']}")

    model_key = args.model or "qwen3vl_8b"

    from inference import run_model
    run_model(model_key, ds, conditions=PROMPT_CONDITIONS, sample_mode=True)

    is_thinking = MODELS.get(model_key, {}).get("thinking", False)
    process_raw_results(model_key, is_thinking=is_thinking)

    from config import EXTRACTED_DIR
    import json
    extracted_path = EXTRACTED_DIR / f"{model_key}.jsonl"
    if extracted_path.exists():
        print("\nResults:")
        with open(extracted_path) as f:
            for line in f:
                r = json.loads(line)
                status = "CORRECT" if r.get("is_correct") else "WRONG"
                print(f"  {r['prompt_condition']:12s} -> {r.get('extracted_answer'):>4s} [{status}]  {r['raw_response'][:80]}...")


def cmd_inference(args):
    ds = load_dataset()
    issues = validate_dataset(ds)
    if issues:
        print(f"WARNING: {len(issues)} validation issues")

    model_keys = [args.model] if args.model else list(MODELS.keys())

    for model_key in model_keys:
        start = time.time()
        from inference import run_model
        run_model(model_key, ds, conditions=PROMPT_CONDITIONS, sample_mode=False)
        print(f"{model_key} done in {(time.time()-start)/60:.1f}m")


def cmd_extract(args):
    model_keys = [args.model] if args.model else list(MODELS.keys())
    for mk in model_keys:
        is_thinking = MODELS.get(mk, {}).get("thinking", False)
        process_raw_results(mk, is_thinking=is_thinking)


def cmd_evaluate(args):
    model_keys = [args.model] if args.model else list(MODELS.keys())
    run_full_evaluation(model_keys)


def cmd_all(args):
    cmd_inference(args)
    cmd_extract(args)
    cmd_evaluate(args)


def main():
    parser = argparse.ArgumentParser(description="VLMBias Prompting Experiment")
    parser.add_argument(
        "--mode",
        choices=["sample", "inference", "extract", "evaluate", "all"],
        required=True,
    )
    parser.add_argument("--model", type=str, default=None, help="Run single model only")
    args = parser.parse_args()

    dispatch = {
        "sample": cmd_sample,
        "inference": cmd_inference,
        "extract": cmd_extract,
        "evaluate": cmd_evaluate,
        "all": cmd_all,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
