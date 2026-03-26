from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
EXTRACTED_DIR = RESULTS_DIR / "extracted"
TABLES_DIR = RESULTS_DIR / "tables"

for d in [RAW_DIR, EXTRACTED_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "anvo25/vlms-are-biased"
DATASET_SPLIT = "main"

MODELS = {
    "qwen3vl_8b": {
        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
        "max_tokens": 512,
        "max_model_len": 8192,
        "thinking": False,
    },
    "internvl3_8b": {
        "hf_id": "OpenGVLab/InternVL3-8B",
        "max_tokens": 512,
        "max_model_len": 8192,
        "thinking": False,
    },
    "gemma3_12b": {
        "hf_id": "google/gemma-3-12b-it",
        "max_tokens": 512,
        "max_model_len": 8192,
        "thinking": False,
    },
    "glm41v_9b": {
        "hf_id": "THUDM/GLM-4.1V-9B-Thinking",
        "max_tokens": 2048,
        "max_model_len": 8192,
        "thinking": True,
    },
}

FALLBACK_MODEL = {
    "glm41v_9b": {
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "max_tokens": 512,
        "max_model_len": 8192,
        "thinking": False,
    }
}

PROMPT_CONDITIONS = ["baseline", "debiased", "cot", "ccot", "doublecheck"]

SINGLE_TURN_CONDITIONS = ["baseline", "debiased", "cot"]
TWO_TURN_CONDITIONS = ["ccot", "doublecheck"]

DEBIASED_PREFIX = "Do not assume from prior knowledge and answer only based on what is visible in the image. "

ANSWER_SUFFIX = "Answer with a number in curly brackets, e.g., {9}."
COT_REPLACEMENT = "Let's think step by step. After your reasoning, provide your final answer with a number in curly brackets, e.g., {9}."

CCOT_TURN1_PROMPT = (
    "Generate a scene graph for this image as a JSON object with three keys: "
    "'objects' (list of objects you see), 'attributes' (list of object attributes), "
    "and 'relationships' (list of relationships between objects)."
)

DOUBLECHECK_TURN2_PROMPT = (
    "Please double-check your answer and give your final answer "
    "in curly brackets, following the format above."
)

BATCH_SIZE = 35

SAMPLING_TEMPERATURE = 0.0
SAMPLING_TOP_P = 1.0
