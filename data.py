import datasets
from tqdm import tqdm
from config import DATASET_NAME, DATASET_SPLIT, ANSWER_SUFFIX


def load_dataset():
    ds = datasets.load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    return ds


def validate_dataset(ds):
    issues = []
    for i, sample in enumerate(tqdm(ds, desc="Validating dataset")):
        if ANSWER_SUFFIX not in sample["prompt"]:
            issues.append(f"Sample {i} (ID={sample['ID']}): missing answer suffix in prompt")
        if not sample["ground_truth"].strip():
            issues.append(f"Sample {i} (ID={sample['ID']}): empty ground_truth")
        if not sample["expected_bias"].strip():
            issues.append(f"Sample {i} (ID={sample['ID']}): empty expected_bias")
        if sample["image"] is None:
            issues.append(f"Sample {i} (ID={sample['ID']}): missing image")
    return issues


def get_domain_stats(ds):
    from collections import Counter
    topics = Counter(sample["topic"] for sample in ds)
    subtopics = Counter(sample["sub_topic"] for sample in ds)
    resolutions = Counter(sample["pixel"] for sample in ds)
    qtypes = Counter(sample["type_of_question"] for sample in ds)
    return {
        "total": len(ds),
        "topics": dict(topics),
        "subtopics": dict(subtopics),
        "resolutions": dict(resolutions),
        "question_types": dict(qtypes),
    }
