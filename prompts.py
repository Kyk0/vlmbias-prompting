from config import (
    DEBIASED_PREFIX,
    ANSWER_SUFFIX,
    COT_REPLACEMENT,
    CCOT_TURN1_PROMPT,
    DOUBLECHECK_TURN2_PROMPT,
)


def build_baseline(prompt_text):
    return prompt_text


def build_debiased(prompt_text):
    return DEBIASED_PREFIX + prompt_text


def build_cot(prompt_text):
    if ANSWER_SUFFIX in prompt_text:
        return prompt_text.replace(ANSWER_SUFFIX, COT_REPLACEMENT)
    return prompt_text + " Let's think step by step."


def build_ccot_turn1():
    return CCOT_TURN1_PROMPT


def build_ccot_turn2(prompt_text, scene_graph_response):
    return (
        f"Use the image and the following scene graph as context:\n"
        f"{scene_graph_response}\n\n"
        f"Now answer the following question: {prompt_text}"
    )


def build_doublecheck_turn1(prompt_text):
    return prompt_text


def build_doublecheck_turn2():
    return DOUBLECHECK_TURN2_PROMPT


def build_prompt(condition, prompt_text, turn=1, previous_response=None):
    if condition == "baseline":
        return build_baseline(prompt_text)
    elif condition == "debiased":
        return build_debiased(prompt_text)
    elif condition == "cot":
        return build_cot(prompt_text)
    elif condition == "ccot":
        if turn == 1:
            return build_ccot_turn1()
        else:
            return build_ccot_turn2(prompt_text, previous_response or "")
    elif condition == "doublecheck":
        if turn == 1:
            return build_doublecheck_turn1(prompt_text)
        else:
            return build_doublecheck_turn2()
    raise ValueError(f"Unknown condition: {condition}")
