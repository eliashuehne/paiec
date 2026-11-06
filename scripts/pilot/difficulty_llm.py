"""Score questions with a small local language model: three single-pass difficulty signals.

rating   - expected value of a 1-9 difficulty rating (digit-token probabilities)
entropy  - entropy of the first answer token when asked to answer
top_prob - probability of that most likely first answer token
nll      - mean token negative log-likelihood of the question text

    python difficulty_llm.py SAMPLE.parquet OUT.parquet [model]
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SAMPLE, OUT = sys.argv[1], sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-1.5B-Instruct"
MAX_QUESTION_TOKENS = 768

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
digits = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(1, 10)]


def clip(text: str) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)[:MAX_QUESTION_TOKENS]
    return tokenizer.decode(ids)


def chat(user: str, prefix: str = "") -> torch.Tensor:
    text = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                         add_generation_prompt=True) + prefix
    return tokenizer(text, return_tensors="pt").input_ids.to("cuda")


@torch.no_grad()
def next_token_probs(ids: torch.Tensor) -> torch.Tensor:
    return torch.softmax(model(ids).logits[0, -1].float(), dim=-1)


@torch.no_grad()
def score(question: str) -> dict:
    question = clip(question)
    rating_probs = next_token_probs(chat(
        "How difficult is the following task for a capable AI system? Reply with a single digit "
        "from 1 (trivial) to 9 (almost impossible).\n\nTask:\n" + question, "Difficulty: "))[digits]
    rating = float((rating_probs / rating_probs.sum() * torch.arange(1, 10, device="cuda")).sum())
    answer = next_token_probs(chat(question + "\n\nAnswer as briefly as possible.", "Answer: "))
    entropy = float(-(answer * torch.log(answer.clamp_min(1e-12))).sum())
    ids = tokenizer(question, return_tensors="pt").input_ids.to("cuda")
    nll = float(model(ids, labels=ids).loss) if ids.shape[1] > 1 else float("nan")
    return {"rating": rating, "entropy": entropy, "top_prob": float(answer.max()), "nll": nll}


sample = pd.read_parquet(SAMPLE)
started, rows = time.time(), []
for number, text in enumerate(sample["content"], start=1):
    rows.append(score(text))
    if number % 250 == 0:
        print(f"{number}/{len(sample)} ({time.time() - started:.0f}s)", flush=True)
result = pd.concat([sample.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
result.to_parquet(OUT, index=False)
print(f"scored {len(result)} questions with {MODEL} in {time.time() - started:.0f}s -> {OUT}")
