import re, sys, argparse, os
import numpy as np
import torch
import shap
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    pipeline,
)
from config import CFG


def load_model():
    if not os.path.isdir(CFG.SAVE_DIR):
        print(f"No model at '{CFG.SAVE_DIR}'. Run train_distilbert.py first.")
        sys.exit(1)
    print(f"Loading model from '{CFG.SAVE_DIR}'...")
    tok   = DistilBertTokenizerFast.from_pretrained(CFG.SAVE_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(CFG.SAVE_DIR)
    model.to(CFG.DEVICE).eval()
    print(f"Ready on {CFG.DEVICE}\n")
    return model, tok


def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, tok, max_len, stride=64):
    ids    = tok.encode(text, add_special_tokens=False)
    usable = max_len - 2
    if len(ids) <= usable:
        return [text]
    chunks, start = [], 0
    while start < len(ids):
        end = min(start + usable, len(ids))
        chunks.append(tok.decode(ids[start:end], skip_special_tokens=True))
        if end == len(ids): break
        start += usable - stride
    return chunks


@torch.no_grad()
def predict(text, model, tok):
    cleaned   = clean_text(text)
    chunks    = chunk_text(cleaned, tok, CFG.MAX_LEN)
    all_probs = []
    for chunk in chunks:
        enc  = tok(chunk, max_length=CFG.MAX_LEN, padding="max_length",
                   truncation=True, return_tensors="pt")
        out  = model(input_ids=enc["input_ids"].to(CFG.DEVICE),
                     attention_mask=enc["attention_mask"].to(CFG.DEVICE))
        prob = torch.softmax(out.logits, dim=-1).squeeze().cpu().tolist()
        all_probs.append(prob)
    avg = np.mean(all_probs, axis=0)

    # Use threshold instead of 0.5 to reduce false AI predictions
    prediction = "AI" if avg[1] >= CFG.AI_THRESHOLD else "Human"

    return {
        "prediction": prediction,
        "confidence": f"{max(avg)*100:.2f}%",
        "prob_human": float(avg[0]),
        "prob_ai":    float(avg[1]),
        "num_chunks": len(chunks),
    }


def print_result(text, r):
    preview = text[:60] + "..." if len(text) > 60 else text
    ai_bar  = int(r["prob_ai"]  * 30)
    hum_bar = 30 - ai_bar
    print(f"\n  Input      : {preview}")
    print(f"  Prediction : {r['prediction']}")
    print(f"  Confidence : {r['confidence']}")
    print(f"  Human      : {r['prob_human']*100:.2f}%  {'#'*hum_bar}")
    print(f"  AI         : {r['prob_ai']*100:.2f}%  {'#'*ai_bar}")
    if r["num_chunks"] > 1:
        print(f"  (split into {r['num_chunks']} chunks, averaged)")


# SHAP
_explainer = None

def get_explainer(model, tok):
    global _explainer
    if _explainer is not None:
        return _explainer
    print("Building SHAP explainer (first time ~10s)...")
    pipe = pipeline("text-classification", model=model, tokenizer=tok,
                    device=-1, top_k=None)
    def pipe_fn(texts):
        out = pipe(list(texts), truncation=True, max_length=CFG.MAX_LEN)
        result = []
        for item in out:
            sorted_item = sorted(item, key=lambda x: x["label"])
            result.append([sorted_item[0]["score"], sorted_item[1]["score"]])
        return np.array(result)
    _explainer = shap.Explainer(pipe_fn, shap.maskers.Text(tok))
    return _explainer


def get_shap_words(text, model, tok, top_n=10):
    cleaned = clean_text(text)
    chunk   = chunk_text(cleaned, tok, CFG.MAX_LEN)[0]
    try:
        exp       = get_explainer(model, tok)
        shap_vals = exp([chunk])
        tokens    = shap_vals.data[0]
        scores    = shap_vals.values[0, :, 1]
        skip      = {".", ",", "!", "?", ";", ":", "-", "'", '"', "", " "}
        pairs     = [(t.strip(), float(s)) for t, s in zip(tokens, scores) if t.strip() not in skip]
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        return pairs[:top_n]
    except Exception as e:
        print(f"SHAP error: {e}")
        return []


def show_explain(text, result, model, tok):
    words = get_shap_words(text, model, tok)
    if not words:
        print("Could not compute explainability.")
        return
    print(f"\nTop influential words ({result['prediction']}, {result['confidence']}):")
    for i, (word, score) in enumerate(words, 1):
        direction = "-> AI" if score > 0.01 else ("-> Human" if score < -0.01 else "neutral")
        print(f"  {i}. {word}  ({score:+.4f})  {direction}")


def read_input():
    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            return "quit"
        stripped = line.strip()
        if stripped.upper() == "END":
            break
        if stripped.lower() in {"quit","exit","q","explain"} and not lines:
            return stripped.lower()
        lines.append(line)
    return "\n".join(lines).strip()


def interactive_mode(model, tok):
    print("="*50)
    print("  AI vs Human Detector")
    print("  Paste text, type END on new line to submit.")
    print("  Commands: explain | quit")
    print("="*50)
    last_text = None
    last_result = None
    while True:
        print("\nPaste text then type END:")
        user_input = read_input()
        if not user_input:
            print("Empty input.")
            continue
        if user_input in {"quit","exit","q"}:
            print("Goodbye!"); break
        if user_input == "explain":
            if last_text is None:
                print("No prediction yet.")
            else:
                show_explain(last_text, last_result, model, tok)
            continue
        last_result = predict(user_input, model, tok)
        last_text   = user_input
        print_result(user_input, last_result)
        print("  Type 'explain' to see influential words.")


def single_mode(text, model, tok):
    result = predict(text, model, tok)
    print_result(text, result)
    show_explain(text, result, model, tok)


def file_mode(path, model, tok):
    with open(path,"r",encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    for i, line in enumerate(lines, 1):
        r = predict(line, model, tok)
        print(f"  [{i}] {r['prediction']}  ({r['confidence']})  {line[:70]}")


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--text", type=str)
    g.add_argument("--file", type=str)
    args = parser.parse_args()
    model, tok = load_model()
    if args.text:   single_mode(args.text, model, tok)
    elif args.file: file_mode(args.file, model, tok)
    else:           interactive_mode(model, tok)


if __name__ == "__main__":
    main()
