import re, os, sys
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template_string
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    pipeline,
)
import shap
from config import CFG

app = Flask(__name__)

# Load model once
print(f"Loading model from {CFG.SAVE_DIR}...")
if not os.path.isdir(CFG.SAVE_DIR):
    print(f"No model at {CFG.SAVE_DIR}. Run train_distilbert.py first.")
    sys.exit(1)

tokenizer = DistilBertTokenizerFast.from_pretrained(CFG.SAVE_DIR)
model     = DistilBertForSequenceClassification.from_pretrained(CFG.SAVE_DIR)
model.to(CFG.DEVICE).eval()
print(f"Model ready on {CFG.DEVICE}")

# SHAP explainer — built once, cached
_explainer = None

def get_explainer():
    global _explainer
    if _explainer is not None:
        return _explainer
    print("Building SHAP explainer...")
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer,
                    device=-1, top_k=None)
    def pipe_fn(texts):
        out = pipe(list(texts), truncation=True, max_length=CFG.MAX_LEN)
        result = []
        for item in out:
            s = sorted(item, key=lambda x: x["label"])
            result.append([s[0]["score"], s[1]["score"]])
        return np.array(result)
    _explainer = shap.Explainer(pipe_fn, shap.maskers.Text(tokenizer))
    return _explainer


def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, max_len, stride=64):
    ids    = tokenizer.encode(text, add_special_tokens=False)
    usable = max_len - 2
    if len(ids) <= usable:
        return [text]
    chunks, start = [], 0
    while start < len(ids):
        end = min(start + usable, len(ids))
        chunks.append(tokenizer.decode(ids[start:end], skip_special_tokens=True))
        if end == len(ids): break
        start += usable - stride
    return chunks


@torch.no_grad()
def run_predict(text):
    cleaned   = clean_text(text)
    chunks    = chunk_text(cleaned, CFG.MAX_LEN)
    all_probs = []
    for chunk in chunks:
        enc  = tokenizer(chunk, max_length=CFG.MAX_LEN, padding="max_length",
                         truncation=True, return_tensors="pt")
        out  = model(input_ids=enc["input_ids"].to(CFG.DEVICE),
                     attention_mask=enc["attention_mask"].to(CFG.DEVICE))
        prob = torch.softmax(out.logits, dim=-1).squeeze().cpu().tolist()
        all_probs.append(prob)
    avg = np.mean(all_probs, axis=0)
    return {
        "prediction": "AI" if avg[1] >= CFG.AI_THRESHOLD else "Human",
        "prob_human": round(float(avg[0])*100, 2),
        "prob_ai":    round(float(avg[1])*100, 2),
        "confidence": round(float(max(avg))*100, 2),
        "num_chunks": len(chunks),
    }

def run_explain(text, top_n=10):
    cleaned = clean_text(text)
    chunk   = chunk_text(cleaned, CFG.MAX_LEN)[0]
    try:
        exp       = get_explainer()
        shap_vals = exp([chunk])
        tokens    = shap_vals.data[0]
        scores    = shap_vals.values[0, :, 1]
        skip      = {".", ",", "!", "?", ";", ":", "-", "'", '"', "", " "}
        pairs     = [(t.strip(), float(s)) for t, s in zip(tokens, scores) if t.strip() not in skip]
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        return [{"word": w, "score": round(s, 4)} for w, s in pairs[:top_n]]
    except Exception as e:
        print(f"SHAP error: {e}")
        return []


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI vs Human Detector</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 20px;
  }
  h1 { font-size: 1.8rem; margin-bottom: 6px; color: #fff; }
  p.sub { color: #888; margin-bottom: 30px; font-size: 0.9rem; }
  .card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 24px;
    width: 100%;
    max-width: 700px;
    margin-bottom: 20px;
  }
  textarea {
    width: 100%; height: 170px;
    background: #0f1117; color: #e0e0e0;
    border: 1px solid #2a2d3a; border-radius: 8px;
    padding: 12px; font-size: 0.95rem;
    resize: vertical; outline: none; font-family: inherit;
  }
  textarea:focus { border-color: #5865f2; }
  .btn-row { display: flex; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
  button {
    padding: 9px 20px; border: none; border-radius: 8px;
    font-size: 0.9rem; cursor: pointer; font-weight: 600; transition: opacity 0.2s;
  }
  button:hover { opacity: 0.8; }
  #btn-predict { background: #5865f2; color: #fff; }
  #btn-explain { background: #2a2d3a; color: #ccc; border: 1px solid #3a3d4a; }
  #btn-clear   { background: transparent; color: #666; border: 1px solid #2a2d3a; }
  #result-card { display: none; }
  .pred-label { font-size: 2rem; font-weight: 700; margin-bottom: 4px; }
  .pred-label.ai    { color: #e05c5c; }
  .pred-label.human { color: #5ce07a; }
  .conf-text { color: #888; font-size: 0.88rem; margin-bottom: 16px; }
  .bar-row { margin-bottom: 8px; }
  .bar-label { font-size: 0.83rem; color: #aaa; margin-bottom: 3px; }
  .bar-track { background: #0f1117; border-radius: 6px; height: 13px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 6px; transition: width 0.5s ease; }
  .bar-fill.human { background: #5ce07a; }
  .bar-fill.ai    { background: #e05c5c; }
  .chunks-note { font-size: 0.8rem; color: #555; margin-top: 6px; }
  #explain-box { margin-top: 18px; display: none; }
  #explain-box h3 { font-size: 0.95rem; color: #aaa; margin-bottom: 10px; }
  .word-list { list-style: none; }
  .word-list li {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0; border-bottom: 1px solid #2a2d3a; font-size: 0.9rem;
  }
  .word-list li:last-child { border-bottom: none; }
  .w-name  { font-weight: 600; min-width: 120px; }
  .w-score { font-family: monospace; color: #888; font-size: 0.85rem; }
  .w-dir   { font-size: 0.8rem; padding: 2px 9px; border-radius: 20px; font-weight: 600; }
  .w-dir.ai    { background: #3d1f1f; color: #e05c5c; }
  .w-dir.human { background: #1f3d28; color: #5ce07a; }
  .w-dir.neutral { background: #2a2d3a; color: #888; }
  #status { color: #888; font-size: 0.88rem; margin-top: 10px; min-height: 20px; }
  #error  { color: #e05c5c; font-size: 0.88rem; margin-top: 8px; display: none; }
</style>
</head>
<body>
<h1>🔍 AI vs Human Detector</h1>
<p class="sub">Fine-tuned DistilBERT + SHAP explainability</p>

<div class="card">
  <textarea id="txt" placeholder="Paste your text here..."></textarea>
  <div class="btn-row">
    <button id="btn-predict" onclick="doPredict()">Analyze</button>
    <button id="btn-explain" onclick="doExplain()">Explain</button>
    <button id="btn-clear"   onclick="doClear()">Clear</button>
  </div>
  <div id="status"></div>
  <div id="error"></div>
</div>

<div class="card" id="result-card">
  <div class="pred-label" id="pred-label"></div>
  <div class="conf-text"  id="conf-text"></div>
  <div class="bar-row">
    <div class="bar-label">Human <span id="h-pct"></span></div>
    <div class="bar-track"><div class="bar-fill human" id="h-bar" style="width:0%"></div></div>
  </div>
  <div class="bar-row">
    <div class="bar-label">AI <span id="a-pct"></span></div>
    <div class="bar-track"><div class="bar-fill ai" id="a-bar" style="width:0%"></div></div>
  </div>
  <div class="chunks-note" id="chunks-note"></div>

  <div id="explain-box">
    <h3>Top influential words (SHAP)</h3>
    <ul class="word-list" id="word-list"></ul>
  </div>
</div>

<script>
function setStatus(msg) { document.getElementById('status').textContent = msg; }
function setError(msg)  {
  const e = document.getElementById('error');
  e.textContent = msg; e.style.display = msg ? 'block' : 'none';
}
function getText() { return document.getElementById('txt').value.trim(); }

async function doPredict() {
  const text = getText();
  if (!text) return setError('Please enter some text.');
  setError(''); setStatus('Analyzing...');
  try {
    const res  = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})});
    const data = await res.json();
    if (data.error) return setError(data.error);
    showResult(data);
    setStatus('');
  } catch(e) { setError('Request failed.'); setStatus(''); }
}

async function doExplain() {
  const text = getText();
  if (!text) return setError('Please enter some text.');
  setError(''); setStatus('Running SHAP explainability (may take ~15s on first run)...');
  document.getElementById('btn-explain').disabled = true;
  try {
    const res  = await fetch('/explain', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})});
    const data = await res.json();
    if (data.error) return setError(data.error);
    showResult(data.result);
    showWords(data.words);
    setStatus('');
  } catch(e) { setError('Request failed.'); setStatus(''); }
  finally { document.getElementById('btn-explain').disabled = false; }
}

function showResult(d) {
  document.getElementById('result-card').style.display = 'block';
  document.getElementById('explain-box').style.display = 'none';
  const lbl = document.getElementById('pred-label');
  lbl.textContent = d.prediction;
  lbl.className   = 'pred-label ' + d.prediction.toLowerCase();
  document.getElementById('conf-text').textContent = `Confidence: ${d.confidence}%`;
  document.getElementById('h-pct').textContent = `${d.prob_human}%`;
  document.getElementById('a-pct').textContent = `${d.prob_ai}%`;
  document.getElementById('h-bar').style.width = `${d.prob_human}%`;
  document.getElementById('a-bar').style.width = `${d.prob_ai}%`;
  document.getElementById('chunks-note').textContent =
    d.num_chunks > 1 ? `ℹ️ Long text split into ${d.num_chunks} chunks, averaged.` : '';
}

function showWords(words) {
  if (!words || words.length === 0) {
    setError('No words returned from SHAP.'); return;
  }
  const list = document.getElementById('word-list');
  list.innerHTML = '';
  words.forEach((w, i) => {
    const dc = w.score > 0.01 ? 'ai' : (w.score < -0.01 ? 'human' : 'neutral');
    const dt = w.score > 0.01 ? '→ AI' : (w.score < -0.01 ? '→ Human' : 'neutral');
    const sign = w.score >= 0 ? '+' : '';
    list.innerHTML += `<li>
      <span class="w-name">${i+1}. ${esc(w.word)}</span>
      <span class="w-score">${sign}${w.score.toFixed(4)}</span>
      <span class="w-dir ${dc}">${dt}</span>
    </li>`;
  });
  document.getElementById('explain-box').style.display = 'block';
}

function doClear() {
  document.getElementById('txt').value = '';
  document.getElementById('result-card').style.display = 'none';
  setError(''); setStatus('');
}

function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    text = data.get("text","").strip()
    if not text: return jsonify({"error": "No text provided."})
    try:    return jsonify(run_predict(text))
    except Exception as e: return jsonify({"error": str(e)})

@app.route("/explain", methods=["POST"])
def explain():
    data = request.get_json() or {}
    text = data.get("text","").strip()
    if not text: return jsonify({"error": "No text provided."})
    try:
        result = run_predict(text)
        words  = run_explain(text)
        return jsonify({"result": result, "words": words})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    print("\nStarting Flask app → http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
