import os, re, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertConfig,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import CFG

warnings.filterwarnings("ignore")


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

seed_everything(CFG.SEED)


def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_data():
    df = pd.read_csv(CFG.CSV_PATH)
    df = df[[CFG.TEXT_COL, CFG.LABEL_COL]].dropna(subset=[CFG.TEXT_COL, CFG.LABEL_COL])
    df[CFG.TEXT_COL]  = df[CFG.TEXT_COL].astype(str).apply(clean_text)
    df[CFG.LABEL_COL] = df[CFG.LABEL_COL].astype(float).astype(int)
    df = df[df[CFG.TEXT_COL].str.len() > 20].reset_index(drop=True)

    print(f"Loaded {len(df):,} rows")
    print(df[CFG.LABEL_COL].value_counts().rename({0:"Human",1:"AI"}).to_string())

    if CFG.MAX_SAMPLES_PER_CLASS is not None:
        parts = []
        for label, group in df.groupby(CFG.LABEL_COL):
            parts.append(group.sample(min(len(group), CFG.MAX_SAMPLES_PER_CLASS), random_state=CFG.SEED))
        df = pd.concat(parts).reset_index(drop=True)
        print(f"Sampled to {CFG.MAX_SAMPLES_PER_CLASS:,}/class → {len(df):,} total")

    return df


def split_data(df):
    train_val, test = train_test_split(df, test_size=CFG.TEST_SPLIT, stratify=df[CFG.LABEL_COL], random_state=CFG.SEED)
    train, val = train_test_split(train_val, test_size=CFG.VAL_SPLIT/(1-CFG.TEST_SPLIT), stratify=train_val[CFG.LABEL_COL], random_state=CFG.SEED)
    print(f"Split → train {len(train):,} | val {len(val):,} | test {len(test):,}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts  = df[CFG.TEXT_COL].tolist()
        self.labels = df[CFG.LABEL_COL].tolist()
        self.tok    = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], max_length=CFG.MAX_LEN, padding="max_length",
                       truncation=True, return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_loaders(train_df, val_df, test_df, tokenizer):
    def make(df, shuffle):
        return DataLoader(TextDataset(df, tokenizer), batch_size=CFG.BATCH_SIZE, shuffle=shuffle, num_workers=0)
    return make(train_df, True), make(val_df, False), make(test_df, False)


def build_model():
    config = DistilBertConfig.from_pretrained(
        CFG.MODEL_NAME, num_labels=2,
        dropout=CFG.DROPOUT,
        seq_classif_dropout=CFG.DROPOUT,
    )
    return DistilBertForSequenceClassification.from_pretrained(CFG.MODEL_NAME, config=config, ignore_mismatched_sizes=True)


class LivePlot:
    def __init__(self):
        self.tl, self.vl, self.ta, self.va = [], [], [], []
        plt.ion()
        self.fig = plt.figure(figsize=(12,5))
        self.fig.suptitle("Training — AI vs Human", fontweight="bold")
        gs = gridspec.GridSpec(1,2,figure=self.fig)
        self.ax1 = self.fig.add_subplot(gs[0,0])
        self.ax2 = self.fig.add_subplot(gs[0,1])
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.show(block=False)

    def update(self, ep, tl, vl, ta, va):
        self.tl.append(tl); self.vl.append(vl)
        self.ta.append(ta*100); self.va.append(va*100)
        ep_range = list(range(1, len(self.tl)+1))
        for ax, y1, y2, title, ylabel in [
            (self.ax1, self.tl, self.vl, "Loss", "Loss"),
            (self.ax2, self.ta, self.va, "Accuracy %", "Accuracy %")
        ]:
            ax.cla()
            ax.plot(ep_range, y1, "b-o", label="Train")
            ax.plot(ep_range, y2, "r-o", label="Val")
            ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
            ax.legend(); ax.grid(True, alpha=0.3)
        self.fig.canvas.draw(); self.fig.canvas.flush_events()

    def save(self):
        os.makedirs(CFG.GRAPH_DIR, exist_ok=True)
        path = os.path.join(CFG.GRAPH_DIR, "training_curves.png")
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Graph saved → {path}")

    def keep_open(self):
        plt.ioff(); plt.show()


class EarlyStopping:
    def __init__(self):
        self.best = float("inf"); self.counter = 0; self.best_epoch = 0

    def step(self, val_loss, epoch):
        if val_loss < self.best - CFG.MIN_DELTA:
            self.best = val_loss; self.counter = 0; self.best_epoch = epoch; return False
        self.counter += 1
        print(f"  No improvement {self.counter}/{CFG.PATIENCE}")
        return self.counter >= CFG.PATIENCE


def train_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss, preds, truths = 0.0, [], []
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.FP16)
    for batch in tqdm(loader, desc="  Train", leave=False):
        optimizer.zero_grad()
        ids  = batch["input_ids"].to(CFG.DEVICE)
        mask = batch["attention_mask"].to(CFG.DEVICE)
        lbls = batch["labels"].to(CFG.DEVICE)
        with torch.cuda.amp.autocast(enabled=CFG.FP16):
            out  = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, lbls)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        total_loss += loss.item()
        preds  += out.logits.argmax(-1).cpu().tolist()
        truths += lbls.cpu().tolist()
    return total_loss/len(loader), accuracy_score(truths, preds)


@torch.no_grad()
def evaluate(model, loader, loss_fn, return_probs=False):
    model.eval()
    total_loss, preds, truths, all_probs = 0.0, [], [], []
    for batch in tqdm(loader, desc="  Eval ", leave=False):
        ids  = batch["input_ids"].to(CFG.DEVICE)
        mask = batch["attention_mask"].to(CFG.DEVICE)
        lbls = batch["labels"].to(CFG.DEVICE)
        out  = model(input_ids=ids, attention_mask=mask)
        loss = loss_fn(out.logits, lbls)
        total_loss += loss.item()
        probs = torch.softmax(out.logits, dim=-1)
        all_probs += probs.cpu().tolist()
        preds  += probs.argmax(-1).cpu().tolist()
        truths += lbls.cpu().tolist()
    acc = accuracy_score(truths, preds)
    if return_probs: return total_loss/len(loader), acc, preds, truths, all_probs
    return total_loss/len(loader), acc


def print_metrics(preds, truths, all_probs):
    print("\n" + "="*55)
    print("  FINAL TEST RESULTS")
    print("="*55)
    print(f"  Accuracy  : {accuracy_score(truths,preds)*100:.2f}%")
    print(f"  ROC-AUC   : {roc_auc_score(truths,[p[1] for p in all_probs]):.4f}")
    print("\n" + classification_report(truths, preds, target_names=["Human","AI"], digits=4))
    cm = confusion_matrix(truths, preds)
    print(f"  Confusion Matrix:")
    print(f"            Human    AI")
    print(f"  Human  {cm[0][0]:>6}  {cm[0][1]:>6}")
    print(f"  AI     {cm[1][0]:>6}  {cm[1][1]:>6}")
    print("="*55)


def main():
    print(f"Device: {CFG.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"LR={CFG.LR}  Dropout={CFG.DROPOUT}  WeightDecay={CFG.WEIGHT_DECAY}  LabelSmooth={CFG.LABEL_SMOOTHING}")

    df = load_data()
    train_df, val_df, test_df = split_data(df)

    print(f"\nLoading {CFG.MODEL_NAME}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(CFG.MODEL_NAME)
    model     = build_model().to(CFG.DEVICE)

    train_loader, val_loader, test_loader = build_loaders(train_df, val_df, test_df, tokenizer)

    # Class weights — penalize wrong predictions equally for both classes
    class_counts = train_df[CFG.LABEL_COL].value_counts().sort_index().values
    weights      = torch.tensor(1.0 / class_counts, dtype=torch.float).to(CFG.DEVICE)
    weights      = weights / weights.sum()
    loss_fn      = nn.CrossEntropyLoss(weight=weights, label_smoothing=CFG.LABEL_SMOOTHING)
    print(f"Class weights — Human: {weights[0]:.4f}  AI: {weights[1]:.4f}")

    optimizer    = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    total_steps  = len(train_loader) * CFG.EPOCHS
    scheduler    = get_linear_schedule_with_warmup(optimizer,
                       num_warmup_steps=int(total_steps*CFG.WARMUP_RATIO),
                       num_training_steps=total_steps)

    stopper      = EarlyStopping()
    plotter      = LivePlot()
    best_val_acc = 0.0

    for epoch in range(1, CFG.EPOCHS+1):
        print(f"\n── Epoch {epoch}/{CFG.EPOCHS} ─────────────────────")
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, loss_fn)
        vl_loss, vl_acc = evaluate(model, val_loader, loss_fn)
        print(f"  Train  loss={tr_loss:.4f}  acc={tr_acc*100:.2f}%")
        print(f"  Val    loss={vl_loss:.4f}  acc={vl_acc*100:.2f}%")

        # Flag if val is suspiciously higher than train — likely memorization
        if vl_acc > tr_acc + 0.03:
            print(f"  ⚠️  Warning: val acc much higher than train — possible data leakage or easy dataset")

        plotter.update(epoch, tr_loss, vl_loss, tr_acc, vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            os.makedirs(CFG.SAVE_DIR, exist_ok=True)
            model.save_pretrained(CFG.SAVE_DIR)
            tokenizer.save_pretrained(CFG.SAVE_DIR)
            print(f"  Saved best model (val_acc={vl_acc*100:.2f}%)")

        if stopper.step(vl_loss, epoch):
            print(f"\nEarly stop. Best epoch: {stopper.best_epoch}")
            break

    plotter.save()

    print("\nEvaluating on test set...")
    model = DistilBertForSequenceClassification.from_pretrained(CFG.SAVE_DIR).to(CFG.DEVICE)
    _, _, preds, truths, all_probs = evaluate(model, test_loader, loss_fn, return_probs=True)
    print_metrics(preds, truths, all_probs)
    print(f"\nDone. Model saved to {CFG.SAVE_DIR}")
    print("[Close graph window to exit]")
    plotter.keep_open()


if __name__ == "__main__":
    main()
