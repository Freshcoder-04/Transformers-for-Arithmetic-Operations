#!/usr/bin/env python3
"""
analyze.py
===========

Compare three pretrained ArithmeticTransformer checkpoints on the
test and generalization splits, producing:

  • Exact‐match accuracy
  • Character‐level accuracy
  • Perplexity

and saving both a color‐coded Excel file and a plain CSV of predictions
for each split.

Requires:
  • ArithmeticTransformer.py defining `ArithmeticTransformer`
    with a `.generate(...)` method and `forward(src, tgt_input)`.
  • Trainer.py defining `SimpleTokenizer` with `.encode(...), .decode(...)`.
"""

import os
import math
import random
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from os import path

from ArithmeticTransformer import ArithmeticTransformer
from train import SimpleTokenizer

# ------- USER CONFIG ------------------------------------------------- #

MODEL_CONFIGS = [
    {
        "ckpt": "models/M1/models/best_model.pt",
        "d_model": 256,
        "num_heads": 8,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "d_ff": 2048,
        "dropout": 0.0,
        "max_length": 20,
    },
    {
        "ckpt": "models/M2/models/best_model.pt",
        "d_model": 256,
        "num_heads": 8,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "d_ff": 2048,
        "dropout": 0.0,
        "max_length": 20,
    },
    {
        "ckpt": "models/M3/models/best_model.pt",
        "d_model": 128,
        "num_heads": 4,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "d_ff": 1024,
        "dropout": 0.0,
        "max_length": 20,
    },
]

DATA_DIR        = "data"            # must contain test.csv & generalization.csv
OUTPUT_DIR      = "analysis_out"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE_EVAL = 1024
SEED            = 42

# --------------------------------------------------------------------- #

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_models(tokenizer):
    loaded = []
    for idx, cfg in enumerate(MODEL_CONFIGS, 1):
        model = ArithmeticTransformer(
            input_vocab_size  = len(tokenizer.vocab),
            output_vocab_size = len(tokenizer.vocab),
            d_model            = cfg["d_model"],
            num_heads          = cfg["num_heads"],
            num_encoder_layers = cfg["num_encoder_layers"],
            num_decoder_layers = cfg["num_decoder_layers"],
            d_ff               = cfg["d_ff"],
            dropout            = cfg["dropout"],
            max_seq_length     = cfg["max_length"],
            pad_token_id       = tokenizer.vocab["<pad>"],
        ).to(DEVICE)

        # load weights
        ckpt = torch.load(cfg["ckpt"], map_location=DEVICE)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        model.eval()

        name = f"model_{idx}"
        print(f"Loaded {name} from {cfg['ckpt']}")
        loaded.append((name, model, cfg["max_length"]))
    return loaded

def batch_generate(models, input_tensor, tokenizer):
    all_preds = [[] for _ in models]
    with torch.no_grad():
        for i in tqdm(range(0, input_tensor.size(0), BATCH_SIZE_EVAL), desc="Generating"):
            batch = input_tensor[i : i + BATCH_SIZE_EVAL].to(DEVICE)
            for m_idx, (_, model, ml) in enumerate(models):
                outs = model.generate(
                    batch,
                    max_length     = ml,
                    start_token_id = tokenizer.vocab["<sos>"],
                    end_token_id   = tokenizer.vocab["<eos>"],
                )
                for seq in outs:
                    all_preds[m_idx].append(tokenizer.decode(seq.cpu().numpy()))
    return all_preds

def style_and_save(df, pred_cols, out_excel):
    def color_cell(val, target):
        return (
            "background-color:#C6EFCE;color:#006100"
            if val == target
            else "background-color:#FFC7CE;color:#9C0006"
        )

    styler = df.style
    for col in pred_cols:
        styler = styler.apply(
            lambda s: [color_cell(v, t) for v, t in zip(s, df["target"])],
            subset=[col], axis=0
        )

    os.makedirs(os.path.dirname(out_excel), exist_ok=True)
    styler.to_excel(out_excel, engine="xlsxwriter", index=False)
    print(f"Saved Excel report: {out_excel}")

def evaluate_split(csv_name, models, tokenizer):
    df = pd.read_csv(path.join(DATA_DIR, csv_name), dtype=str)
    N = len(df)
    print(f"\n=== Evaluating {csv_name} ({N} examples) ===")

    # --- 1) Encode inputs & targets ---
    max_len_all = max(cfg["max_length"] for cfg in MODEL_CONFIGS)
    inputs = [tokenizer.encode(x, max_len_all) for x in df["input"]]
    targets = [tokenizer.encode(x, max_len_all) for x in df["target"]]
    input_ids  = torch.tensor(inputs,  dtype=torch.long)
    target_ids = torch.tensor(targets, dtype=torch.long).to(DEVICE)

    # --- 2) Generate predictions + compute exact/char acc ---
    preds_per_model = batch_generate(models, input_ids, tokenizer)
    pred_cols = []

    for (name, _, _), preds in zip(models, preds_per_model):
        col = f"{name}_pred"
        df[col] = preds
        pred_cols.append(col)

        # exact-match
        exact = (df[col] == df["target"]).mean()

        # char-level
        char_scores = []
        for p, t in zip(df[col], df["target"]):
            matches = sum(pc == tc for pc, tc in zip(p, t))
            char_scores.append(matches / max(len(t), 1))
        char_acc = sum(char_scores) / N

        print(f"  {name}: exact-match = {exact:.2%}, char-accuracy = {char_acc:.2%}")

    # --- 3) Compute perplexity via teacher-forced loss ---
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.vocab["<pad>"], reduction="sum"
    )
    for name, model, _ in models:
        total_loss = 0.0
        total_tok  = 0
        model.eval()

        with torch.no_grad():
            for i in range(0, N, BATCH_SIZE_EVAL):
                src = input_ids[i:i+BATCH_SIZE_EVAL].to(DEVICE)
                tgt = target_ids[i:i+BATCH_SIZE_EVAL]

                decoder_input = tgt[:, :-1]
                labels        = tgt[:, 1:]

                logits = model(src, decoder_input)  # [B, T, V]
                B, T, V = logits.size()

                # use reshape to avoid contiguity error
                loss = criterion(
                    logits.reshape(-1, V),
                    labels.reshape(-1)
                )
                total_loss += loss.item()
                total_tok  += (labels != tokenizer.vocab["<pad>"]).sum().item()

        ppl = math.exp(total_loss / total_tok)
        print(f"  {name}: perplexity = {ppl:.4f}")

    # --- 4) Save CSV + Excel ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv   = path.join(OUTPUT_DIR, csv_name.replace(".csv", "_predictions.csv"))
    out_excel = path.join(OUTPUT_DIR, csv_name.replace(".csv", "_predictions.xlsx"))

    df.to_csv(out_csv, index=False)
    print(f"Saved CSV predictions: {out_csv}")

    style_and_save(df[["input", "target"] + pred_cols], pred_cols, out_excel)

def main():
    set_seed()
    tokenizer = SimpleTokenizer()
    models    = load_models(tokenizer)

    for split in ["test.csv", "generalization.csv"]:
        evaluate_split(split, models, tokenizer)

if __name__ == "__main__":
    main()
