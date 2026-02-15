
import csv
import re
from pathlib import Path

def infer_label(name):
    n = name.lower()
    if n.startswith('y') or '/y/' in n:
        return 1
    if n.startswith('n') or '/n/' in n:
        return 0
    if any(k in n for k in ["tumor", "positive", "_1", "-1", "pos", " yes", "yes", "/yes/"]):
        return 1
    if any(k in n for k in ["no_tumor", "negative", "_0", "-0", "neg", " normal", "normal", " no", "no", "/no/"]):
        return 0
    return None

csv_path = Path('输出结果/cla_pre_dual.csv')
tp = 0
tn = 0
fp = 0
fn = 0

print(f"Reading {csv_path}...")
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = row['filename']
        pred = int(row['label'])
        gt = infer_label(fname)
        
        if gt is None:
            print(f"Skipping ambiguous file: {fname}")
            continue
            
        if pred == 1 and gt == 1:
            tp += 1
        elif pred == 0 and gt == 0:
            tn += 1
        elif pred == 1 and gt == 0:
            fp += 1
        elif pred == 0 and gt == 1:
            fn += 1

total = tp + tn + fp + fn
acc = (tp + tn) / total if total > 0 else 0
prec = tp / (tp + fp) if (tp + fp) > 0 else 0
rec = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

print("-" * 30)
print(f"Total: {total}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print("-" * 30)
