"""Minimal CRNN+CTC line recognizer for the HTR pilot (plain torch).

Engine decision is documented in evaluation/htr_pilot_gates.md: PyLaia is
not installable on this machine's Python 3.12, so the pilot uses this
self-contained trainer (conv stack + BiLSTM + CTC — the standard
PyLaia/kraken recipe) on torch 2.13+cu126. No other dependencies.

Data comes exclusively from scripts/htr_train_prepare.py's workspace
(train/val lists, char symbols, fixed-height 128 px line images). Model
selection is on val only; internal_test decoding requires
--allow-internal-test (single final report).

    .venv-train/Scripts/python.exe scripts/htr_pilot_train.py train \
        [--epochs 250] [--batch-size 16] [--lr 3e-4] [--patience 30]
    .venv-train/Scripts/python.exe scripts/htr_pilot_train.py decode \
        --split val --out decodes/val_trial01.txt

Decode output: "<id> <confidence> <text>" per line — the format
scripts/htr_pilot_eval.py consumes. Confidence is exp(mean log max
posterior) over non-blank frames: the model's own calibration, no LM.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WS = Path("evaluation/htr_train_workspace")
SEED = 20260714
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_syms(ws: Path) -> tuple[dict[str, int], list[str]]:
    pairs = [line.split(" ", 1) for line in
             (ws / "syms.txt").read_text(encoding="utf-8").splitlines() if line]
    sym2id = {s: int(i) for s, i in pairs}
    id2sym = [None] * len(sym2id)
    for s, i in sym2id.items():
        id2sym[i] = s
    assert id2sym[0] == "<ctc>", "symbol 0 must be the CTC blank"
    return sym2id, id2sym


def read_text_table(path: Path) -> dict[str, list[str]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sid, *syms = line.split(" ")
            out[sid] = syms
    return out


class Lines(Dataset):
    def __init__(self, ws: Path, split: str, sym2id: dict[str, int],
                 with_labels: bool = True):
        self.dir = ws / "imgs" / split
        self.ids = [s for s in (ws / "lists" / f"{split}.txt")
                    .read_text(encoding="utf-8").splitlines() if s]
        self.labels = read_text_table(ws / "text" / f"{split}.txt") \
            if with_labels else {}
        self.sym2id = sym2id

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        import cv2
        img = cv2.imread(str(self.dir / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        x = 1.0 - torch.from_numpy(img).float() / 255.0  # ink = high
        y = torch.tensor([self.sym2id[s] for s in self.labels.get(sid, [])],
                         dtype=torch.long)
        return sid, x, y


def collate(batch):
    sids, xs, ys = zip(*batch)
    widths = [x.shape[1] for x in xs]
    W = max(widths)
    imgs = torch.zeros(len(xs), 1, xs[0].shape[0], W)
    for i, x in enumerate(xs):
        imgs[i, 0, :, :x.shape[1]] = x
    ylens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    return sids, imgs, torch.tensor(widths), torch.cat(ys) if ys else None, ylens


class CRNN(nn.Module):
    """Conv stack (H128 -> 8, W/4) -> BiLSTM(2x256) -> per-frame logits.

    Width is subsampled only 4x: at 8x the narrowest real lines drop to
    ~1.3 frames per label symbol and CTC cannot align (verified in the
    overfit diagnostics); 4x keeps >= ~2.6 and typically 4-6."""

    def __init__(self, n_syms: int):
        super().__init__()
        ch = [1, 16, 32, 48, 64]
        blocks = []
        for i in range(4):
            blocks += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1),
                       nn.BatchNorm2d(ch[i + 1]), nn.LeakyReLU(0.1)]
            # pool H always; pool W only in the first 2 blocks (W/4)
            blocks += [nn.MaxPool2d((2, 2) if i < 2 else (2, 1))]
        self.cnn = nn.Sequential(*blocks)
        self.rnn = nn.LSTM(64 * 8, 256, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(512, n_syms)

    def forward(self, x):  # x: B,1,128,W
        f = self.cnn(x)                       # B,64,8,W/4
        B, C, H, W = f.shape
        f = f.permute(0, 3, 1, 2).reshape(B, W, C * H)
        out, _ = self.rnn(f)
        return self.fc(out)                   # B,W/4,n_syms

    @staticmethod
    def out_widths(widths: torch.Tensor) -> torch.Tensor:
        return torch.clamp(widths // 4, min=1)


def greedy_decode(logits: torch.Tensor, out_w: int, id2sym: list[str]
                  ) -> tuple[str, float]:
    """Returns LOGICAL-order text (display->logical via the same involution
    used at label-preparation time) plus a confidence."""
    from scripts.htr_train_prepare import to_display_order
    lp = F.log_softmax(logits[:out_w], dim=-1)
    best = lp.argmax(-1)
    conf_terms, syms, prev = [], [], 0
    for t in range(out_w):
        k = int(best[t])
        if k != 0:
            conf_terms.append(float(lp[t, k]))
            if k != prev:
                syms.append(id2sym[k])
        prev = k
    text = "".join(" " if s == "<space>" else s for s in syms)
    conf = math.exp(sum(conf_terms) / len(conf_terms)) if conf_terms else 0.0
    return to_display_order(text).strip(), conf


def cer(a: str, b: str) -> float:
    if not b:
        return 1.0 if a else 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(b)


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # cuDNN's RNN-backward teardown fail-fasts (0xC0000409) on this
    # Windows + torch 2.13+cu126 combination, corrupting exit codes even
    # under TerminateProcess (isolated 2026-07-14). Native kernels are
    # fast enough at pilot scale and make runs bit-deterministic too.
    torch.backends.cudnn.enabled = False


def log_trial(ws: Path, kind: str, args: dict) -> None:
    with (ws / "trials.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": kind, "at": time.strftime("%F %T"),
                            **args}, ensure_ascii=False) + "\n")


def hard_exit(code: int) -> None:
    """Exit while CUDA objects are still alive: freeing them (frame cleanup
    or DLL detach) fail-fasts with 0xC0000409 on Windows torch+cu126 and
    corrupts the exit code of an otherwise successful run."""
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.TerminateProcess(
            ctypes.windll.kernel32.GetCurrentProcess(), code)
    import os
    os._exit(code)


def evaluate(model, loader, id2sym, device) -> tuple[float, list]:
    model.eval()
    rows, cs = [], []
    with torch.no_grad():
        for sids, imgs, widths, _y, _yl in loader:
            logits = model(imgs.to(device))
            ows = CRNN.out_widths(widths)
            for i, sid in enumerate(sids):
                text, conf = greedy_decode(logits[i].cpu(), int(ows[i]), id2sym)
                rows.append((sid, conf, text))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(WS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--epochs", type=int, default=250)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--patience", type=int, default=30)
    tr.add_argument("--min-epochs", type=int, default=0,
                    help="no early stop before this epoch (CTC spends long "
                         "in the all-blank phase with val CER pinned at 1.0)")
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    de = sub.add_parser("decode")
    de.add_argument("--split", default="val")
    de.add_argument("--out", required=True)
    de.add_argument("--allow-internal-test", action="store_true")
    de.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ws = Path(args.workspace)
    set_seed()
    sym2id, id2sym = load_syms(ws)
    device = torch.device(args.device)
    model_path = ws / "model" / "crnn_best.pt"

    if args.cmd == "train":
        train_ds = Lines(ws, "train", sym2id)
        val_ds = Lines(ws, "val", sym2id)
        if not len(train_ds):
            print("empty train list — run htr_train_prepare.py after annotating")
            return 3
        g = torch.Generator().manual_seed(SEED)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=g, collate_fn=collate, num_workers=0)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)
        model = CRNN(len(id2sym)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        # Schedule on TRAIN LOSS: val CER is pinned at 1.0 through the CTC
        # blank-collapse phase, and stepping on it strangles the LR before
        # the model can escape (observed in the overfit diagnostics).
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, "min", factor=0.5, patience=25, min_lr=1e-5)
        ctc = nn.CTCLoss(blank=0, zero_infinity=True)
        from scripts.htr_train_prepare import to_display_order
        val_refs = {  # text tables hold display order; compare in logical
            sid: to_display_order(
                "".join(" " if s == "<space>" else s for s in syms)).strip()
            for sid, syms in read_text_table(ws / "text" / "val.txt").items()}
        best, bad_epochs = float("inf"), 0
        model_path.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            tot = n = 0
            for _sids, imgs, widths, y, ylens in train_dl:
                opt.zero_grad()
                logits = model(imgs.to(device))
                lp = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
                loss = ctc(lp, y.to(device), CRNN.out_widths(widths), ylens)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += loss.detach().item() * len(widths)
                n += len(widths)
            rows = evaluate(model, val_dl, id2sym, device) if len(val_ds) else []
            if rows:
                val_cer = sum(cer(t, val_refs.get(sid, "")) for sid, _c, t in rows) / len(rows)
            else:
                val_cer = tot / max(n, 1)  # no val labels yet: select on loss
            train_loss = tot / max(n, 1)
            sched.step(train_loss)
            marker = ""
            if val_cer < best - 1e-4:
                best, bad_epochs = val_cer, 0
                torch.save({"model": model.state_dict(), "n_syms": len(id2sym),
                            "epoch": epoch, "val_cer": val_cer}, model_path)
                marker = "  <- saved"
            else:
                bad_epochs += 1
            print(f"epoch {epoch:3d} loss {train_loss:.4f} "
                  f"val_cer {val_cer:.4f}{marker}")
            if bad_epochs >= args.patience and epoch >= args.min_epochs:
                print(f"early stop (patience {args.patience})")
                break
        log_trial(ws, "train", {"epochs_run": epoch, "batch_size": args.batch_size,
                                "lr": args.lr, "best_val_cer": round(best, 4),
                                "train_lines": len(train_ds),
                                "val_lines": len(val_ds), "device": str(device)})
        print(f"best val CER {best:.4f} -> {model_path}")
        hard_exit(0)

    if args.cmd == "decode":
        if args.split == "internal_test" and not args.allow_internal_test:
            print("REFUSING: internal_test decode without --allow-internal-test "
                  "(single final report only — evaluation/htr_pilot_gates.md)")
            return 2
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        model = CRNN(ckpt["n_syms"]).to(device)
        model.load_state_dict(ckpt["model"])
        ds = Lines(ws, args.split, sym2id, with_labels=False)
        dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)
        rows = evaluate(model, dl, id2sym, device)
        outp = ws / args.out
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("".join(f"{sid} {conf:.4f} {text}\n"
                                for sid, conf, text in rows), encoding="utf-8")
        log_trial(ws, "decode", {"split": args.split, "out": str(outp),
                                 "n": len(rows), "ckpt_epoch": ckpt["epoch"]})
        print(f"{len(rows)} lines -> {outp}")
        hard_exit(0)

    return 1


if __name__ == "__main__":
    sys.exit(main())
