"""Smoke-test the annotation app end-to-end WITHOUT touching real data.

Builds a throwaway package root (first 3 train samples + their images
copied from the built pilot package), points the app at it via
HTR_PILOT_ROOT, and drives it with streamlit's AppTest:

1. app renders; image paths resolve;
2. typing a dummy transcription + "Save and next" writes the record
   (autosave) and advances;
3. a FRESH app session resumes at the first undecided sample;
4. "Whole line unreadable" forces the token record;
5. editing then navigating Previous autosaves a draft;
6. all writes land in the temp root only.

    .venv/Scripts/python.exe scripts/htr_annotation_smoke.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.htr_annotation_lib import UNREADABLE_TOKEN  # noqa: E402

REAL = Path("evaluation/htr_pilot")
N = 3


def build_temp_root(tmp: Path) -> list[dict]:
    samples = json.loads((REAL / "splits/train.json").read_text(encoding="utf-8"))[:N]
    (tmp / "splits").mkdir(parents=True)
    for split in ("train", "val", "internal_test"):
        recs = samples if split == "train" else []
        (tmp / "splits" / f"{split}.json").write_text(
            json.dumps(recs, ensure_ascii=False), encoding="utf-8")
        (tmp / "annotations" / split).mkdir(parents=True)
    for s in samples:
        for rel in s["images"].values():
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copyfile(REAL / rel, dst)
    return samples


def find_button(at, label_part: str):
    for b in at.button:
        if label_part in b.label:
            return b
    raise AssertionError(f"button {label_part!r} not found: "
                         f"{[b.label for b in at.button]}")


def main() -> int:
    from streamlit.testing.v1 import AppTest

    assert (REAL / "splits/train.json").exists(), "pilot package not built"
    tmp = Path(tempfile.mkdtemp(prefix="htr_smoke_"))
    try:
        samples = build_temp_root(tmp)
        os.environ["HTR_PILOT_ROOT"] = str(tmp)

        def fresh():
            at = AppTest.from_file("scripts/htr_annotation_app.py",
                                   default_timeout=30)
            at.run()
            assert not at.exception, at.exception
            return at

        # 1) renders + images resolve (st.image would fail on missing files)
        at = fresh()
        assert any(samples[0]["sample_id"] in h.value for h in at.subheader), \
            "first sample not shown"
        print("1. app renders, sample 1 shown, images resolve")

        # 2) dummy transcription + Save and next -> autosaved record
        at.text_area[0].set_value("טקסט בדיקה — לא תווית אמת").run()
        find_button(at, "Save and next").click().run()
        assert not at.exception, at.exception
        rec_path = tmp / "annotations/train" / f"{samples[0]['sample_id']}.json"
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        assert rec["status"] == "ok" and rec["human_verified"]
        assert rec["transcription"].startswith("טקסט בדיקה")
        assert any(samples[1]["sample_id"] in h.value for h in at.subheader), \
            "did not advance to sample 2"
        print("2. Save and next: record written atomically, advanced")

        # 3) fresh session resumes at first undecided (= sample 2)
        at2 = fresh()
        assert any(samples[1]["sample_id"] in h.value for h in at2.subheader), \
            "resume did not land on sample 2"
        print("3. fresh session resumed at first undecided sample")

        # 4) unreadable button forces the token
        find_button(at2, "unreadable").click().run()
        rec2 = json.loads((tmp / "annotations/train" /
                           f"{samples[1]['sample_id']}.json").read_text(encoding="utf-8"))
        assert rec2["transcription"] == UNREADABLE_TOKEN and rec2["unreadable"]
        print("4. unreadable_full: token record written")

        # 5) edit then Previous -> draft autosave for sample 3
        at3 = fresh()  # resumes at sample 3
        assert any(samples[2]["sample_id"] in h.value for h in at3.subheader)
        at3.text_area[0].set_value("טיוטה זמנית").run()
        find_button(at3, "Previous").click().run()
        draft = json.loads((tmp / "annotations/train" /
                            f"{samples[2]['sample_id']}.json").read_text(encoding="utf-8"))
        assert draft["status"] == "draft" and not draft["human_verified"]
        print("5. navigation autosaved a draft (not verified)")

        # 6) nothing leaked into the real package
        real_ann = list((REAL / "annotations").rglob("*.json"))
        assert not real_ann, f"real annotations touched: {real_ann[:3]}"
        print("6. real package untouched (dummy annotations in temp only)")
        print("SMOKE: PASS")
        return 0
    finally:
        os.environ.pop("HTR_PILOT_ROOT", None)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
