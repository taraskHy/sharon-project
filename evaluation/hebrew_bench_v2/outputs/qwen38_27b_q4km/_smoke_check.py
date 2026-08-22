import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
ids = sys.argv[2].split(",")
ok = err = empty = 0
for i in ids:
    p = run / (i + ".json")
    if not p.exists():
        err += 1
        continue
    r = json.loads(p.read_text(encoding="utf-8"))
    if r.get("error"):
        err += 1
    elif not (r.get("transcription") or "").strip():
        empty += 1
    else:
        ok += 1
print(ok, err, empty)
