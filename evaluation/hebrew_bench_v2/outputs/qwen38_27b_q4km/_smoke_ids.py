import json
d = json.load(open(r"evaluation\unlimited_ocr\gate_items.json", encoding="utf-8"))
print(",".join(d["smoke_5_frozen"]))
