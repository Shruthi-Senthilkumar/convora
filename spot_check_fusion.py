import json

with open("eval/fusion_result.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for r in data[:15]:
    c = r["candidate"]
    f_res = r["fusion"]
    decision = str(f_res["is_end_of_speech"])
    conf = f_res["confidence"]
    fragment_tail = c["fragment"][-50:]
    print(f"{decision:<7} conf={conf:.2f}  {fragment_tail}")
