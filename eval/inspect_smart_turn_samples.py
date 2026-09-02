import json

with open("eval/smart_turn_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

cres = data["tolerances"]["tol_0.5s"]["candidate_eval"]
details = cres["candidate_details"]
scored = data["scored_candidates"]

# Find 2 examples of each category: TP, FP, TN, FN, and Borderline (0.40 - 0.60)
tps, fps, tns, fns, borders = [], [], [], [], []

for i, (r, s) in enumerate(zip(details, scored)):
    status = r["match_status"]
    prob = s["smart_turn_probability"]
    delta = r["delta_s"]
    frag = r["fragment"]
    ts = r["pause_start"]
    spk = r.get("speaker", "N/A")
    item = (i, ts, spk, prob, status, delta, frag)

    if 0.40 <= prob <= 0.60:
        borders.append(item)
    if status == "TP" and len(tps) < 3:
        tps.append(item)
    elif status == "FP" and len(fps) < 3:
        fps.append(item)
    elif status == "TN" and len(tns) < 3:
        tns.append(item)
    elif status == "FN" and len(fns) < 3:
        fns.append(item)

print("=" * 120)
print(f"{'Idx':<4} | {'Pause(s)':<8} | {'Speaker':<7} | {'Prob':<8} | {'Pred':<10} | {'Status':<6} | {'Delta':<7} | {'Fragment / Context Text'}")
print("=" * 120)

all_selected = tps + fps + tns + fns + borders[:2]
# Sort by timestamp
all_selected.sort(key=lambda x: x[1])

for idx, ts, spk, prob, status, delta, frag in all_selected:
    pred = "Complete" if prob > 0.50 else "Incomplete"
    d_str = f"{delta:>6.3f}s" if delta is not None else "   N/A "
    print(f"{idx:<4} | {ts:>7.2f}s | {spk:<7} | {prob:>6.4f} | {pred:<10} | {status:<6} | {d_str} | {frag}")
