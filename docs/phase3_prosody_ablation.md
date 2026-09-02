# Phase 3 Prosody Ablation Study: 2-Signal vs 3-Signal Fusion

This document presents the empirical results of incorporating **Prosodic Features (Fusion Signal 3)** into Convora's turn-boundary detection pipeline, evaluated against official human-annotated turn boundaries from the **AMI Meeting Corpus** (`ES2002a`, 4 speakers, 1,272 seconds).

---

## 1. Executive Summary & PRD Section 2.5 Validation Verdict

Per PRD Section 2.5, the validation rule explicitly states:
> *"Prosody is kept only if it improves the result."*

### **Verdict: KEPT (With Documented Methodological Limitations)**

Integrating prosodic features extracted via Parselmouth (Praat pitch and intensity tracking) improves turn-boundary detection across both full-dataset metrics and held-out cross-validation:

* **Full Dataset (ES2002a)**:
  * **Recall (GT-centric)** increases from **29.02%** to **33.04%** (**+4.02%** absolute gain, +9 true boundaries hit).
  * **FNR (GT-centric)** falls from **70.98%** to **66.96%** (**-4.02%** relative error reduction).
  * **Precision** increases from **63.44%** to **67.00%** (**+3.56%** absolute gain).
  * **False Positive Rate (Early Cutoff)** decreases from **18.28%** to **17.74%** (**-0.54%**).
  * **F1 Score** increases from **0.3939** to **0.4381** (**+0.0442**).
* **Held-Out Generalization (50/50 Temporal Split)**:
  * On completely unseen Half B (trained only on Half A): Recall gains **+5.61%** (35.96% $\rightarrow$ 41.57%) and F1 gains **+0.0439** (0.4433 $\rightarrow$ 0.4872).
  * On completely unseen Half A (trained only on Half B): Recall gains **+2.97%** (24.44% $\rightarrow$ 27.41%) and F1 gains **+0.0370** (0.3547 $\rightarrow$ 0.3917).

> [!WARNING]
> **Stated Methodology Limitation & Overfitting Warning**:
> The full-dataset weights (`sem=0.20`, `pause=0.20`, `spk=0.40`, `pros=0.20`, `threshold=0.55`) were selected via grid search on the same 224 boundaries being reported in the full comparison table. Because this was a second round of weight optimization against a single meeting (`ES2002a`), full-dataset metrics alone carry an inherent risk of overfitting. While the held-out temporal cross-validation below confirms within-meeting generalization across speakers, **cross-corpus and cross-acoustic domain generalization remains unverified until evaluated on additional external meetings.**

---

## 2. Full-Dataset Evaluation Comparison Table (+/-0.5s Tolerance)

Evaluated against **224 genuine floor-transfer turn boundaries** (excluding listener backchannels) using the standardized GT-centric nearest-neighbor matching logic in [`eval/gt_matching.py`](file:///c:/Users/shrut/convora/eval/gt_matching.py):

| Metric | 2-Signal Fusion (Base) | 3-Signal Fusion (+Prosody) | Delta | Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Recall (GT-centric)** | 29.02% | **33.04%** | **+4.02%** | Recovers 9 additional true boundaries |
| **False Negative Rate (FNR)** | 70.98% | **66.96%** | **-4.02%** | Reduces missed turn transitions |
| **Precision (Candidate-centric)** | 63.44% | **67.00%** | **+3.56%** | Higher confidence on triggered EOS |
| **False Positive Rate (FPR)** | 18.28% | **17.74%** | **-0.54%** | Lowers early speaker interruptions |
| **F1 Score (GT-centric)** | 0.3939 | **0.4381** | **+0.0442** | Solid improvement in balanced accuracy |
| **Overall Accuracy** | 73.78% | **76.92%** | **+3.14%** | Improves general candidate classification |
| **GT Boundaries Hit (TP)** | 65 | **74** | **+9** | 9 additional human turn boundaries hit |
| **GT Misses: FN_FUSION** | 49 | **40** | **-9** | Fusion misclassifications reduced by 18.4% |
| **GT Misses: FN_VAD (Gaps)** | 110 | 110 | 0 | Unchanged (governed by 300ms pause trigger) |
| **Early Cutoffs (FP)** | 34 | **33** | **-1** | 1 fewer false alarm on natural pauses |

---

## 3. Held-Out Generalization Evaluation (Chronological 50/50 Split)

To address overfitting risk and verify true generalization, meeting `ES2002a` was partitioned at the temporal midpoint ($t = 636.32$s):
* **Half A (First Half, $0 \le t < 636$s)**: 135 GT boundaries, 156 candidates.
* **Half B (Second Half, $636 \le t \le 1273$s)**: 89 GT boundaries, 130 candidates.

### 3.1 Fold 1: Fit on Half A $\rightarrow$ Test ONLY on Held-Out Half B
Weights were optimized via grid search strictly on Half A (`sem=0.20, pause=0.20, spk=0.35, pros=0.25, threshold=0.52`) and evaluated out-of-sample on Half B:

| Pipeline Variant on Held-Out Half B | Recall (GT) | Precision | FPR (Early Cut) | F1 Score | GT Hits (TP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2-Signal Baseline** (Half B) | 35.96% | 62.22% | **20.48%** | 0.4433 | 32 / 89 |
| **3-Signal Fusion** (Weights from Half A) | **41.57%** | **62.75%** | 22.89% | **0.4872** | **37 / 89** |
| **Held-Out Delta (Fold 1)** | **+5.61%** | **+0.53%** | *+2.41%* | **+0.0439** | **+5 Hits** |

### 3.2 Fold 2: Fit on Half B $\rightarrow$ Test ONLY on Held-Out Half A
Weights were optimized strictly on Half B (`sem=0.20, pause=0.15, spk=0.45, pros=0.20, threshold=0.60`) and evaluated out-of-sample on Half A:

| Pipeline Variant on Held-Out Half A | Recall (GT) | Precision | FPR (Early Cut) | F1 Score | GT Hits (TP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2-Signal Baseline** (Half A) | 24.44% | 64.58% | 16.50% | 0.3547 | 33 / 135 |
| **3-Signal Fusion** (Weights from Half B) | **27.41%** | **68.63%** | **15.53%** | **0.3917** | **37 / 135** |
| **Held-Out Delta (Fold 2)** | **+2.97%** | **+4.05%** | **-0.97%** | **+0.0370** | **+4 Hits** |

**Conclusion from Cross-Validation**: Prosody improves recall and F1 score out-of-sample in both directions, confirming that the prosody signal provides genuine predictive value rather than arbitrary noise fitting.

---

## 4. Tolerance Sensitivity Analysis (Full Dataset)

The performance lift from prosody is robust across all evaluated tolerance windows:

| Tolerance Window | 2-Signal Recall | 3-Signal Recall | 2-Signal Precision | 3-Signal Precision | 2-Signal F1 | 3-Signal F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+/-0.3s** | 19.64% | **21.88%** | 44.09% | **46.00%** | 0.2697 | **0.2944** |
| **+/-0.5s** (Baseline) | 29.02% | **33.04%** | 63.44% | **67.00%** | 0.3939 | **0.4381** |
| **+/-0.8s** | 34.82% | **38.84%** | 75.27% | **76.00%** | 0.4624 | **0.5021** |

---

## 5. Acoustic Extraction Methodology & Edge Cases

Prosodic feature extraction is implemented in [`detection/prosody.py`](file:///c:/Users/shrut/convora/detection/prosody.py) using `praat-parselmouth` (Python interface to Praat):

### 5.1 Extracted Features (over 300ms Pre-Pause Window)
1. **Pitch Slope ($Hz/s$)**: Linear regression over Praat pitch tracker voiced frames ($75-500$ Hz). Terminal declarative drops ($\le -30$ Hz/s) signal completion; rising pitch contours ($\ge +30$ Hz/s) signal questions, listing, or turn-holding.
2. **Intensity Slope ($dB/s$)**: Linear regression of intensity contour. Decrescendo ($< -15$ dB/s) indicates trailing volume at utterance completion.
3. **Pre-Boundary Lengthening (Relative Syllable Duration)**: Ratio of the final syllable nucleus duration to the speaker's running average syllable duration tracked across the conversation.

### 5.2 Extraction Breakdown on 286 Real Candidates
Acoustic extraction across all 286 pause candidates in `ES2002a.Mix-Headset.wav` yielded:

* **Valid Extraction (`ok`)**: **194 candidates (67.8%)** — Both pitch and intensity contours were extracted with $\ge 3$ voiced frames and mean energy $\ge 20$ dB.
* **Low-Energy / Silence (`low_energy`)**: **56 candidates (19.6%)** — Pre-pause audio fell below 20 dB (e.g. whispered trailing speech, soft breath, or background ambient silence). Gracefully handled by falling back to neutral prior ($0.50$).
* **Unvoiced Speech (`unvoiced`)**: **36 candidates (12.6%)** — Trailing phonemes contained unvoiced fricatives/stops (e.g., `/s/`, `/t/`, `/k/`) with fewer than 3 voiced pitch frames. Gracefully handled by falling back to neutral prior ($0.50$).
* **Too Short (`too_short`)**: **0 candidates (0.0%)** — All candidate pre-pause windows were valid $\ge 50$ms segments.

---

## 6. Sample Extracted Prosody & Evidence

The table below shows real extracted acoustic measurements from the audio file across the first 10 candidate pause points:

| PauseStart | Status | Pitch Slope | Intensity Slope | Relative Duration | Mapped Prosody Score | Fragment / Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `7.84s` | `ok` | **-15.1 Hz/s** | **-18.2 dB/s** | 0.77x | `0.577` | *my gosh you've already produced a powerpoint* |
| `9.44s` | `ok` | **-138.7 Hz/s** | +35.8 dB/s | 0.97x | `0.617` | *i think it's already on actually* |
| `16.23s` | `ok` | **+183.1 Hz/s** | -165.7 dB/s | 0.77x | `0.375` | *think it's already on actually god how* |
| `35.84s` | `ok` | **+35.5 Hz/s** | -17.4 dB/s | 0.50x | `0.408` | *make this thing work i've plugged it in* |
| `39.20s` | `unvoiced` | N/A | -49.2 dB/s | N/A | `0.500` | *it's got it* |
| `39.68s` | `low_energy` | N/A | N/A | N/A | `0.500` | *okay* |
| `40.80s` | `ok` | **+22.3 Hz/s** | +11.1 dB/s | 1.07x | `0.408` | *okay right* |
| `42.35s` | `unvoiced` | N/A | +153.2 dB/s | N/A | `0.500` | *okay right kinda all* |
| `49.08s` | `ok` | **-476.0 Hz/s** | -67.6 dB/s | 0.31x | `0.785` | *okay right kinda all okay* |
| `53.88s` | `ok` | **+742.8 Hz/s** | -33.7 dB/s | 1.17x | `0.391` | *okay right kinda all okay right* |

---

## 7. Qualitative Analysis: Why Prosody Helps

Prosody addresses the core failure mode discovered in Phase 3 evaluation: **pragmatic ambiguity where transcripts look syntactically incomplete, but prosodic pitch contours clearly signal turn-release or continuation.**

### 7.1 True Turn Boundaries Recovered (False Negative -> True Positive)
* **Candidate at `507.74s` (*"so basically 12 alright yeah k so"*):**
  * Semantic judge voted `incomplete` with confidence $0.90$ (due to trailing fragment structure).
  * 2-signal fusion voted `Cont` (Confidence $0.40 < 0.55$).
  * Pitch slope was **$-55.6$ Hz/s** (falling declarative contour, mapped prosody score **$0.873$**).
  * 3-signal fusion scored **$0.55 \ge 0.55$** $\rightarrow$ **Voted EOS (Hit GT boundary at 507.82s)**.
* **Candidate at `852.94s` (*"such and look at it that's a good"*):**
  * Trailing fragment misjudged by semantic model as cut off mid-thought.
  * Pitch slope dropped precipitously at **$-613.7$ Hz/s** (mapped prosody score **$0.766$**).
  * 3-signal fusion flipped decision from `Cont` $\rightarrow$ `EOS` (**Hit GT boundary at 853.03s**).

### 7.2 Mid-Turn Hesitations Protected (False Positive -> True Negative)
* **Candidate at `956.50s` (*"something like that yeah okay or"*):**
  * Trailing hesitation with silence gap.
  * Pitch slope was strongly rising at **$+724.2$ Hz/s** (prosody score **$0.117$**, indicating question/floor-holding intonation).
  * 3-signal fusion suppressed the early cutoff: confidence dropped from $0.57 \rightarrow 0.52$ $\rightarrow$ **Voted `Cont` (prevented false interruption)**.

---

## 8. Fusion Formulation & Tuned Weights

The fusion engine in [`detection/fusion.py`](file:///c:/Users/shrut/convora/detection/fusion.py) incorporates the prosody signal:

$$\text{Confidence} = w_{\text{sem}} \cdot V_{\text{sem}} + w_{\text{pause}} \cdot V_{\text{pause}} + w_{\text{spk}} \cdot V_{\text{spk}} + w_{\text{pros}} \cdot V_{\text{pros}}$$

### Active 3-Signal Configuration:
* `WEIGHT_SPEAKER_CHANGE = 0.40`
* `WEIGHT_PROSODY = 0.20`
* `WEIGHT_PAUSE = 0.20`
* `WEIGHT_SEMANTIC = 0.20`
* `DECISION_THRESHOLD = 0.55`
* `USE_PROSODY = True`

---

## 9. Reproducibility Instructions

To re-run the ablation study and verify results:

```bash
# Run the complete ablation study (full dataset + 2-fold held-out cross-validation):
python eval/ablation_prosody.py

# Re-run the ground-truth benchmark with 3-signal fusion:
python eval/evaluate_against_ami_ground_truth.py
```
