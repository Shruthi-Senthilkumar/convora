"""
detection/prosody.py

Prosodic Feature Extraction for Convora Turn-Boundary Detection.
Phase 3, Task 1 per PRD sec 2.5 ("Prosodic Features - Fusion Signal 3").

Extracts acoustic turn-boundary cues over the final 300ms window of speech
preceding a candidate pause using Parselmouth (Praat pitch/intensity tracking):
  1. Pitch slope (Hz/s): linear regression over voiced frames.
     - Falling pitch (negative slope) signals statement completion (turn finality).
     - Rising pitch (positive slope) signals question, listing, or turn holding.
  2. Intensity slope (dB/s): linear regression over intensity envelope.
     - Decrescendo (negative slope) indicates natural turn trailing-off.
  3. Final-syllable relative duration: ratio of pre-boundary syllable/nucleus
     duration to speaker's running average (pre-boundary lengthening cue).

Handles edge cases gracefully (unvoiced frames, silence/low energy, short audio).
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Union
from collections import defaultdict
import numpy as np
import parselmouth


@dataclass
class ProsodyFeatures:
    """
    Extracted prosodic cues over a pre-pause speech window.
    """
    pitch_slope: Optional[float] = None
    intensity_slope: Optional[float] = None
    final_syllable_duration: Optional[float] = None
    final_syllable_relative_duration: Optional[float] = None
    mean_intensity: Optional[float] = None
    voiced_fraction: Optional[float] = None
    is_available: bool = False
    status: str = "unavailable"  # "ok", "unvoiced", "low_energy", "too_short", "error", "unavailable"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SpeakerProsodyTracker:
    """
    Maintains running prosodic statistics (e.g. running average syllable duration)
    per speaker across an audio session.
    """
    def __init__(self, default_syllable_duration: float = 0.150):
        self.default_syllable_duration = default_syllable_duration
        self.history: Dict[Any, list] = defaultdict(list)

    def get_running_average_duration(self, speaker_id: Any) -> float:
        durs = self.history[speaker_id]
        if len(durs) >= 3:
            return float(np.mean(durs))
        return self.default_syllable_duration

    def update(self, speaker_id: Any, syllable_duration: float):
        if 0.030 <= syllable_duration <= 0.600:
            self.history[speaker_id].append(syllable_duration)


def extract_prosodic_features(
    audio_source: Union[str, Path, parselmouth.Sound],
    start_time: float,
    end_time: float,
    speaker_id: Optional[Any] = None,
    tracker: Optional[SpeakerProsodyTracker] = None,
    time_step: float = 0.005,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 500.0,
    min_voiced_frames: int = 3,
    min_intensity_db: float = 20.0,
) -> ProsodyFeatures:
    """
    Extract prosodic features from audio over the window [start_time, end_time].

    Parameters
    ----------
    audio_source : str, Path, or parselmouth.Sound
        Path to audio file or pre-loaded Parselmouth Sound object.
    start_time : float
        Start timestamp of analysis window in seconds (e.g. pause_start - 0.3s).
    end_time : float
        End timestamp of analysis window in seconds (e.g. pause_start).
    speaker_id : Any, optional
        Speaker identifier for speaker-normalized duration tracking.
    tracker : SpeakerProsodyTracker, optional
        Tracker maintaining per-speaker duration baselines.
    time_step : float
        Temporal resolution for pitch and intensity analysis (default 5ms).
    pitch_floor : float
        Minimum pitch in Hz (Praat default 75Hz).
    pitch_ceiling : float
        Maximum pitch in Hz (Praat default 500Hz).
    min_voiced_frames : int
        Minimum voiced frames required to compute pitch slope (default 3).
    min_intensity_db : float
        Energy floor in dB; below this is treated as low-energy/silence (default 20dB).

    Returns
    -------
    ProsodyFeatures
        Extracted features with availability status.
    """
    window_duration = end_time - start_time
    if window_duration < 0.050:
        return ProsodyFeatures(
            is_available=False,
            status="too_short"
        )

    try:
        # Load or slice sound
        if isinstance(audio_source, (str, Path)):
            sound_path = str(audio_source)
            sound = parselmouth.Sound(sound_path)
            part = sound.extract_part(from_time=start_time, to_time=end_time)
        elif isinstance(audio_source, parselmouth.Sound):
            part = audio_source.extract_part(from_time=start_time, to_time=end_time)
        else:
            return ProsodyFeatures(is_available=False, status="error")

        # 1. Extract Intensity Envelope
        intensity = part.to_intensity(time_step=time_step)
        i_vals = intensity.values[0]
        i_times = intensity.xs()

        if len(i_vals) == 0:
            return ProsodyFeatures(is_available=False, status="low_energy")

        mean_intensity = float(np.mean(i_vals))
        if mean_intensity < min_intensity_db:
            return ProsodyFeatures(
                mean_intensity=mean_intensity,
                is_available=False,
                status="low_energy"
            )

        # Intensity slope (dB/s)
        if len(i_vals) >= 3:
            i_slope, _ = np.polyfit(i_times, i_vals, 1)
            intensity_slope = float(i_slope)
        else:
            intensity_slope = 0.0

        # 2. Extract Pitch Contour (Praat pitch tracking)
        pitch = part.to_pitch(time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
        p_vals = pitch.selected_array['frequency']
        p_times = np.array([pitch.get_time_from_frame_number(k + 1) for k in range(len(p_vals))])
        
        voiced_mask = p_vals > 0
        voiced_count = int(np.sum(voiced_mask))
        total_frames = len(p_vals)
        voiced_fraction = float(voiced_count / total_frames) if total_frames > 0 else 0.0

        if voiced_count < min_voiced_frames:
            return ProsodyFeatures(
                intensity_slope=intensity_slope,
                mean_intensity=mean_intensity,
                voiced_fraction=voiced_fraction,
                is_available=False,
                status="unvoiced"
            )

        # Pitch slope (Hz/s) over voiced frames
        vt = p_times[voiced_mask]
        vf = p_vals[voiced_mask]
        p_slope, _ = np.polyfit(vt, vf, 1)
        pitch_slope = float(p_slope)

        # 3. Syllable Nucleus / Pre-boundary Lengthening
        # Find peak intensity and compute peak width at -3dB
        peak_idx = int(np.argmax(i_vals))
        peak_val = i_vals[peak_idx]
        above_3db = i_vals >= (peak_val - 3.0)
        final_syllable_dur = float(np.sum(above_3db) * time_step)

        # Relative duration vs running speaker average
        if tracker is not None and speaker_id is not None:
            running_avg = tracker.get_running_average_duration(speaker_id)
            rel_dur = float(final_syllable_dur / running_avg) if running_avg > 0 else 1.0
            tracker.update(speaker_id, final_syllable_dur)
        else:
            rel_dur = float(final_syllable_dur / 0.150)

        return ProsodyFeatures(
            pitch_slope=pitch_slope,
            intensity_slope=intensity_slope,
            final_syllable_duration=final_syllable_dur,
            final_syllable_relative_duration=rel_dur,
            mean_intensity=mean_intensity,
            voiced_fraction=voiced_fraction,
            is_available=True,
            status="ok"
        )

    except Exception as e:
        return ProsodyFeatures(
            is_available=False,
            status=f"error: {str(e)}"
        )


def get_prosody_completion_value(features: Union[ProsodyFeatures, Dict[str, Any], None]) -> float:
    """
    Maps extracted prosodic features to a completion probability on [0.0, 1.0].

    Mapping Logic:
    - If prosody is unavailable/unvoiced: returns neutral prior (0.50).
    - Pitch slope:
        * Strongly falling (< -50 Hz/s): statement finality -> score ~ 0.85 - 1.0
        * Flat (~ 0 Hz/s): neutral / continuation -> score ~ 0.50
        * Strongly rising (> +50 Hz/s): question / turn holding -> score ~ 0.0 - 0.20
    - Intensity slope:
        * Decrescendo (< -20 dB/s): trailing off -> higher completion
        * Crescendo / sustained (> +10 dB/s): holding turn -> lower completion
    - Relative Syllable Duration:
        * Lengthened (> 1.25x): pre-boundary lengthening -> higher completion
        * Rushed (< 0.85x): mid-sentence pace -> lower completion
    """
    if features is None:
        return 0.5

    if isinstance(features, dict):
        is_available = features.get("is_available", False)
        p_slope = features.get("pitch_slope")
        i_slope = features.get("intensity_slope")
        rel_dur = features.get("final_syllable_relative_duration")
    else:
        is_available = features.is_available
        p_slope = features.pitch_slope
        i_slope = features.intensity_slope
        rel_dur = features.final_syllable_relative_duration

    if not is_available or p_slope is None or i_slope is None:
        return 0.5

    # 1. Pitch component (logistic sigmoid mapped so -100 Hz/s -> ~0.90, +100 Hz/s -> ~0.10)
    # pitch_score: falling = high, rising = low
    # clip slope to [-150, 150]
    clipped_p = np.clip(p_slope, -150.0, 150.0)
    pitch_score = float(1.0 / (1.0 + np.exp(clipped_p / 45.0)))

    # 2. Intensity component (logistic sigmoid mapped so -40 dB/s -> ~0.80, +40 dB/s -> ~0.20)
    clipped_i = np.clip(i_slope, -60.0, 60.0)
    intensity_score = float(1.0 / (1.0 + np.exp(clipped_i / 20.0)))

    # 3. Syllable duration component (lengthening relative to baseline)
    if rel_dur is not None:
        # 0.5x -> 0.2, 1.0x -> 0.5, 1.5x -> 0.8
        dur_score = float(np.clip(0.5 + (rel_dur - 1.0) * 0.6, 0.0, 1.0))
    else:
        dur_score = 0.5

    # Weighted combination of sub-features (pitch 50%, intensity 30%, duration 20%)
    combined = 0.50 * pitch_score + 0.30 * intensity_score + 0.20 * dur_score
    return float(np.clip(combined, 0.0, 1.0))
