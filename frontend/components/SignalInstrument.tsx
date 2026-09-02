"use client";

import type { EndOfSpeechCandidateMessage } from "@/lib/types";

interface Props {
  candidate: EndOfSpeechCandidateMessage | null;
  micLevel: number;
  isListening: boolean;
}

// Full static class names (not built dynamically) so Tailwind's JIT
// compiler can actually detect and generate them at build time.
const SIGNAL_META = [
  { key: "semantic", label: "Semantic", gradient: "linear-gradient(180deg, #3FB6AC, #1D8F86)" },
  { key: "pause", label: "Pause", gradient: "linear-gradient(180deg, #E0A93F, #B5790E)" },
  { key: "speaker_change", label: "Speaker change", gradient: "linear-gradient(180deg, #9385D6, #6C5CC4)" },
  { key: "prosody", label: "Prosody", gradient: "linear-gradient(180deg, #C866A0, #B14C89)" },
] as const;

export function SignalInstrument({ candidate, micLevel, isListening }: Props) {
  const confidence = candidate?.confidence ?? 0;
  const isComplete = candidate?.is_end_of_speech ?? false;
  const ringGradientId = candidate
    ? isComplete
      ? "ringGradientComplete"
      : "ringGradientContinuing"
    : "ringGradientIdle";
  const glowClass = candidate
    ? isComplete
      ? "bg-decision-complete/15"
      : "bg-decision-continuing/15"
    : "bg-signal-semantic/10";

  // SVG ring math: circumference-based stroke-dashoffset for the
  // confidence sweep.
  const radius = 84;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference * (1 - confidence);

  return (
    <div className="flex flex-col items-center gap-6 sm:gap-8">
      <div className="relative h-40 w-40 sm:h-48 sm:w-48 md:h-56 md:w-56">
        {/* Ambient glow reflecting the current decision - color matches
            the ring, richer/larger now for visible presence */}
        <div
          className={`absolute inset-[-35%] -z-10 rounded-full blur-3xl transition-colors duration-500 ${glowClass}`}
        />

        {/* Idle listening pulse - only animates while actively listening
            with no candidate yet, so it reads as "waiting," not decoration */}
        {isListening && !candidate && (
          <div className="absolute inset-0 rounded-full border border-signal-semantic/30 animate-pulse-ring" />
        )}

        <svg viewBox="0 0 192 192" className="h-full w-full -rotate-90">
          <defs>
            <linearGradient id="ringGradientComplete" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#3FB98C" />
              <stop offset="100%" stopColor="#22785A" />
            </linearGradient>
            <linearGradient id="ringGradientContinuing" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#D6685A" />
              <stop offset="100%" stopColor="#A83A2E" />
            </linearGradient>
            <linearGradient id="ringGradientIdle" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#1D8F86" />
              <stop offset="100%" stopColor="#6C5CC4" />
            </linearGradient>
          </defs>
          <circle
            cx="96"
            cy="96"
            r={radius}
            className="fill-none stroke-paper-200"
            strokeWidth="6"
          />
          <circle
            cx="96"
            cy="96"
            r={radius}
            stroke={`url(#${ringGradientId})`}
            className="fill-none transition-[stroke-dashoffset] duration-300 ease-out"
            strokeWidth="7"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
          />
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center gap-1">
          <span className="font-mono text-2xl tabular-nums text-ink-900 sm:text-3xl md:text-4xl">
            {(confidence * 100).toFixed(0)}
            <span className="text-base text-ink-500 sm:text-lg md:text-xl">%</span>
          </span>
          <span
            className={`font-display text-xs font-medium sm:text-sm ${
              !candidate
                ? "text-ink-500"
                : isComplete
                ? "text-decision-complete"
                : "text-decision-continuing"
            }`}
          >
            {!candidate
              ? isListening
                ? "listening"
                : "idle"
              : isComplete
              ? "end of speech"
              : "continuing"}
          </span>
        </div>

        {/* Mic level indicator - small arc at the base, real signal not decoration */}
        <div
          className="absolute bottom-1 left-1/2 h-1 -translate-x-1/2 rounded-full bg-signal-semantic transition-all duration-100"
          style={{ width: `${Math.max(8, micLevel * 64)}px` }}
        />
      </div>

      {/* Signal breakdown - the actual inspectable-judgment payoff */}
      <div className="grid w-full max-w-sm grid-cols-4 gap-2 sm:gap-3">
        {SIGNAL_META.map(({ key, label, gradient }) => {
          const signal = candidate?.contributing_signals?.[key];
          const contribution = signal?.weighted_contribution ?? 0;
          // Prosody specifically can be genuinely unavailable (audio
          // segment too quiet/short for reliable pitch tracking) -
          // distinguish that from "contributed near zero on purpose".
          const isUnavailable =
            key === "prosody" &&
            signal &&
            "is_available" in signal &&
            (signal as { is_available?: boolean }).is_available === false;
          return (
            <div key={key} className="flex flex-col items-center gap-1.5">
              <div className="flex h-12 w-2.5 items-end overflow-hidden rounded-full bg-paper-200 shadow-inner sm:h-16">
                <div
                  className="w-full transition-all duration-300 ease-out"
                  style={{
                    height: `${Math.min(100, contribution * 150)}%`,
                    background: isUnavailable ? "#CBD2D6" : gradient,
                  }}
                />
              </div>
              <span className="text-center text-[10px] leading-tight text-ink-500 sm:text-[11px]">
                {label}
              </span>
              <span className="font-mono text-[10px] tabular-nums text-ink-700 sm:text-[11px]">
                {contribution.toFixed(2)}
                {isUnavailable && <span className="text-ink-500">*</span>}
              </span>
            </div>
          );
        })}
      </div>

      {candidate?.contributing_signals?.prosody &&
        "is_available" in candidate.contributing_signals.prosody &&
        (candidate.contributing_signals.prosody as { is_available?: boolean })
          .is_available === false && (
          <p className="text-center text-[10px] text-ink-500">
            * prosody unavailable for this segment - falls back to a neutral default
          </p>
        )}

      {candidate && (
        <div className="flex w-full max-w-xs flex-wrap items-center justify-center gap-x-4 gap-y-1 font-mono text-[11px] tabular-nums text-ink-500 sm:text-xs">
          <span>
            speaker {candidate.speaker ?? "?"}
            {candidate.speaker_changed ? " (changed)" : ""}
          </span>
          <span>{candidate.detection_latency_ms.toFixed(1)}ms</span>
        </div>
      )}
    </div>
  );
}
