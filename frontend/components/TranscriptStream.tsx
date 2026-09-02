"use client";

import { useEffect, useMemo, useRef } from "react";
import type { EndOfSpeechCandidateMessage } from "@/lib/types";

interface TranscriptLine {
  id: string;
  text: string;
  isFinal: boolean;
  timestamp_s: number;
}

interface Props {
  lines: TranscriptLine[];
  candidates: EndOfSpeechCandidateMessage[];
}

type TimelineItem =
  | { kind: "transcript"; timestamp_s: number; line: TranscriptLine }
  | { kind: "candidate"; timestamp_s: number; candidate: EndOfSpeechCandidateMessage };

function formatTimestamp(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(1);
  return `${mins}:${secs.padStart(4, "0")}`;
}

export function TranscriptStream({ lines, candidates }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Merge transcript text and candidate decisions into one real
  // chronological timeline, sorted by timestamp - this is the whole
  // point: confidence and speaker should sit right next to the words
  // that produced them, not live only in a separate widget.
  const timeline = useMemo<TimelineItem[]>(() => {
    const items: TimelineItem[] = [
      ...lines.map((line) => ({
        kind: "transcript" as const,
        timestamp_s: line.timestamp_s,
        line,
      })),
      ...candidates.map((candidate) => ({
        kind: "candidate" as const,
        timestamp_s: candidate.timestamp_s,
        candidate,
      })),
    ];
    return items.sort((a, b) => a.timestamp_s - b.timestamp_s);
  }, [lines, candidates]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [timeline]);

  return (
    <div
      ref={scrollRef}
      className="card-gradient-border h-72 overflow-y-auto rounded-xl p-3 shadow-[0_4px_20px_-6px_rgba(20,25,30,0.1)] sm:h-96 sm:p-4"
    >
      {timeline.length === 0 ? (
        <p className="text-sm text-ink-500">
          Nothing said yet. Start speaking and this fills in as it happens.
        </p>
      ) : (
        <div className="space-y-2">
          {timeline.map((item, i) =>
            item.kind === "transcript" ? (
              <div key={`t-${item.line.id}-${i}`} className="flex items-baseline gap-3">
                <span className="shrink-0 font-mono text-[11px] tabular-nums text-ink-500">
                  {formatTimestamp(item.timestamp_s)}
                </span>
                <p
                  className={
                    item.line.isFinal
                      ? "text-sm text-ink-900"
                      : "text-sm italic text-ink-500"
                  }
                >
                  {item.line.text}
                </p>
              </div>
            ) : (
              <div
                key={`c-${item.candidate.timestamp_s}-${i}`}
                className={`ml-6 flex items-center gap-3 rounded border-l-2 py-1 pl-3 text-xs ${
                  item.candidate.is_end_of_speech
                    ? "border-decision-complete bg-decision-complete/5"
                    : "border-paper-300 bg-paper-200/40"
                }`}
              >
                <span className="font-mono tabular-nums text-ink-500">
                  {formatTimestamp(item.timestamp_s)}
                </span>
                <span
                  className={`font-medium ${
                    item.candidate.is_end_of_speech
                      ? "text-decision-complete"
                      : "text-ink-500"
                  }`}
                >
                  {item.candidate.is_end_of_speech ? "end of speech" : "continuing"}
                </span>
                <span className="font-mono tabular-nums text-ink-700">
                  {(item.candidate.confidence * 100).toFixed(0)}%
                </span>
                <span className="font-mono text-ink-500">
                  {item.candidate.speaker !== null
                    ? `speaker ${item.candidate.speaker}${item.candidate.speaker_changed ? " (changed)" : ""}`
                    : "speaker unknown"}
                </span>
                <span className="font-mono text-ink-500">
                  {item.candidate.detection_latency_ms.toFixed(0)}ms
                </span>
              </div>
            )
          )}
        </div>
      )}
    </div>
  );
}
