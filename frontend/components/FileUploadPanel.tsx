"use client";

import { useCallback, useState } from "react";
import type { BatchResult } from "@/lib/types";

const DEFAULT_API_URL =
  process.env.NEXT_PUBLIC_CONVORA_API_URL || "http://localhost:8000";

export function FileUploadPanel() {
  const [status, setStatus] = useState<"idle" | "uploading" | "done" | "error">(
    "idle"
  );
  const [result, setResult] = useState<BatchResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleFile = useCallback(async (file: File) => {
    setStatus("uploading");
    setErrorMessage(null);
    setResult(null);
    setFileName(file.name);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${DEFAULT_API_URL}/api/process-file`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data: BatchResult = await response.json();
      setResult(data);
      setStatus("done");
    } catch (err) {
      setErrorMessage(
        err instanceof Error ? err.message : "Upload failed for an unknown reason."
      );
      setStatus("error");
    }
  }, []);

  return (
    <div className="space-y-6">
      <label className="group relative flex cursor-pointer flex-col items-center justify-center gap-2 overflow-hidden rounded-xl border-2 border-dashed border-paper-300 px-4 py-10 text-center shadow-[0_4px_20px_-6px_rgba(20,25,30,0.08)] transition-all hover:border-signal-semantic/50 hover:shadow-[0_8px_28px_-6px_rgba(29,143,134,0.25)] sm:px-6 sm:py-14" style={{ background: "linear-gradient(180deg, #ffffff 0%, #fafbfb 100%)" }}>
        <span className="font-display text-sm font-medium text-ink-900">
          Upload a recording
        </span>
        <span className="text-xs text-ink-500">
          WAV, run through the same batch pipeline used in evaluation
        </span>
        <input
          type="file"
          accept="audio/*"
          className="sr-only"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
      </label>

      {status === "uploading" && (
        <p className="text-sm text-ink-500">Processing {fileName}...</p>
      )}

      {status === "error" && errorMessage && (
        <p className="text-sm text-decision-continuing">{errorMessage}</p>
      )}

      {status === "done" && result && (
        <div className="space-y-4">
          <div className="flex flex-wrap gap-x-4 gap-y-1.5 font-mono text-[11px] tabular-nums text-ink-500 sm:gap-x-6 sm:text-xs">
            <span>{result.duration_s.toFixed(1)}s audio</span>
            <span>{result.metadata.total_pause_candidates} candidates</span>
            <span>{result.metadata.resolved_to_end_of_speech} end-of-speech events</span>
            <span>{result.metadata.processing_time_s.toFixed(1)}s to process</span>
          </div>

          {result.events.length === 0 && (
            <p className="rounded-md border border-signal-pause/40 bg-signal-pause/10 px-4 py-3 text-sm text-ink-700">
              No candidate crossed the end-of-speech threshold. This is a
              real result, not an error - see every candidate and its
              actual confidence score below to understand why.
            </p>
          )}

          {/* Confirmed end-of-speech events */}
          {result.events.length > 0 && (
            <div className="space-y-2">
              <h3 className="font-mono text-[11px] uppercase tracking-wide text-ink-500">
                End-of-speech events
              </h3>
              {result.events.map((event, i) => (
                <CandidateRow key={`event-${i}`} candidate={event} highlight />
              ))}
            </div>
          )}

          {/* All candidates considered - the diagnostic view, especially
              useful when zero events crossed threshold */}
          {result.all_candidates && result.all_candidates.length > 0 && (
            <div className="space-y-2">
              <h3 className="font-mono text-[11px] uppercase tracking-wide text-ink-500">
                All {result.all_candidates.length} candidates considered
              </h3>
              <div className="max-h-96 space-y-2 overflow-y-auto">
                {result.all_candidates.map((c, i) => (
                  <CandidateRow key={`cand-${i}`} candidate={c} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function CandidateRow({
  candidate,
  highlight = false,
}: {
  candidate: import("@/lib/types").BatchEvent;
  highlight?: boolean;
}) {
  return (
    <div
      className={`flex items-start justify-between gap-4 rounded-lg border p-3 transition-shadow hover:shadow-[0_2px_8px_-2px_rgba(20,25,30,0.08)] ${
        highlight
          ? "border-decision-complete/40 bg-decision-complete/5"
          : "border-paper-200 bg-paper-100"
      }`}
    >
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm text-ink-900">{candidate.fragment}</p>
        <p className="font-mono text-xs text-ink-500">
          {candidate.timestamp_s.toFixed(2)}s
          {candidate.speaker !== null ? ` - speaker ${candidate.speaker}` : " - speaker unknown"}
        </p>
      </div>
      <span
        className={`shrink-0 font-mono text-xs tabular-nums ${
          candidate.confidence >= 0.5 ? "text-decision-complete" : "text-ink-500"
        }`}
      >
        {(candidate.confidence * 100).toFixed(0)}%
      </span>
    </div>
  );
}
