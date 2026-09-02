"use client";

import { useState } from "react";
import { useConvoraSocket } from "@/lib/useConvoraSocket";
import { SignalInstrument } from "@/components/SignalInstrument";
import { TranscriptStream } from "@/components/TranscriptStream";
import { FileUploadPanel } from "@/components/FileUploadPanel";
import { ModeToggle } from "@/components/ModeToggle";

const SIGNAL_LEGEND = [
  {
    name: "Semantic",
    barClass: "bg-signal-semantic",
    description:
      "A rule-based gate plus an LLM fallback reads the actual words - does this sound like a finished thought, or one still in flight.",
  },
  {
    name: "Pause",
    barClass: "bg-signal-pause",
    description:
      "How long has it been quiet. Longer silence leans toward complete, but on its own this is the naive approach Convora exists to beat.",
  },
  {
    name: "Speaker change",
    barClass: "bg-signal-speaker",
    description:
      "Real-time diarization - if a different voice starts talking, that's strong independent evidence a turn just ended.",
  },
];

export default function Home() {
  const [mode, setMode] = useState<"live" | "upload">("live");
  const { status, micLevel, transcript, latestCandidate, candidateHistory, errorMessage, start, stop } =
    useConvoraSocket();

  const isListening = status === "listening";
  const isConnecting = status === "connecting";

  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col px-4 py-10 sm:px-8 sm:py-14 lg:px-12">
      {/* Top bar */}
      <div className="mb-12 flex items-center justify-between sm:mb-16">
        <span className="font-display text-lg font-medium tracking-tight text-ink-900 sm:text-xl">
          Convora
        </span>
        <ModeToggle mode={mode} onChange={setMode} />
      </div>

      {mode === "live" ? (
        <>
          {/* Hero: two columns on desktop */}
          <section className="grid gap-10 lg:grid-cols-[1.1fr_1fr] lg:items-center lg:gap-16">
            <div className="space-y-7">
              <span className="inline-block font-mono text-[11px] font-medium uppercase tracking-[0.15em] text-signal-semantic">
                Real-time turn detection
              </span>
              <h1 className="text-gradient-headline font-display text-4xl font-medium leading-[1.05] tracking-tight sm:text-5xl lg:text-6xl">
                Your voice AI doesn&apos;t need to guess when you&apos;re
                done talking.
              </h1>
              <p className="max-w-md text-lg leading-relaxed text-ink-500">
                It should understand. Watch Convora listen in real time,
                weighing three independent signals before it ever
                interrupts you.
              </p>

              <div className="flex flex-wrap items-center gap-4 pt-3">
                <button
                  onClick={isListening || isConnecting ? stop : () => start()}
                  className="btn-gradient-primary group flex min-h-[52px] items-center gap-2.5 rounded-full px-7 py-3 text-sm font-medium text-white transition-transform hover:-translate-y-0.5 active:translate-y-0 active:opacity-90 disabled:translate-y-0 disabled:opacity-50 disabled:shadow-none"
                  disabled={isConnecting}
                >
                  <span
                    className={`h-2 w-2 rounded-full bg-paper-100 ${
                      isListening ? "animate-pulse" : ""
                    }`}
                  />
                  {isListening || isConnecting ? "Stop listening" : "Start listening"}
                </button>
                <span className="font-mono text-xs uppercase tracking-wide text-ink-500">
                  {status}
                </span>
              </div>

              {errorMessage && (
                <p className="rounded-md border border-decision-continuing/40 bg-decision-continuing/10 px-4 py-3 text-sm text-decision-continuing">
                  {errorMessage}
                </p>
              )}
            </div>

            <div className="flex justify-center lg:justify-end">
              <SignalInstrument
                candidate={latestCandidate}
                micLevel={micLevel}
                isListening={isListening}
              />
            </div>
          </section>

          {/* Transcript */}
          <section className="mt-14 sm:mt-20">
            <h2 className="mb-3 font-display text-sm font-medium uppercase tracking-wide text-ink-500">
              Live transcript
            </h2>
            <TranscriptStream lines={transcript} candidates={candidateHistory} />
          </section>

          {/* Signal legend - fills the page with real, substantive content */}
          <section className="mt-16 pt-10 sm:mt-24 sm:pt-14">
            <div className="divider-gradient mb-10 sm:mb-14" />
            <h2 className="mb-6 font-display text-sm font-medium uppercase tracking-wide text-ink-500">
              What the instrument is actually measuring
            </h2>
            <div className="grid gap-6 sm:grid-cols-3">
              {SIGNAL_LEGEND.map((s) => (
                <div
                  key={s.name}
                  className="card-gradient-border group space-y-2.5 rounded-xl p-5 shadow-[0_2px_10px_-4px_rgba(20,25,30,0.08)] transition-all hover:-translate-y-1 hover:shadow-[0_12px_28px_-8px_rgba(29,143,134,0.25)]"
                >
                  <div className="flex items-center gap-2.5">
                    <span
                      className={`h-2.5 w-2.5 rounded-full ${s.barClass} shadow-[0_0_0_4px_rgba(29,143,134,0.1)]`}
                    />
                    <span className="font-display text-sm font-medium text-ink-900">
                      {s.name}
                    </span>
                  </div>
                  <p className="text-sm leading-relaxed text-ink-500">
                    {s.description}
                  </p>
                </div>
              ))}
            </div>
          </section>
        </>
      ) : (
        <section className="mx-auto w-full max-w-2xl">
          <h1 className="mb-3 font-display text-2xl font-medium text-ink-900 sm:text-3xl">
            Test against a recording
          </h1>
          <p className="mb-8 max-w-md text-sm leading-relaxed text-ink-500">
            Runs the same batch pipeline used throughout evaluation - upload
            any real audio and see every detected end-of-speech event.
          </p>
          <FileUploadPanel />
        </section>
      )}

      <footer className="mt-16 pt-8 text-xs text-ink-500 sm:mt-24">
        <div className="divider-gradient mb-8" />
        Every decision here is a real, inspectable judgment - not a black
        box. The bars above are the actual weighted signals that produced
        it.
      </footer>
    </main>
  );
}
