"use client";

interface Props {
  mode: "live" | "upload";
  onChange: (mode: "live" | "upload") => void;
}

export function ModeToggle({ mode, onChange }: Props) {
  const isLive = mode === "live";

  return (
    <div className="relative inline-flex rounded-full border border-paper-200 bg-paper-100 p-1 shadow-[0_1px_2px_rgba(20,25,30,0.06)]">
      {/* Sliding indicator - real motion, not just a color swap */}
      <div
        className="absolute inset-y-1 w-[calc(50%-4px)] rounded-full bg-ink-900 shadow-sm transition-transform duration-300 ease-out"
        style={{ transform: isLive ? "translateX(0)" : "translateX(calc(100% + 8px))" }}
      />
      {(["live", "upload"] as const).map((option) => (
        <button
          key={option}
          onClick={() => onChange(option)}
          className={`relative z-10 min-h-[40px] rounded-full px-4 py-2 text-xs font-medium transition-colors sm:px-5 sm:py-1.5 sm:text-sm ${
            mode === option
              ? "text-paper-100"
              : "text-ink-500 hover:text-ink-900"
          }`}
        >
          {option === "live" ? "Live microphone" : "Upload a file"}
        </button>
      ))}
    </div>
  );
}
