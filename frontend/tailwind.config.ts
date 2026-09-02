import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Light theme: dark ink text on light paper - names now match
        // their actual role (previously inverted for the dark theme).
        paper: {
          50: "#F7F8F9",   // page background - cool off-white, not the AI-tell cream
          100: "#FFFFFF",  // card/panel surfaces
          200: "#E4E8EA",  // subtle borders, unfilled track backgrounds
          300: "#CBD2D6",  // stronger borders, dashed upload zone
        },
        ink: {
          500: "#6B747B",  // muted/secondary text, labels
          700: "#3D454C",  // supporting text
          900: "#14191E",  // primary text, headings
        },
        // Functional signal colors - darkened/saturated from the dark-
        // theme values to hold WCAG contrast against a light background
        // (dark-mode wants light pastel tints; light-mode wants the
        // opposite - deeper, more saturated hues).
        signal: {
          semantic: "#1D8F86",   // teal - meaning/language
          pause: "#B5790E",      // amber - time/silence
          speaker: "#6C5CC4",    // violet - identity/voice
          prosody: "#B14C89",    // magenta - pitch/intensity/duration
        },
        decision: {
          complete: "#2F9D5C",
          continuing: "#C14A3B",
        },
      },
      fontFamily: {
        display: ["var(--font-space-grotesk)", "sans-serif"],
        sans: ["var(--font-inter)", "sans-serif"],
        mono: ["var(--font-plex-mono)", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
