// Matches the exact message contract already in production use by
// backend/main.py and its Python reference clients
// (backend/test_client.py, backend/live_mic_test.py).
// Do not diverge from this shape without updating the backend too.

export interface PartialTranscriptMessage {
  type: "partial_transcript";
  text: string;
  timestamp_s: number;
}

export interface FinalTranscriptMessage {
  type: "final_transcript";
  text: string;
  timestamp_s: number;
}

export interface ContributingSignal {
  raw_confidence?: number;
  mapped_value?: number;
  weighted_contribution: number;
  label?: string;
  source?: string;
  duration_s?: number;
  changed?: boolean;
}

export interface ContributingSignals {
  semantic: ContributingSignal;
  pause: ContributingSignal;
  speaker_change: ContributingSignal;
  prosody?: ContributingSignal;
}

export interface EndOfSpeechCandidateMessage {
  type: "end_of_speech_candidate";
  timestamp_s: number;
  is_end_of_speech: boolean;
  confidence: number;
  speaker: number | null;
  speaker_changed: boolean;
  fragment: string;
  contributing_signals: ContributingSignals;
  detection_latency_ms: number;
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export type ServerMessage =
  | PartialTranscriptMessage
  | FinalTranscriptMessage
  | EndOfSpeechCandidateMessage
  | ErrorMessage;

// --- Batch mode (detection/end_of_speech_detector.py output schema) ---

export interface BatchEvent {
  event_type: "end_of_speech";
  timestamp_s: number;
  confidence: number;
  speaker: number | null;
  fragment: string;
  contributing_signals: ContributingSignals;
}

export interface BatchMetadata {
  total_pause_candidates: number;
  resolved_to_end_of_speech: number;
  processing_time_s: number;
  semantic_source_breakdown: Record<string, number>;
}

export interface BatchResult {
  audio_file: string;
  duration_s: number;
  events: BatchEvent[];
  all_candidates?: BatchEvent[]; // every candidate considered, not just
                                   // ones that crossed the EOS threshold -
                                   // optional since older backend
                                   // responses may not include it yet
  metadata: BatchMetadata;
}

export type ConnectionStatus = "idle" | "connecting" | "listening" | "error" | "closed";
