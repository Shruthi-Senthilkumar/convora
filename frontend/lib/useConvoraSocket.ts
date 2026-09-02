"use client";

import { useCallback, useRef, useState } from "react";
import { startMicCapture, type AudioCaptureHandle } from "./audio";
import type {
  ConnectionStatus,
  EndOfSpeechCandidateMessage,
  ServerMessage,
} from "./types";

const DEFAULT_WS_URL =
  process.env.NEXT_PUBLIC_CONVORA_WS_URL || "ws://localhost:8000/ws/transcribe";

interface TranscriptLine {
  id: string;
  text: string;
  isFinal: boolean;
  timestamp_s: number;
}

export function useConvoraSocket() {
  const [status, setStatus] = useState<ConnectionStatus>("idle");
  const [micLevel, setMicLevel] = useState(0);
  const [transcript, setTranscript] = useState<TranscriptLine[]>([]);
  const [latestCandidate, setLatestCandidate] =
    useState<EndOfSpeechCandidateMessage | null>(null);
  const [candidateHistory, setCandidateHistory] = useState<
    EndOfSpeechCandidateMessage[]
  >([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const captureRef = useRef<AudioCaptureHandle | null>(null);

  const stop = useCallback(() => {
    captureRef.current?.stop();
    captureRef.current = null;
    socketRef.current?.close();
    socketRef.current = null;
    setStatus("closed");
    setMicLevel(0);
  }, []);

  const start = useCallback(async (wsUrl: string = DEFAULT_WS_URL) => {
    setErrorMessage(null);
    setStatus("connecting");
    setTranscript([]);
    setLatestCandidate(null);
    setCandidateHistory([]);

    try {
      const socket = new WebSocket(wsUrl);
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;

      socket.onopen = async () => {
        try {
          const capture = await startMicCapture({
            onChunk: (pcm16) => {
              if (socket.readyState === WebSocket.OPEN) {
                socket.send(pcm16);
              }
            },
            onLevel: setMicLevel,
            onError: (err) => setErrorMessage(err.message),
          });
          captureRef.current = capture;
          setStatus("listening");
        } catch (err) {
          setErrorMessage(
            err instanceof Error
              ? `Microphone access failed: ${err.message}`
              : "Microphone access failed."
          );
          setStatus("error");
          socket.close();
        }
      };

      socket.onmessage = (event) => {
        try {
          const msg: ServerMessage = JSON.parse(event.data);
          if (msg.type === "partial_transcript") {
            setTranscript((prev) => {
              const withoutLastPartial = prev.filter((l) => l.isFinal);
              return [
                ...withoutLastPartial,
                {
                  id: `partial-${msg.timestamp_s}`,
                  text: msg.text,
                  isFinal: false,
                  timestamp_s: msg.timestamp_s,
                },
              ];
            });
          } else if (msg.type === "final_transcript") {
            setTranscript((prev) => [
              ...prev.filter((l) => l.isFinal),
              {
                id: `final-${msg.timestamp_s}`,
                text: msg.text,
                isFinal: true,
                timestamp_s: msg.timestamp_s,
              },
            ]);
          } else if (msg.type === "end_of_speech_candidate") {
            setLatestCandidate(msg);
            setCandidateHistory((prev) => [...prev, msg]);
          } else if (msg.type === "error") {
            setErrorMessage(msg.message);
          }
        } catch {
          // Non-JSON or malformed message - ignore rather than crash
          // the whole session over one bad frame.
        }
      };

      socket.onerror = () => {
        setErrorMessage("WebSocket connection error.");
        setStatus("error");
      };

      socket.onclose = () => {
        captureRef.current?.stop();
        captureRef.current = null;
        setStatus((prev) => (prev === "error" ? prev : "closed"));
      };
    } catch (err) {
      setErrorMessage(
        err instanceof Error ? err.message : "Could not open connection."
      );
      setStatus("error");
    }
  }, []);

  return {
    status,
    micLevel,
    transcript,
    latestCandidate,
    candidateHistory,
    errorMessage,
    start,
    stop,
  };
}
