// Mic capture + resampling to match the exact format the backend
// expects: 16kHz, mono, 16-bit signed PCM (little-endian) - the same
// format proven working by backend/live_mic_test.py's sounddevice
// capture. Browsers capture at their native rate (usually 44.1kHz or
// 48kHz) and float32 samples, so real conversion work is required
// here, not just a pass-through.

const TARGET_SAMPLE_RATE = 16000;
const CHUNK_DURATION_MS = 30; // matches the ~20-30ms chunking already
                                // used by the Python streaming clients

export interface AudioCaptureHandle {
  stop: () => void;
}

export interface AudioCaptureCallbacks {
  onChunk: (pcm16: ArrayBuffer) => void;
  onLevel?: (rmsLevel: number) => void; // 0.0-1.0, for the waveform/level UI
  onError?: (err: Error) => void;
}

/**
 * Downsamples a Float32Array buffer from `inputRate` to TARGET_SAMPLE_RATE
 * using simple linear interpolation. Good enough for speech (not
 * audiophile-grade resampling, but the backend/Deepgram pipeline
 * doesn't need that - intelligibility, not fidelity, is the bar).
 */
function downsampleTo16k(input: Float32Array, inputRate: number): Float32Array {
  if (inputRate === TARGET_SAMPLE_RATE) return input;
  const ratio = inputRate / TARGET_SAMPLE_RATE;
  const outputLength = Math.floor(input.length / ratio);
  const output = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i++) {
    const srcIndex = i * ratio;
    const srcIndexFloor = Math.floor(srcIndex);
    const srcIndexCeil = Math.min(srcIndexFloor + 1, input.length - 1);
    const frac = srcIndex - srcIndexFloor;
    output[i] = input[srcIndexFloor] * (1 - frac) + input[srcIndexCeil] * frac;
  }
  return output;
}

function float32ToInt16PCM(input: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(input.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < input.length; i++) {
    const clamped = Math.max(-1, Math.min(1, input[i]));
    const int16 = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    view.setInt16(i * 2, int16, true); // little-endian, matches backend expectation
  }
  return buffer;
}

function computeRmsLevel(input: Float32Array): number {
  let sumSquares = 0;
  for (let i = 0; i < input.length; i++) {
    sumSquares += input[i] * input[i];
  }
  const rms = Math.sqrt(sumSquares / input.length);
  // Speech RMS is typically well under 1.0 - scale for a usable UI meter
  return Math.min(1, rms * 4);
}

export async function startMicCapture(
  callbacks: AudioCaptureCallbacks
): Promise<AudioCaptureHandle> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
    },
  });

  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);

  // ScriptProcessorNode is deprecated but has universal support and is
  // simple/reliable for this use case; AudioWorklet is the modern
  // replacement but adds real setup complexity for marginal benefit
  // at speech sample rates. Documented tradeoff, not an oversight.
  const bufferSize = 4096;
  const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

  const samplesPerChunk = Math.floor(
    (TARGET_SAMPLE_RATE * CHUNK_DURATION_MS) / 1000
  );
  let pending: number[] = [];

  processor.onaudioprocess = (event) => {
    try {
      const inputData = event.inputBuffer.getChannelData(0);
      const downsampled = downsampleTo16k(inputData, audioContext.sampleRate);

      if (callbacks.onLevel) {
        callbacks.onLevel(computeRmsLevel(downsampled));
      }

      for (let i = 0; i < downsampled.length; i++) {
        pending.push(downsampled[i]);
      }

      while (pending.length >= samplesPerChunk) {
        const chunkSamples = pending.slice(0, samplesPerChunk);
        pending = pending.slice(samplesPerChunk);
        const pcm16 = float32ToInt16PCM(Float32Array.from(chunkSamples));
        callbacks.onChunk(pcm16);
      }
    } catch (err) {
      callbacks.onError?.(err as Error);
    }
  };

  source.connect(processor);
  processor.connect(audioContext.destination);

  return {
    stop: () => {
      processor.disconnect();
      source.disconnect();
      stream.getTracks().forEach((track) => track.stop());
      audioContext.close();
    },
  };
}
