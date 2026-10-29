import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import {
  AudioLines,
  BrainCircuit,
  CheckCircle2,
  FileAudio,
  FileUp,
  Fingerprint,
  Headphones,
  ShieldCheck,
  X,
} from "lucide-react";
import "./App.css";
import { predictAudio, uploadAudio } from "./services/audioApi";
import type { AudioUploadResponse } from "./services/audioApi";

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const SUPPORTED_EXTENSIONS = /\.(wav|mp3|flac|ogg)$/i;

type AnalysisStatus = "idle" | "ready" | "unavailable";

function App() {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [error, setError] = useState("");
  const [predictionError, setPredictionError] = useState("");
  const [waveformError, setWaveformError] = useState("");
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [analysisStatus, setAnalysisStatus] =
    useState<AnalysisStatus>("idle");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] =
    useState<AudioUploadResponse | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const waveformRef = useRef<HTMLDivElement>(null);
  const waveSurferRef = useRef<WaveSurfer | null>(null);

  useEffect(() => {
    if (!audioFile || !waveformRef.current) return;

    let cancelled = false;
    const objectUrl = URL.createObjectURL(audioFile);

    setWaveformError("");
    setIsPlaying(false);
    setDuration(0);
    setCurrentTime(0);

    const waveSurfer = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: "#a5b4fc",
      progressColor: "#6366f1",
      cursorColor: "#4338ca",
      height: 80,
      barWidth: 3,
      barGap: 2,
      barRadius: 3,
      normalize: true,
    });

    waveSurferRef.current = waveSurfer;

    waveSurfer.on("ready", () => {
      if (cancelled) return;
      setDuration(waveSurfer.getDuration());
      setWaveformError("");
    });

    waveSurfer.on("timeupdate", (time) => {
      if (!cancelled) setCurrentTime(time);
    });

    waveSurfer.on("play", () => {
      if (!cancelled) setIsPlaying(true);
    });

    waveSurfer.on("pause", () => {
      if (!cancelled) setIsPlaying(false);
    });

    waveSurfer.on("finish", () => {
      if (!cancelled) setIsPlaying(false);
    });

    waveSurfer.on("error", () => {
      if (!cancelled) {
        setWaveformError(
          "This audio could not be played in your browser. You can still try uploading it to the backend.",
        );
        setIsPlaying(false);
      }
    });

    waveSurfer.load(objectUrl).catch(() => {
      if (!cancelled) {
        setWaveformError(
          "Waveform preview is unavailable for this file. You can still try uploading it.",
        );
      }
    });

    return () => {
      cancelled = true;
      waveSurfer.destroy();

      if (waveSurferRef.current === waveSurfer) {
        waveSurferRef.current = null;
      }

      URL.revokeObjectURL(objectUrl);
    };
  }, [audioFile]);

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;

    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const formatTime = (seconds: number) => {
    if (!Number.isFinite(seconds)) return "0:00";

    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);

    return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
  };

  const handleFile = (file?: File) => {
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    if (!file) return;

    if (!SUPPORTED_EXTENSIONS.test(file.name)) {
      setError(
        "Unsupported file format. Please choose WAV, MP3, FLAC, or OGG.",
      );
      return;
    }

    if (file.size === 0) {
      setError("This file is empty. Please choose a valid audio file.");
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      setError(
        "This file exceeds the 20 MB limit. Please choose a smaller file.",
      );
      return;
    }

    setError("");
    setPredictionError("");
    setWaveformError("");
    setUploadResult(null);
    setAnalysisStatus("ready");
    setAudioFile(file);
  };

  const togglePlayback = async () => {
    const waveSurfer = waveSurferRef.current;

    if (!waveSurfer || duration <= 0 || waveformError) return;

    try {
      await waveSurfer.playPause();
    } catch {
      setWaveformError("Unable to play this audio file in your browser.");
    }
  };

  const removeAudio = () => {
    if (isUploading) return;

    waveSurferRef.current?.stop();
    setAudioFile(null);
    setError("");
    setPredictionError("");
    setWaveformError("");
    setIsPlaying(false);
    setDuration(0);
    setCurrentTime(0);
    setAnalysisStatus("idle");
    setUploadResult(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleUpload = async () => {
    if (!audioFile || isUploading) return;

    setError("");
    setPredictionError("");
    setUploadResult(null);
    setIsUploading(true);

    try {
      const result = await uploadAudio(audioFile);
      setUploadResult(result);

      try {
        await predictAudio(audioFile);
        setAnalysisStatus("unavailable");
      } catch (err) {
        setAnalysisStatus("unavailable");
        setPredictionError(
          err instanceof Error
            ? err.message
            : "Prediction is currently unavailable. Please try again later.",
        );
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Audio upload failed. Please try again.",
      );
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="app-shell">
      <nav className="navbar">
        <a className="brand" href="#home" aria-label="AcousticSpace home">
          <span className="brand-icon">
            <AudioLines size={23} />
          </span>
          <span>AcousticSpace</span>
        </a>

        <div className="nav-links">
          <a href="#home">Home</a>
          <a href="#analyzer">Audio Analyzer</a>
          <a href="#about">About RIR</a>
        </div>

        <span className="research-badge">Research Preview</span>
      </nav>

      <main>
        <section className="hero-section" id="home">
          <div className="hero-copy">
            <span className="eyebrow">
              <ShieldCheck size={15} />
              AUDIO AUTHENTICITY RESEARCH
            </span>

            <h1>
              Hear beyond
              <br />
              the <span>surface.</span>
            </h1>

            <p>
              Explore an approach to deepfake audio detection using Room Impulse
              Response (RIR) characteristics and acoustic environments.
            </p>

            <a className="primary-button hero-button" href="#analyzer">
              <Headphones size={18} />
              Analyze audio
            </a>

            <div className="hero-features">
              <span>
                <ShieldCheck size={16} />
                Privacy-conscious workflow
              </span>
              <span>
                <Fingerprint size={16} />
                Acoustic analysis research
              </span>
            </div>
          </div>

          <div className="hero-visual" aria-hidden="true">
            <div className="hero-orb">
              <div className="orb-ring orb-ring-one" />
              <div className="orb-ring orb-ring-two" />
              <div className="orb-ring orb-ring-three" />
              <AudioLines size={76} strokeWidth={1.2} />
            </div>
            <div className="floating-label floating-label-top">
              <AudioLines size={16} />
              Acoustic signals
            </div>
            <div className="floating-label floating-label-bottom">
              <BrainCircuit size={17} />
              RIR-based research
            </div>
          </div>
        </section>

        <section className="analyzer-section" id="analyzer">
          <div className="section-heading">
            <div>
              <span className="eyebrow">YOUR WORKSPACE</span>
              <h2>Audio Analyzer</h2>
              <p>
                Upload an audio sample to preview it and send it to the
                AcousticSpace backend.
              </p>
            </div>

            <span className="file-limit">
              <FileAudio size={15} />
              Max. 20 MB
            </span>
          </div>

          <div className="analyzer-grid">
            <section className="upload-panel">
              <div className="panel-heading">
                <span className="step-number">01</span>
                <div>
                  <h3>Upload your audio</h3>
                  <p>Select a supported audio file to get started.</p>
                </div>
              </div>

              <input
                ref={fileInputRef}
                className="visually-hidden"
                type="file"
                accept=".wav,.mp3,.flac,.ogg,audio/wav,audio/mpeg,audio/flac,audio/ogg"
                onChange={(event) => handleFile(event.target.files?.[0])}
              />

              {!audioFile ? (
                <button
                  className="drop-zone"
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={(event) => {
                    event.preventDefault();
                    handleFile(event.dataTransfer.files[0]);
                  }}
                >
                  <span className="upload-icon">
                    <FileUp size={27} />
                  </span>
                  <strong>Drop your audio file here</strong>
                  <span>or click to browse your files</span>
                  <span className="supported-formats">
                    WAV · MP3 · FLAC · OGG
                  </span>
                </button>
              ) : (
                <div className="selected-file">
                  <span className="file-icon">
                    <FileAudio size={23} />
                  </span>

                  <div className="file-details">
                    <strong title={audioFile.name}>{audioFile.name}</strong>
                    <span>{formatSize(audioFile.size)}</span>
                  </div>

                  <button
                    type="button"
                    className="remove-file"
                    onClick={removeAudio}
                    disabled={isUploading}
                    aria-label="Remove selected audio"
                    title="Remove audio"
                  >
                    <X size={18} />
                  </button>
                </div>
              )}

              {error && (
                <p className="error-message" role="alert">
                  {error}
                </p>
              )}

              {audioFile && (
                <div className="waveform-panel">
                  <div className="waveform-heading">
                    <span>
                      <AudioLines size={16} />
                      Audio waveform
                    </span>
                    <span>{formatTime(duration)}</span>
                  </div>

                  <div className="waveform-canvas" ref={waveformRef} />

                  {waveformError && (
                    <p className="error-message" role="status">
                      {waveformError}
                    </p>
                  )}

                  <div className="waveform-controls">
                    <button
                      className="waveform-play-button"
                      type="button"
                      onClick={togglePlayback}
                      disabled={duration <= 0 || Boolean(waveformError)}
                    >
                      {isPlaying ? "Pause audio" : "Play audio"}
                    </button>

                    <span className="audio-time">
                      {formatTime(currentTime)} / {formatTime(duration)}
                    </span>
                  </div>
                </div>
              )}

              <div className="privacy-note">
                <ShieldCheck size={17} />
                <p>
                  Your file is sent to the configured backend only when you
                  click Upload audio. Avoid uploading sensitive recordings.
                </p>
              </div>

              <button
                className="primary-button upload-action"
                type="button"
                onClick={handleUpload}
                disabled={!audioFile || isUploading}
              >
                {isUploading ? (
                  <>
                    <AudioLines size={18} />
                    Uploading and checking prediction...
                  </>
                ) : (
                  <>
                    <FileUp size={18} />
                    Upload audio
                  </>
                )}
              </button>
            </section>

            <section className="results-panel">
              <div className="panel-heading">
                <span className="step-number">02</span>
                <div>
                  <h3>Analysis results</h3>
                  <p>Backend upload status and model readiness.</p>
                </div>
              </div>

              {uploadResult ? (
                <div className="empty-results upload-complete">
                  <span className="result-icon">
                    <CheckCircle2 size={30} />
                  </span>

                  <h3>Audio uploaded successfully</h3>
                  <p>
                    The backend confirmed that your audio file was received.
                  </p>

                  <div className="upload-result-details">
                    <div>
                      <span>Server status</span>
                      <strong>{uploadResult.status}</strong>
                    </div>
                    <div>
                      <span>Original filename</span>
                      <strong>{uploadResult.filename}</strong>
                    </div>
                    <div>
                      <span>Saved filename</span>
                      <strong>{uploadResult.saved_filename}</strong>
                    </div>
                    <div>
                      <span>Received size</span>
                      <strong>{formatSize(uploadResult.size_bytes)}</strong>
                    </div>
                  </div>

                  <div className="confidence-panel">
                    <div className="confidence-heading">
                      <span>Prediction confidence</span>
                      <span className="confidence-status">Unavailable</span>
                    </div>

                    <div
                      className="confidence-track"
                      aria-label="Prediction confidence unavailable"
                      aria-valuetext="No confidence score returned"
                    >
                      <div className="confidence-fill" />
                    </div>

                    <p className="confidence-message">
                      No confidence score is available because the backend
                      model inference is not configured.
                    </p>
                  </div>

                  <div className="research-note">
                    <BrainCircuit size={19} />
                    <div>
                      <strong>
                        {predictionError
                          ? "Prediction unavailable"
                          : "Prediction status"}
                      </strong>
                      <p>
                        {predictionError ||
                          "The backend did not return a prediction result."}
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="empty-results">
                  <span className="result-icon">
                    {isUploading ? (
                      <AudioLines size={30} />
                    ) : (
                      <BrainCircuit size={30} />
                    )}
                  </span>

                  <h3>
                    {isUploading
                      ? "Uploading your audio"
                      : analysisStatus === "unavailable"
                        ? "Prediction unavailable"
                        : audioFile
                          ? "Ready to upload"
                          : "Waiting for audio"}
                  </h3>

                  <p>
                    {isUploading
                      ? "Please wait while the backend receives your file and checks prediction availability."
                      : audioFile
                        ? "Your audio is selected. Upload it to check backend and prediction availability."
                        : "Choose an audio file to prepare your workspace."}
                  </p>

                  {!audioFile && (
                    <span className="waiting-indicator">
                      <span className="status-dot" />
                      Waiting for input
                    </span>
                  )}
                </div>
              )}
            </section>
          </div>
        </section>
      </main>

      <footer className="footer">
        <a className="brand footer-brand" href="#home">
          <span className="brand-icon">
            <AudioLines size={19} />
          </span>
          <span>AcousticSpace</span>
        </a>
        <p>Deepfake audio detection research · Research Preview</p>
      </footer>
    </div>
  );
}

export default App;