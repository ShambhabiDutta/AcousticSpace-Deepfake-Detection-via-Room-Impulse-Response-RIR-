
import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import {
  AudioLines,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  FileAudio,
  FileUp,
  Fingerprint,
  Headphones,
  History,
  ShieldCheck,
  X,
} from "lucide-react";
import "./App.css";
import { predictAudio, uploadAudio } from "./services/audioApi";
import type { AudioUploadResponse } from "./services/audioApi";
import type { AnalysisRecord } from "./types/analysis";

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const SUPPORTED_EXTENSIONS = /\.(wav|mp3|flac|ogg)$/i;

type AnalysisStatus = "idle" | "ready" | "success" | "unavailable";

interface PredictionResponse {
  prediction?: string;
  confidence?: number;
}

function App() {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [error, setError] = useState("");
  const [predictionError, setPredictionError] = useState("");
  const [waveformError, setWaveformError] = useState("");
  const [isPlaying, setIsPlaying] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  const [analysisStatus, setAnalysisStatus] =
    useState<AnalysisStatus>("idle");
  const [isUploading, setIsUploading] = useState(false);

  const [uploadResult, setUploadResult] =
    useState<AudioUploadResponse | null>(null);

  const [predictionResult, setPredictionResult] =
    useState<PredictionResponse | null>(null);

  // Analysis history state
  const [analysisHistory, setAnalysisHistory] =
    useState<AnalysisRecord[]>([]);

  const [selectedAnalysisId, setSelectedAnalysisId] =
    useState<string | null>(null);

  const selectedAnalysis =
    analysisHistory.find((record) => record.id === selectedAnalysisId) ??
    null;

  const fileInputRef = useRef<HTMLInputElement>(null);
  const waveformRef = useRef<HTMLDivElement>(null);
  const waveSurferRef = useRef<WaveSurfer | null>(null);

  // Initialize and manage the audio waveform.
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
      waveColor: "#8b7cf6",
      progressColor: "#c4b5fd",
      cursorColor: "#ffffff",
      height: 90,
      barWidth: 3,
      barGap: 2,
      barRadius: 3,
      normalize: true,
      interact: true,
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
          "Unable to generate the audio waveform. Please try another audio file.",
        );
      }
    });

    waveSurfer.load(objectUrl).catch(() => {
      if (!cancelled) {
        setWaveformError("Unable to load this audio file for playback.");
      }
    });

    return () => {
      cancelled = true;
      waveSurfer.destroy();
      waveSurferRef.current = null;
      URL.revokeObjectURL(objectUrl);
    };
  }, [audioFile]);

  // Format seconds into minutes and seconds.
  const formatTime = (time: number): string => {
    if (!Number.isFinite(time)) return "0:00";

    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);

    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

  // Format file size in MB.
  const formatSize = (bytes: number): string => {
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  // Format analysis date.
  const formatDate = (date: string): string => {
    const parsedDate = new Date(date);

    if (Number.isNaN(parsedDate.getTime())) {
      return "Date unavailable";
    }

    return parsedDate.toLocaleString();
  };

  // Validate uploaded audio files.
  const validateFile = (file: File): string => {
    if (!SUPPORTED_EXTENSIONS.test(file.name)) {
      return "Unsupported file format. Please upload WAV, MP3, FLAC, or OGG audio.";
    }

    if (file.size === 0) {
      return "The selected audio file is empty.";
    }

    if (file.size > MAX_FILE_SIZE) {
      return "File size exceeds the 20 MB limit.";
    }

    return "";
  };

  // Handle file selection and validation.
  const handleFile = (file?: File) => {
    if (!file) return;

    const validationError = validateFile(file);

    if (validationError) {
      setError(validationError);
      setPredictionError("");
      setWaveformError("");
      setUploadResult(null);
      setPredictionResult(null);
      setAnalysisStatus("idle");
      setAudioFile(null);

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }

      return;
    }

    setError("");
    setPredictionError("");
    setWaveformError("");
    setUploadResult(null);
    setPredictionResult(null);
    setAnalysisStatus("ready");
    setAudioFile(file);
  };

  // Play or pause the selected audio.
  const togglePlayback = async () => {
    const waveSurfer = waveSurferRef.current;

    if (!waveSurfer || duration === 0) {
      setWaveformError("Please wait until the waveform has loaded.");
      return;
    }

    try {
      await waveSurfer.playPause();
    } catch {
      setWaveformError("Unable to play this audio file.");
    }
  };

  // Clear the currently selected audio.
  const clearAudio = () => {
    waveSurferRef.current?.stop();

    setAudioFile(null);
    setError("");
    setPredictionError("");
    setWaveformError("");
    setIsPlaying(false);
    setIsDragging(false);
    setDuration(0);
    setCurrentTime(0);
    setAnalysisStatus("idle");
    setUploadResult(null);
    setPredictionResult(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Upload audio and save each successful upload in history.
  const handleUpload = async () => {
    if (!audioFile || isUploading) return;

    setError("");
    setPredictionError("");
    setUploadResult(null);
    setPredictionResult(null);
    setAnalysisStatus("ready");
    setIsUploading(true);

    try {
      // First, upload the audio file.
      const result = await uploadAudio(audioFile);
      setUploadResult(result);

      // Create a unique history record.
      const analysisId =
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : `${Date.now()}-${Math.random().toString(36).slice(2)}`;

      const newRecord: AnalysisRecord = {
        id: analysisId,
        filename: result.filename,
        createdAt: new Date().toISOString(),
        status: "uploaded",
        uploadResult: result,
      };

      // Add the new record without removing previous records.
      setAnalysisHistory((previousHistory) => [
        newRecord,
        ...previousHistory,
      ]);

      setSelectedAnalysisId(analysisId);

      // Request prediction after successful upload.
      try {
        const prediction = await predictAudio(audioFile);

        if (
          typeof prediction === "object" &&
          prediction !== null &&
          "prediction" in prediction &&
          typeof prediction.prediction === "string"
        ) {
          const response = prediction as PredictionResponse;

          const validConfidence =
            typeof response.confidence === "number" &&
            Number.isFinite(response.confidence) &&
            response.confidence >= 0 &&
            response.confidence <= 1;

          const savedResponse: PredictionResponse = {
            prediction: response.prediction,
            ...(validConfidence
              ? { confidence: response.confidence }
              : {}),
          };

          setPredictionResult(savedResponse);
          setAnalysisStatus("success");
          setPredictionError("");

          // Update the matching history record.
          setAnalysisHistory((previousHistory) =>
            previousHistory.map((record) =>
              record.id === analysisId
                ? {
                    ...record,
                    prediction: savedResponse.prediction,
                    confidence: savedResponse.confidence,
                    status: "completed",
                    predictionError: undefined,
                  }
                : record,
            ),
          );
        } else {
          const message =
            "The prediction service returned an unsupported response format.";

          setAnalysisStatus("unavailable");
          setPredictionError(message);

          setAnalysisHistory((previousHistory) =>
            previousHistory.map((record) =>
              record.id === analysisId
                ? {
                    ...record,
                    status: "unavailable",
                    predictionError: message,
                  }
                : record,
            ),
          );
        }
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : "Prediction is currently unavailable. Please try again later.";

        setAnalysisStatus("unavailable");
        setPredictionError(message);

        // Keep the upload in history even if prediction fails.
        setAnalysisHistory((previousHistory) =>
          previousHistory.map((record) =>
            record.id === analysisId
              ? {
                  ...record,
                  status: "unavailable",
                  predictionError: message,
                }
              : record,
          ),
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

  // Validate and calculate prediction confidence.
  const confidence =
    typeof predictionResult?.confidence === "number" &&
    Number.isFinite(predictionResult.confidence) &&
    predictionResult.confidence >= 0 &&
    predictionResult.confidence <= 1
      ? predictionResult.confidence * 100
      : null;

  // Convert history status into readable labels.
  const getStatusLabel = (status: AnalysisRecord["status"]) => {
    switch (status) {
      case "completed":
        return "Completed";

      case "uploaded":
        return "Uploaded";

      case "unavailable":
        return "Prediction unavailable";

      default:
        return "Unknown";
    }
  };

  return (
    <main className="app-shell">
      <nav className="navbar">
        <div className="brand">
          <span className="brand-icon">
            <AudioLines size={22} />
          </span>
          <span>AcousticSpace</span>
        </div>

        <div className="nav-links">
          <a href="#home" className="active">
            Home
          </a>
          <a href="#analyzer">Audio Analyzer</a>
          <a href="#history">History</a>
          <a href="#about">About RIR</a>
        </div>

        <span className="research-badge">Research Preview</span>
      </nav>

      <section className="hero-section" id="home">
        <div className="hero-content">
          <span className="eyebrow">
            <Fingerprint size={15} />
            AUDIO AUTHENTICITY RESEARCH
          </span>

          <h1>
            Hear beyond
            <br />
            <span>the surface.</span>
          </h1>

          <p>
            Explore audio authenticity through acoustic characteristics
            and room impulse response analysis.
          </p>

          <a href="#analyzer" className="hero-button">
            <Headphones size={18} />
            Start analyzing
          </a>
        </div>

        <div className="hero-visual">
          <div className="visual-glow" />
          <div className="sound-card">
            <AudioLines size={100} strokeWidth={1} />
            <span>ACOUSTIC ANALYSIS</span>
          </div>
        </div>
      </section>

      <section className="analyzer-section" id="analyzer">
        <div className="section-heading">
          <span className="eyebrow">WORKSPACE</span>
          <h2>Audio Analyzer</h2>
          <p>
            Upload an audio recording to begin the analysis workflow.
          </p>
        </div>

        <div className="analyzer-grid">
          <section className="upload-panel">
            <div className="panel-heading">
              <span className="step-number">01</span>
              <div>
                <h3>Upload audio</h3>
                <p>Select a recording for analysis.</p>
              </div>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept=".wav,.mp3,.flac,.ogg,audio/*"
              hidden
              onChange={(event) => handleFile(event.target.files?.[0])}
            />

            {!audioFile ? (
              <button
                type="button"
                className={`drop-zone ${isDragging ? "dragging" : ""}`}
                onClick={() => fileInputRef.current?.click()}
                onDragEnter={(event) => {
                  event.preventDefault();
                  setIsDragging(true);
                }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={(event) => {
                  if (
                    !event.currentTarget.contains(
                      event.relatedTarget as Node | null,
                    )
                  ) {
                    setIsDragging(false);
                  }
                }}
                onDrop={(event) => {
                  event.preventDefault();
                  setIsDragging(false);
                  handleFile(event.dataTransfer.files?.[0]);
                }}
              >
                <span className="upload-icon">
                  <FileUp size={30} />
                </span>
                <strong>Drop your audio file here</strong>
                <span>or click to browse files</span>
                <small>WAV, MP3, FLAC, OGG · Maximum 20 MB</small>
              </button>
            ) : (
              <div className="selected-file">
                <div className="file-details">
                  <span className="file-icon">
                    <FileAudio size={23} />
                  </span>

                  <div>
                    <strong>{audioFile.name}</strong>
                    <span>{formatSize(audioFile.size)}</span>
                  </div>

                  <button
                    type="button"
                    className="icon-button"
                    onClick={clearAudio}
                    aria-label="Remove audio file"
                  >
                    <X size={18} />
                  </button>
                </div>

                <div className="waveform-container">
                  <div ref={waveformRef} className="waveform-canvas" />

                  {waveformError && (
                    <p className="error-message" role="alert">
                      {waveformError}
                    </p>
                  )}

                  <div className="waveform-controls">
                    <button
                      type="button"
                      className="waveform-play-button"
                      onClick={togglePlayback}
                      disabled={duration === 0}
                      aria-label={isPlaying ? "Pause audio" : "Play audio"}
                    >
                      {isPlaying ? "Pause" : "Play"}
                    </button>

                    <span className="waveform-time">
                      {formatTime(currentTime)} / {formatTime(duration)}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className="error-message" role="alert">
                {error}
              </div>
            )}

            <div className="privacy-note">
              <ShieldCheck size={18} />
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

          <section
            className="results-panel"
            aria-live="polite"
            aria-busy={isUploading}
          >
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

                {analysisStatus === "success" && predictionResult ? (
                  <div className="confidence-panel">
                    <div className="confidence-heading">
                      <span>Prediction result</span>
                      <span className="confidence-status">Available</span>
                    </div>

                    <h3>{predictionResult.prediction}</h3>

                    <div className="confidence-heading">
                      <span>Prediction confidence</span>
                      <span className="confidence-status">
                        {confidence !== null
                          ? `${confidence.toFixed(1)}%`
                          : "Not provided"}
                      </span>
                    </div>

                    <div
                      className="confidence-track"
                      role="meter"
                      aria-label="Prediction confidence"
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={confidence ?? 0}
                      aria-valuetext={
                        confidence !== null
                          ? `${confidence.toFixed(1)} percent`
                          : "No confidence score returned"
                      }
                    >
                      <div
                        className="confidence-fill"
                        style={{ width: `${confidence ?? 0}%` }}
                      />
                    </div>

                    {confidence === null && (
                      <p className="confidence-message">
                        The backend returned a prediction without a
                        valid confidence score.
                      </p>
                    )}
                  </div>
                ) : (
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
                      No confidence score is available because backend
                      model inference is not configured.
                    </p>
                  </div>
                )}

                <div className="research-note">
                  <BrainCircuit size={19} />

                  <div>
                    <strong>
                      {analysisStatus === "success"
                        ? "Prediction completed"
                        : predictionError
                          ? "Prediction unavailable"
                          : "Prediction status"}
                    </strong>

                    <p>
                      {analysisStatus === "success"
                        ? "The backend returned a prediction response."
                        : predictionError ||
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
                    ? "Please wait while the backend receives your recording."
                    : "Upload an audio file to see its backend status and prediction readiness here."}
                </p>
              </div>
            )}
          </section>
        </div>
      </section>

      {/* Analysis History */}
      <section className="history-section" id="history">
        <div className="section-heading">
          <div>
            <span className="eyebrow">YOUR WORKSPACE</span>
            <h2>Analysis History</h2>
            <p>
              Review your previous audio uploads and analysis results.
            </p>
          </div>

          <span className="history-count">
            {analysisHistory.length}{" "}
            {analysisHistory.length === 1 ? "record" : "records"}
          </span>
        </div>

        <div className="history-panel">
          <div className="history-panel-heading">
            <div className="history-title">
              <span className="history-icon">
                <History size={20} />
              </span>

              <div>
                <h3>Previous analyses</h3>
                <p>Your uploaded audio analysis records.</p>
              </div>
            </div>
          </div>

          {analysisHistory.length === 0 ? (
            <div className="history-empty">
              <span className="history-empty-icon">
                <FileAudio size={30} />
              </span>

              <h3>No analysis history yet</h3>
              <p>
                Your previous audio analyses will appear here after
                you upload and process audio recordings.
              </p>

              <a href="#analyzer" className="history-action">
                <FileUp size={16} />
                Upload your first audio
              </a>
            </div>
          ) : (
            <div className="history-content">
              <div className="history-list">
                {analysisHistory.map((record) => (
                  <button
                    type="button"
                    key={record.id}
                    className={`history-record ${
                      selectedAnalysisId === record.id ? "selected" : ""
                    }`}
                    onClick={() => setSelectedAnalysisId(record.id)}
                    aria-pressed={selectedAnalysisId === record.id}
                  >
                    <span className="history-record-icon">
                      <FileAudio size={19} />
                    </span>

                    <span className="history-record-info">
                      <strong>{record.filename}</strong>
                      <span>
                        <Clock3 size={12} />
                        {formatDate(record.createdAt)}
                      </span>
                    </span>

                    <span
                      className={`history-status status-${record.status}`}
                    >
                      {getStatusLabel(record.status)}
                    </span>
                  </button>
                ))}
              </div>

              {selectedAnalysis && (
                <div className="history-detail">
                  <div className="history-detail-heading">
                    <h3>Selected analysis</h3>

                    <span
                      className={`history-status status-${selectedAnalysis.status}`}
                    >
                      {getStatusLabel(selectedAnalysis.status)}
                    </span>
                  </div>

                  <div className="history-detail-grid">
                    <div>
                      <span>Filename</span>
                      <strong>{selectedAnalysis.filename}</strong>
                    </div>

                    <div>
                      <span>Analysis date</span>
                      <strong>
                        {formatDate(selectedAnalysis.createdAt)}
                      </strong>
                    </div>

                    <div>
                      <span>Prediction</span>
                      <strong>
                        {selectedAnalysis.prediction ?? "Not available"}
                      </strong>
                    </div>

                    <div>
                      <span>Confidence</span>
                      <strong>
                        {typeof selectedAnalysis.confidence === "number" &&
                        Number.isFinite(selectedAnalysis.confidence) &&
                        selectedAnalysis.confidence >= 0 &&
                        selectedAnalysis.confidence <= 1
                          ? `${(
                              selectedAnalysis.confidence * 100
                            ).toFixed(1)}%`
                          : "Not available"}
                      </strong>
                    </div>
                  </div>

                  {selectedAnalysis.predictionError && (
                    <p className="history-error">
                      {selectedAnalysis.predictionError}
                    </p>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </section>

      <footer className="footer" id="about">
        <div className="brand">
          <span className="brand-icon">
            <AudioLines size={20} />
          </span>
          <span>AcousticSpace</span>
        </div>

        <p>
          Deepfake detection research through Room Impulse Response.
        </p>

        <span>Research Preview · AcousticSpace</span>
      </footer>
    </main>
  );
}

export default App;
