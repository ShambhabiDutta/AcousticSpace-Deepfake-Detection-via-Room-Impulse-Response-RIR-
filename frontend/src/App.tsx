import { useRef, useState } from 'react'
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
} from 'lucide-react'
import './App.css'

function App() {
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [error, setError] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = (file?: File) => {
    if (!file) return

    const supportedExtensions = /\.(wav|mp3|flac|ogg|m4a|webm)$/i

    if (!supportedExtensions.test(file.name)) {
      setError('Please select a supported audio file: WAV, MP3, FLAC, OGG, M4A, or WebM.')
      return
    }

    if (file.size > 25 * 1024 * 1024) {
      setError('The maximum file size is 25 MB.')
      return
    }

    setError('')
    setAudioFile(file)
  }

  const formatSize = (bytes: number) => {
    return bytes < 1024 * 1024
      ? `${(bytes / 1024).toFixed(1)} KB`
      : `${(bytes / (1024 * 1024)).toFixed(2)} MB`
  }

  return (
    <div className="app-shell">
      <header className="navbar">
        <a className="brand" href="#" aria-label="AcousticSpace home">
          <span className="brand-icon"><AudioLines size={23} /></span>
          <span>Acoustic<span className="brand-highlight">Space</span></span>
        </a>

        <nav className="nav-links" aria-label="Main navigation">
          <a className="active" href="#home">Home</a>
          <a href="#analyzer">Audio Analyzer</a>
          <a href="#about">About RIR</a>
        </nav>

        <span className="status-pill">
          <span className="status-dot" /> Research Preview
        </span>
      </header>

      <main>
        <section className="hero-section" id="home">
          <div className="hero-copy">
            <div className="eyebrow">
              <Fingerprint size={15} /> AUDIO AUTHENTICITY RESEARCH
            </div>

            <h1>
              Hear beyond the
              <br />
              <span>surface.</span>
            </h1>

            <p className="hero-description">
              Explore a new approach to deepfake audio detection using
              Room Impulse Response (RIR) analysis and acoustic environments.
            </p>

            <a href="#analyzer" className="primary-button">
              <AudioLines size={18} /> Explore Audio Analyzer
            </a>

            <div className="hero-points">
              <span><CheckCircle2 size={16} /> Acoustic analysis</span>
              <span><CheckCircle2 size={16} /> Audio visualization</span>
            </div>
          </div>

          <div className="sound-card" aria-label="Decorative audio waveform illustration">
            <div className="sound-card-top">
              <span className="mini-label">ACOUSTIC SIGNATURE</span>
              <span className="live-indicator"><span /> SIGNAL</span>
            </div>

            <div className="sound-orbit orbit-one" />
            <div className="sound-orbit orbit-two" />

            <div className="sound-center">
              <AudioLines size={43} strokeWidth={1.3} />
            </div>

            <div className="waveform" aria-hidden="true">
              {Array.from({ length: 47 }, (_, i) => (
                <span
                  key={i}
                  style={{
                    height: `${12 + Math.abs(Math.sin(i * 0.58) * Math.cos(i * 0.19)) * 78}px`,
                  }}
                />
              ))}
            </div>

            <div className="sound-card-bottom">
              <span>TIME DOMAIN</span>
              <span>RIR / SIGNAL STUDY</span>
            </div>
          </div>
        </section>

        <section className="feature-strip" id="about">
          <article className="feature-item">
            <span className="feature-icon purple"><AudioLines size={21} /></span>
            <div>
              <h3>Acoustic fingerprints</h3>
              <p>Explore sound characteristics and room responses.</p>
            </div>
          </article>

          <article className="feature-item">
            <span className="feature-icon blue"><BrainCircuit size={21} /></span>
            <div>
              <h3>RIR-based research</h3>
              <p>Investigate room acoustics as a detection signal.</p>
            </div>
          </article>

          <article className="feature-item">
            <span className="feature-icon green"><ShieldCheck size={21} /></span>
            <div>
              <h3>Evidence-led results</h3>
              <p>Prepare for model-backed authenticity analysis.</p>
            </div>
          </article>
        </section>

        <section className="analyzer-section" id="analyzer">
          <div className="section-heading">
            <div>
              <div className="eyebrow">YOUR WORKSPACE</div>
              <h2>Audio analyzer</h2>
              <p>Start by selecting an audio recording for inspection.</p>
            </div>
            <span className="file-limit"><FileAudio size={15} /> Max. 25 MB</span>
          </div>

          <div className="analyzer-grid">
            <div className="upload-panel">
              <div className="panel-heading">
                <span className="step-number">01</span>
                <div>
                  <h3>Upload your audio</h3>
                  <p>Select a recording from your device.</p>
                </div>
              </div>

              <button
                type="button"
                className={`drop-zone ${audioFile ? 'has-file' : ''}`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault()
                  handleFile(event.dataTransfer.files[0])
                }}
              >
                {audioFile ? (
                  <>
                    <span className="upload-icon success"><CheckCircle2 size={28} /></span>
                    <strong>Audio file selected</strong>
                    <span className="drop-hint">Click to choose a different file</span>
                  </>
                ) : (
                  <>
                    <span className="upload-icon"><FileUp size={28} /></span>
                    <strong>Drop your audio here</strong>
                    <span className="drop-hint">or click to browse your files</span>
                    <span className="format-tags">WAV <i /> MP3 <i /> FLAC <i /> OGG</span>
                  </>
                )}
              </button>

              <input
                ref={fileInputRef}
                className="visually-hidden"
                type="file"
                accept=".wav,.mp3,.flac,.ogg,.m4a,.webm,audio/*"
                onChange={(event) => handleFile(event.target.files?.[0])}
              />

              {error && <p className="error-message" role="alert">{error}</p>}

              {audioFile && (
                <div className="selected-file">
                  <span className="file-icon"><FileAudio size={21} /></span>
                  <div className="file-details">
                    <strong>{audioFile.name}</strong>
                    <span>{formatSize(audioFile.size)}</span>
                  </div>
                  <button
                    type="button"
                    className="remove-file"
                    aria-label="Remove selected audio"
                    onClick={() => {
                      setAudioFile(null)
                      setError('')
                      if (fileInputRef.current) fileInputRef.current.value = ''
                    }}
                  >
                    <X size={18} />
                  </button>
                </div>
              )}

              <p className="privacy-note">
                <ShieldCheck size={15} /> Your file stays in this browser for now.
              </p>
            </div>

            <div className="results-panel">
              <div className="panel-heading">
                <span className="step-number">02</span>
                <div>
                  <h3>Analysis workspace</h3>
                  <p>Detection results will appear here.</p>
                </div>
              </div>

              <div className="empty-results">
                <div className="empty-illustration">
                  <Headphones size={31} strokeWidth={1.4} />
                  <span><AudioLines size={19} /></span>
                </div>
                <h3>{audioFile ? 'Ready for analysis' : 'Waiting for audio'}</h3>
                <p>
                  {audioFile
                    ? 'Your file is selected. Connect the detection model to analyze its acoustic features.'
                    : 'Upload an audio file to prepare your analysis workspace.'}
                </p>
                <div className="result-placeholder">
                  <span /><span /><span />
                  <small>MODEL OUTPUT NOT AVAILABLE</small>
                </div>
              </div>
            </div>
          </div>

          <div className="research-note">
            <span className="note-icon"><ShieldCheck size={20} /></span>
            <p>
              <strong>Research transparency</strong>
              <span>
                File selection does not perform deepfake detection. Authenticity
                predictions will require a connected and functioning backend model.
              </span>
            </p>
          </div>
        </section>
      </main>

      <footer className="footer">
        <a className="brand footer-brand" href="#home">
          <span className="brand-icon"><AudioLines size={19} /></span>
          <span>Acoustic<span className="brand-highlight">Space</span></span>
        </a>
        <span>Exploring audio authenticity through room acoustics.</span>
        <span>Research prototype · v0.1</span>
      </footer>
    </div>
  )
}

export default App