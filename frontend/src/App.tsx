import { useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowDownToLine,
  CheckCircle2,
  Clock3,
  FileVideo,
  Fingerprint,
  LoaderCircle,
  ScanEye,
  ShieldCheck,
  UploadCloud,
  X,
  XCircle,
} from "lucide-react";

import { getJob, uploadVideo } from "./lib/api";
import type { AnalysisResult, EvidenceFrame } from "./types";
import "./App.css";

const POLL_INTERVAL_MS = 1500;
const MAX_ANALYSIS_TIME_MS = 5 * 60 * 1000;

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(0)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  const remainingSeconds = total % 60;

  return `${String(minutes).padStart(2, "0")}:${String(
    remainingSeconds,
  ).padStart(2, "0")}`;
}

function getEvidenceSrc(frame: EvidenceFrame): string {
  if (frame.image_base64.startsWith("data:")) {
    return frame.image_base64;
  }

  return `data:image/jpeg;base64,${frame.image_base64}`;
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  return "An unexpected error occurred.";
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const activeRequestRef = useRef(false);

  const isProcessing = result?.status === "processing";

  useEffect(() => {
    if (!isProcessing) {
      return;
    }

    const startedAt = Date.now();

    const interval = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);

    return () => window.clearInterval(interval);
  }, [result?.job_id, isProcessing]);

  function selectFile(candidate: File | undefined) {
    if (!candidate) {
      return;
    }

    setError("");

    if (!candidate.name.toLowerCase().endsWith(".mov")) {
      setError("Please select a .mov video file.");
      return;
    }

    setFile(candidate);
    setResult(null);
    setElapsedSeconds(0);
  }

  async function handleAnalyze() {
    if (!file || !apiKey.trim() || activeRequestRef.current) {
      return;
    }

    activeRequestRef.current = true;
    setIsSubmitting(true);
    setError("");
    setResult(null);
    setElapsedSeconds(0);

    try {
      const initialResult = await uploadVideo(file, apiKey.trim());
      setResult(initialResult);

      if (initialResult.status !== "processing") {
        return;
      }

      const startedAt = Date.now();
      let currentResult = initialResult;

      while (currentResult.status === "processing") {
        if (Date.now() - startedAt >= MAX_ANALYSIS_TIME_MS) {
          throw new Error(
            "Analysis polling timed out. Your job may still be running.",
          );
        }

        await new Promise<void>((resolve) =>
          window.setTimeout(resolve, POLL_INTERVAL_MS),
        );

        currentResult = await getJob(initialResult.job_id, apiKey.trim());
        setResult(currentResult);
      }
    } catch (caughtError) {
      setError(getErrorMessage(caughtError));
    } finally {
      activeRequestRef.current = false;
      setIsSubmitting(false);
    }
  }

  function resetAnalysis() {
    setFile(null);
    setResult(null);
    setError("");
    setElapsedSeconds(0);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  const canAnalyze =
    file !== null && apiKey.trim().length > 0 && !isSubmitting;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" aria-label="Cashier Vision AI home">
          <div className="brand-mark">
            <ScanEye size={23} />
          </div>
          <div className="brand-copy">
            <span>Chiaki</span>
            <small>CASHIER INTELLIGENCE</small>
          </div>
        </a>

        <div className="sidebar-divider" />

        <p className="nav-label">WORKSPACE</p>
        <div className="nav-item active">
          <Activity size={18} />
          <span>Video analysis</span>
          <span className="nav-indicator" />
        </div>

        <div className="sidebar-bottom">
          <div className="local-status">
            <span className="status-dot" />
            <div>
              <strong>Local inference</strong>
              <small>API connection configured</small>
            </div>
          </div>

          <div className="sidebar-version">
            <ShieldCheck size={15} />
            <span>Chiaki · v0.1.0</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumbs">
            <span>Workspace</span>
            <span className="breadcrumb-separator">/</span>
            <strong>Video analysis</strong>
          </div>

          <div className="topbar-right">
            <span className="environment-badge">
              <span className="status-dot" />
              DEVELOPMENT
            </span>
            <div className="avatar">
              <Fingerprint size={18} />
            </div>
          </div>
        </header>

        <div className="page-content">
          <section className="page-heading">
            <div>
              <p className="eyebrow">VIDEO INTELLIGENCE / 001</p>
              <h1>Cashier analysis</h1>
              <p className="page-description">
                Analyze cashier-operation footage and review visually
                detected events.
              </p>
            </div>

            <div className="privacy-note">
              <ShieldCheck size={16} />
              <span>Designed for local processing</span>
            </div>
          </section>

          <section className="workspace-grid">
            <div className="panel upload-panel">
              <div className="panel-heading">
                <div className="section-icon">
                  <UploadCloud size={19} />
                </div>
                <div>
                  <h2>Upload footage</h2>
                  <p>Select a video to start a new analysis job.</p>
                </div>
              </div>

              <button
                className={`dropzone ${isDragging ? "dragging" : ""} ${
                  file ? "has-file" : ""
                }`}
                type="button"
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(event) => {
                  event.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setIsDragging(false);
                  selectFile(event.dataTransfer.files[0]);
                }}
                disabled={isSubmitting}
              >
                {file ? (
                  <>
                    <div className="file-icon">
                      <FileVideo size={27} />
                    </div>
                    <strong className="dropzone-title">{file.name}</strong>
                    <span className="dropzone-subtitle">
                      {formatBytes(file.size)} · Ready for analysis
                    </span>
                    <span className="change-file">Choose another file</span>
                  </>
                ) : (
                  <>
                    <div className="upload-icon">
                      <UploadCloud size={29} />
                    </div>
                    <strong className="dropzone-title">
                      Drop your video here
                    </strong>
                    <span className="dropzone-subtitle">
                      or click to browse your files
                    </span>
                    <span className="format-chip">MOV VIDEO</span>
                  </>
                )}
              </button>

              <input
                ref={fileInputRef}
                type="file"
                accept=".mov,video/quicktime"
                hidden
                onChange={(event) => selectFile(event.target.files?.[0])}
              />

              <div className="upload-meta">
                <span><CheckCircle2 size={14} /> MOV format</span>
                <span><CheckCircle2 size={14} /> Max. 200 MB</span>
              </div>

              <label className="field-label" htmlFor="api-key">
                API ACCESS KEY
              </label>
              <div className="api-key-input">
                <Fingerprint size={17} />
                <input
                  id="api-key"
                  type={showApiKey ? "text" : "password"}
                  placeholder="Enter your API key"
                  value={apiKey}
                  onChange={(event) => setApiKey(event.target.value)}
                  autoComplete="off"
                  spellCheck={false}
                />
                <button
                  type="button"
                  className="text-button"
                  onClick={() => setShowApiKey((current) => !current)}
                >
                  {showApiKey ? "Hide" : "Show"}
                </button>
              </div>
              <p className="field-hint">
                Your key is kept in this page's memory and is not saved.
              </p>

              {error && (
                <div className="error-banner" role="alert">
                  <AlertCircle size={17} />
                  <span>{error}</span>
                  <button
                    type="button"
                    aria-label="Dismiss error"
                    onClick={() => setError("")}
                  >
                    <X size={16} />
                  </button>
                </div>
              )}

              <button
                className="primary-button"
                type="button"
                disabled={!canAnalyze}
                onClick={handleAnalyze}
              >
                {isSubmitting ? (
                  <>
                    <LoaderCircle className="spin" size={18} />
                    {isProcessing ? "Analyzing footage..." : "Submitting..."}
                  </>
                ) : (
                  <>
                    <ScanEye size={18} />
                    Analyze video
                  </>
                )}
              </button>

              <p className="privacy-footnote">
                <ShieldCheck size={14} />
                Video is sent only to your configured analysis API.
              </p>
            </div>

            <div className="panel result-panel">
              <div className="panel-heading result-heading">
                <div className="section-icon">
                  <Activity size={19} />
                </div>
                <div>
                  <h2>Analysis report</h2>
                  <p>Review the status and detected events.</p>
                </div>
                {result && (
                  <button
                    className="icon-button"
                    type="button"
                    onClick={resetAnalysis}
                    aria-label="Clear result"
                    title="Clear result"
                  >
                    <X size={17} />
                  </button>
                )}
              </div>

              {!result && !isSubmitting && (
                <div className="empty-state">
                  <div className="empty-illustration">
                    <div className="empty-ring ring-one" />
                    <div className="empty-ring ring-two" />
                    <div className="empty-center">
                      <ScanEye size={34} />
                    </div>
                  </div>
                  <h3>Nothing to review yet</h3>
                  <p>
                    Upload a video and start an analysis. The report will
                    appear here when the API returns a result.
                  </p>
                  <div className="empty-hint">
                    <Clock3 size={15} />
                    <span>Results appear asynchronously</span>
                  </div>
                </div>
              )}

              {isSubmitting && !result && (
                <div className="empty-state">
                  <LoaderCircle className="spin loading-large" size={38} />
                  <h3>Submitting video</h3>
                  <p>Uploading footage and creating an analysis job…</p>
                </div>
              )}

              {result?.status === "processing" && (
                <div className="processing-state">
                  <div className="processing-banner">
                    <LoaderCircle className="spin" size={20} />
                    <div>
                      <strong>Analysis in progress</strong>
                      <p>The worker is processing your video.</p>
                    </div>
                  </div>

                  <div className="job-detail">
                    <span>JOB IDENTIFIER</span>
                    <code>{result.job_id}</code>
                  </div>
                  <div className="job-detail">
                    <span>ELAPSED TIME</span>
                    <strong>{formatTime(elapsedSeconds)}</strong>
                  </div>
                  <div className="progress-track">
                    <div className="progress-indeterminate" />
                  </div>
                  <p className="processing-caption">
                    This is an indeterminate progress indicator; actual
                    processing progress is not reported by the current API.
                  </p>
                </div>
              )}

              {result?.status === "failed" && (
                <div className="failure-state">
                  <div className="verdict-icon failure-icon">
                    <XCircle size={27} />
                  </div>
                  <span className="result-kicker">TASK FAILED</span>
                  <h3>Analysis could not be completed</h3>
                  <p>
                    {result.error === "analyzer_not_configured"
                      ? "The API is working, but the real ML analyzer has not been connected yet."
                      : result.error || "An unexpected analysis error occurred."}
                  </p>
                  <div className="job-detail">
                    <span>JOB IDENTIFIER</span>
                    <code>{result.job_id}</code>
                  </div>
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={resetAnalysis}
                  >
                    Start a new analysis
                  </button>
                </div>
              )}

              {result?.status === "done" && (
                <div className="completed-state">
                  <div
                    className={`verdict-card ${
                      result.verdict === "violation"
                        ? "verdict-warning"
                        : "verdict-clear"
                    }`}
                  >
                    <div className="verdict-icon">
                      {result.verdict === "violation" ? (
                        <AlertTriangle size={25} />
                      ) : (
                        <CheckCircle2 size={25} />
                      )}
                    </div>
                    <div>
                      <span className="result-kicker">ANALYSIS VERDICT</span>
                      <h3>
                        {result.verdict === "violation"
                          ? "Potential violation detected"
                          : result.verdict === "no_violation"
                            ? "No violation detected"
                            : "Analysis completed"}
                      </h3>
                      <p>
                        {result.violations.length} event
                        {result.violations.length === 1 ? "" : "s"} reported
                      </p>
                    </div>
                  </div>

                  <div className="report-summary">
                    <div>
                      <span>REPORTED EVENTS</span>
                      <strong>{result.violations.length}</strong>
                    </div>
                    <div>
                      <span>JOB IDENTIFIER</span>
                      <code>{result.job_id.slice(0, 13)}…</code>
                    </div>
                  </div>

                  {result.violations.length === 0 ? (
                    <div className="no-events">
                      <CheckCircle2 size={18} />
                      <span>No events were returned by the analyzer.</span>
                    </div>
                  ) : (
                    <div className="violations-list">
                      <h3 className="list-heading">Detected events</h3>
                      {result.violations.map((violation, index) => (
                        <article
                          className="violation-card"
                          key={`${violation.type}-${violation.timecode_sec}-${index}`}
                        >
                          <div className="violation-topline">
                            <span className="event-number">
                              EVENT {String(index + 1).padStart(2, "0")}
                            </span>
                            <span className="timecode">
                              <Clock3 size={13} />
                              {formatTime(violation.timecode_sec)}
                            </span>
                          </div>

                          <h4>{violation.type.replaceAll("_", " ")}</h4>
                          <p>{violation.description}</p>

                          {violation.evidence_frames.length > 0 && (
                            <div className="evidence-grid">
                              {violation.evidence_frames.map(
                                (frame, frameIndex) => (
                                  <a
                                    className="evidence-item"
                                    href={getEvidenceSrc(frame)}
                                    target="_blank"
                                    rel="noreferrer"
                                    key={`${frame.timestamp_sec}-${frameIndex}`}
                                  >
                                    <img
                                      src={getEvidenceSrc(frame)}
                                      alt={`Evidence at ${formatTime(frame.timestamp_sec)}`}
                                    />
                                    <span>
                                      <Clock3 size={12} />
                                      {formatTime(frame.timestamp_sec)}
                                      <ArrowDownToLine size={13} />
                                    </span>
                                  </a>
                                ),
                              )}
                            </div>
                          )}
                        </article>
                      ))}
                    </div>
                  )}

                  <p className="review-disclaimer">
                    Results flag visual indicators for human review. They do
                    not establish that a legal violation occurred.
                  </p>
                </div>
              )}
            </div>
          </section>

          <footer className="page-footer">
            <span> CASHIER VISION AI <span className="footer-dot">/</span> LOCAL VIDEO ANALYTICS </span>
            <span><ShieldCheck size={14} /> Privacy-conscious by design</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;