export type JobStatus = "processing" | "done" | "failed";

export type Verdict = "no_violation" | "violation";

export interface EvidenceFrame {
  timestamp_sec: number;
  image_base64: string;
}

export interface Violation {
  type: string;
  timecode_sec: number;
  description: string;
  evidence_frames: EvidenceFrame[];
}

export interface AnalysisResult {
  job_id: string;
  status: JobStatus;
  verdict: Verdict | null;
  violations: Violation[];
  error: string | null;
}