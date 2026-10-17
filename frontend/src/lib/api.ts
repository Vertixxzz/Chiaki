import type { AnalysisResult } from "../types";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `HTTP ${response.status}`;

    try {
      const body = await response.json();
      message = body.detail ?? message;
    } catch {
      // Ответ сервера не обязательно содержит JSON.
    }

    throw new Error(String(message));
  }

  return response.json() as Promise<T>;
}

export async function uploadVideo(
  file: File,
  apiKey: string,
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
    method: "POST",
    headers: {
      "X-API-Key": apiKey,
    },
    body: formData,
  });

  return parseResponse<AnalysisResult>(response);
}

export async function getJob(
  jobId: string,
  apiKey: string,
): Promise<AnalysisResult> {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/jobs/${encodeURIComponent(jobId)}`,
    {
      method: "GET",
      headers: {
        "X-API-Key": apiKey,
      },
    },
  );

  return parseResponse<AnalysisResult>(response);
}