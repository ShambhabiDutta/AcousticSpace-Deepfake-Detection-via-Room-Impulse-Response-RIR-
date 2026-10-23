/**
 * AcousticSpace - Audio API Service
 * Connects the frontend to the FastAPI backend.
 */

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export interface AudioUploadResponse {
  message: string;
  filename: string;
  saved_filename: string;
  size_bytes: number;
  status: string;
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) return false;

    const data = await response.json();
    return data.status === "healthy";
  } catch {
    return false;
  }
}

export async function uploadAudio(file: File): Promise<AudioUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}/upload`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error(
      "Cannot connect to the backend. Make sure the FastAPI server is running.",
    );
  }

  if (!response.ok) {
    let message = `Upload failed (${response.status}).`;

    try {
      const data = await response.json();
      if (typeof data.detail === "string") {
        message = data.detail;
      }
    } catch {
      // Keep the default error message if the response is not JSON.
    }

    throw new Error(message);
  }

  return (await response.json()) as AudioUploadResponse;
}

export { API_BASE_URL };
