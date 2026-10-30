
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

async function getResponseData(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    throw new Error(
      `The server returned an invalid response (${response.status}).`,
    );
  }
}

function getErrorMessage(data: unknown, fallback: string): string {
  if (typeof data === "object" && data !== null) {
    const responseData = data as {
      detail?: unknown;
      message?: unknown;
    };

    if (typeof responseData.detail === "string") {
      return responseData.detail;
    }

    if (typeof responseData.message === "string") {
      return responseData.message;
    }
  }

  return fallback;
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);

    if (!response.ok) {
      return false;
    }

    const data = await response.json();
    return data.status === "healthy";
  } catch {
    return false;
  }
}

export async function uploadAudio(
  file: File,
): Promise<AudioUploadResponse> {
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

  const data = await getResponseData(response);

  if (!response.ok) {
    throw new Error(
      getErrorMessage(data, `Upload failed (${response.status}).`),
    );
  }

  return data as AudioUploadResponse;
}

/**
 * Sends audio to the prediction endpoint.
 * The backend currently returns HTTP 503 until model inference
 * and audio preprocessing are configured.
 */
export async function predictAudio(file: File): Promise<unknown> {
  const formData = new FormData();
  formData.append("file", file);

  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error(
      "Cannot connect to the prediction service. Check that the backend is running.",
    );
  }

  const data = await getResponseData(response);

  if (!response.ok) {
    throw new Error(
      getErrorMessage(data, `Prediction failed (${response.status}).`),
    );
  }

  return data;
}

export { API_BASE_URL };
