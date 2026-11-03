import type { AudioUploadResponse } from "../services/audioApi";

export type AnalysisStatus =
  | "uploaded"
  | "completed"
  | "unavailable";

export interface AnalysisRecord {
  id: string;
  filename: string;
  createdAt: string;
  prediction?: string;
  confidence?: number;
  status: AnalysisStatus;
  uploadResult: AudioUploadResponse;
  predictionError?: string;
}
