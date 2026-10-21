
/**
 * AcousticSpace - Audio API Configuration
 *
 * This file will connect the frontend to the FastAPI backend.
 * The actual endpoint and request format will be added after
 * the backend details are confirmed by the backend team member.
 */

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export { API_BASE_URL };
