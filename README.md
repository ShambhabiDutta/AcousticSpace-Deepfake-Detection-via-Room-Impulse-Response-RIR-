# AcousticSpace-Deepfake-Detection-via-Room-Impulse-Response-RIR-


## Overview

AcousticSpace is a research project focused on audio deepfake detection using Room Impulse Response (RIR) and acoustic features. The project aims to investigate how acoustic characteristics can support the identification of synthetic or manipulated audio.

## Technology Stack

- Python
- FastAPI
- Uvicorn
- pytest and HTTPX (backend testing)

## Backend API

The backend is implemented using FastAPI and provides health-check and audio-upload endpoints.

### Available Endpoints

| Method | Endpoint  | Description                                |
| ------ | --------- | ------------------------------------------ |
| GET    | `/`       | Confirms that the API is running           |
| GET    | `/health` | Returns the backend health status          |
| POST   | `/upload` | Accepts and validates an audio file upload |

### Audio Upload Validation

- Supported file extensions: `.wav`, `.mp3`, `.flac`, `.ogg`
- Maximum file size: 20 MB
- Empty files are rejected.
- Unsupported file formats are rejected.
- Uploaded files are saved using unique generated filenames to avoid overwriting existing files.

### Backend Setup

Ensure Python is installed. Open a terminal at the project root and run:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Start the API Server

From the `backend` directory, run:

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### API Documentation

http://127.0.0.1:8000/docs


### Run Backend Tests

From the `backend` directory, run:

```powershell
python -m pytest -v
```

The backend tests cover the root endpoint, health endpoint, successful audio uploads, unsupported formats, empty files, and oversized uploads.

## Current Implementation Status

The FastAPI server, health-check endpoints, audio-upload validation, and backend tests are implemented.

