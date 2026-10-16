# AcousticSpace-Deepfake-Detection-via-Room-Impulse-Response-RIR-


## Overview
AcousticSpace explores audio deepfake detection using Room Impulse Response (RIR) and acoustic features.

## Technology Stack
- Python and FastAPI


## Backend API
- `GET /health` — Checks server status
- `POST /upload` — Uploads audio files
- Supports WAV, MP3, FLAC, and OGG
- Validates file format, size, and empty uploads

## Setup
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

API documentation: http://127.0.0.1:8000/docs

## Current Status
The FastAPI backend and basic upload validation are implemented and tested. 
