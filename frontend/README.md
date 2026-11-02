
# AcousticSpace â€“ Frontend

AcousticSpace is a web-based audio analysis dashboard for a deepfake detection project using Room Impulse Response (RIR).

The frontend is built with React, TypeScript, Vite, and WaveSurfer.js. It provides audio upload, waveform visualization, playback controls, and integration with the FastAPI backend.

## Tech Stack

- React 19
- TypeScript
- Vite
- WaveSurfer.js
- Lucide React
- CSS

## Prerequisites

- Node.js and npm
- Python environment for running the backend
- FastAPI backend configured locally

## Getting Started

From the project root, navigate to the frontend:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open the local URL displayed in the terminal.

## Available Scripts

Run these commands from the `frontend` directory:

| Command | Description |
|---|---|
| `npm run dev` | Starts the development server |
| `npm run build` | Runs TypeScript checks and creates a production build |
| `npm run lint` | Runs ESLint |
| `npm run preview` | Previews the production build |

## Backend Integration

The frontend communicates with the FastAPI backend.

Default API base URL:

```text
http://127.0.0.1:8000
```

You can override it by creating a `.env` file inside the `frontend` directory:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The backend provides these endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Checks backend availability |
| `/upload` | POST | Uploads an audio file |
| `/predict` | POST | Requests an audio prediction |

The backend currently supports WAV, MP3, FLAC, and OGG files, with a maximum upload size of 20 MB.

**Prediction status:** The prediction interface is integrated, but actual predictions require the trained model and audio preprocessing to be configured on the backend. Until then, the dashboard should indicate that prediction is unavailable rather than display fabricated results.

## Frontend Regression Testing Checklist

The following checks can be performed manually through the browser.

### 1. Audio Selection and Validation

- [ ] Select an audio file using the browse option.
- [ ] Drag and drop a supported audio file.
- [ ] Verify WAV, MP3, FLAC, and OGG files are accepted.
- [ ] Verify unsupported file formats show an error.
- [ ] Verify empty files are rejected.
- [ ] Verify files larger than 20 MB are rejected.
- [ ] Verify selecting an invalid file clears the previous selection.

### 2. Waveform and Playback

- [ ] Verify the selected audio waveform loads.
- [ ] Verify the audio duration is displayed.
- [ ] Test play and pause controls.
- [ ] Verify the playback position updates.
- [ ] Verify removing the selected file resets the audio player.

### 3. Backend Integration

- [ ] Start the FastAPI backend.
- [ ] Verify the dashboard can connect to the backend.
- [ ] Upload a valid supported audio file.
- [ ] Verify the upload success message appears.
- [ ] Verify the uploaded filename and status are displayed.
- [ ] Stop the backend and verify that connection errors are handled clearly.

### 4. Prediction Feedback

- [ ] Verify the prediction request is handled after upload.
- [ ] Verify unavailable prediction status is displayed when the model is not configured.
- [ ] Verify confidence is shown only when a valid confidence value is returned.
- [ ] Verify API errors do not produce misleading prediction results.

### 5. Responsive Layout and Accessibility

- [ ] Check the dashboard at desktop width.
- [ ] Check the dashboard at tablet width.
- [ ] Check the dashboard at mobile width.
- [ ] Verify file names do not overflow their containers.
- [ ] Verify buttons and links have visible keyboard focus.
- [ ] Verify upload and result feedback is accessible.

## Production Verification

Run the following commands from the project root:

```bash
npm run build --prefix frontend
npm run lint --prefix frontend
```

A successful build and lint run indicate that the frontend passes the available TypeScript, production-build, and ESLint checks.
