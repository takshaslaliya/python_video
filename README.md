# 🎬 Python AI Video Generator Engine

This repository contains the standalone, local-first Python AI engine that powers the **AIToolHub** suite. It splits input text scripts, generates neural speech narration, fetches stock or AI-generated visuals, builds captions, and merges them using MoviePy and FFmpeg. It also supports independent AI Image Generation (latent diffusion) and AI Copywriting/Script Writing (GPT-style prompt completions).

## 📦 Directory Contents

*   `main.py`: FastAPI web service exposing `/generate` and `/status/{taskId}` endpoints.
*   `video_generator.py`: Core composition module (handles Edge-TTS, Pexels Stock photo API fallback, Pillow procedural graphics generator, SRT builder, and FFmpeg subtitle burner).
*   `requirements.txt`: Pip dependencies list.
*   `.gitignore`: Prevents virtual environments, temporary assets, and compiled videos from being tracked.

## 🚀 Running the Engine Standalone

1.  **Create and Activate Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Start FastAPI Service**:
    ```bash
    python main.py
    ```
    The engine will start on `http://localhost:8000`.

## ⚙️ REST endpoints

*   `POST /generate`: Start a video generation task in the background. Returns a `taskId`.
*   `GET /status/{taskId}`: Get progress percentage and step logs.
*   `GET /health`: Check GPU/CUDA compatibility.
