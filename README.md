# 🎙️ AI Meeting Assistant Live

An AI-powered live meeting assistant that captures microphone audio, converts speech into text in real time, generates live summaries, and extracts important meeting information such as topics, decisions, and action items.

The application is built using Python, Streamlit, Faster-Whisper, Hugging Face Transformers, and NLP techniques.

---

## 🚀 Features

### 🎙️ Live Audio Capture
- Captures speech directly from the microphone.
- Automatically detects a suitable microphone input.
- Supports continuous audio capture.
- Handles audio chunks for near real-time processing.

### 📝 Real-Time Transcription
- Converts speech into text using Faster-Whisper.
- Displays the meeting transcript while the meeting is running.
- Supports multiple speech languages.
- Uses audio resampling before Whisper inference.

### 🧠 Live Meeting Summary
- Generates periodic summaries during the meeting.
- Uses Transformer-based summarization.
- Supports pretrained BART models.
- Configurable summary interval.

### 📌 Topic Detection
Automatically identifies important discussion topics from the transcript.

### ✅ Action Item Extraction
Detects statements related to:
- Tasks
- Follow-ups
- Responsibilities
- Work to be completed
- Deadlines when available

### 📋 Decision Detection
Identifies statements containing:
- Decisions
- Agreements
- Approvals
- Final decisions

### 📄 Final Meeting Report

After ending the meeting, the application generates a final report containing:

- Executive Summary
- Key Topics
- Decisions
- Action Items
- Deadlines
- Meeting Transcript

### 💾 Export

The application allows users to download:

- Meeting report
- Complete transcript

---

# 🏗️ System Architecture

```text
                 ┌─────────────────────┐
                 │     Microphone      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Audio Capture     │
                 │   SoundDevice       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Audio Processing    │
                 │ Resampling / Queue  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Faster-Whisper    │
                 │ Speech-to-Text      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Live Transcript    │
                 └──────────┬──────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  Topics  │  │ Decisions│  │  Actions │
        └──────────┘  └──────────┘  └──────────┘
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                 ┌─────────────────────┐
                 │ AI Meeting Summary  │
                 │ BART Transformer    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Final Meeting      │
                 │ Report              │
                 └─────────────────────┘
