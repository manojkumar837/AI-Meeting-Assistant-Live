# AI Meeting Assistant — Live Meeting Summarizer

A local live meeting assistant: microphone capture → faster-whisper
transcription → live transcript + periodic live summary → topic / decision /
action-item extraction → final downloadable meeting report.

## Flow

```
Microphone -> Whisper -> Live Transcript -> NLP/Summarization -> Live Summary
                                                              -> END MEETING -> Final Report
```

## Install on Windows

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run:

```cmd
python -m streamlit run app.py
```

Or just double-click `setup_windows.bat`, which does all of the above.

The microphone is captured on the machine running Streamlit — this is a
local prototype, not a hosted service.

For CPU, use Whisper `tiny` or `base`. For NVIDIA CUDA, use `small` or
larger after installing a CUDA-enabled PyTorch build.

The summarizer defaults to `facebook/bart-large-cnn`; you can enter a local
fine-tuned checkpoint path in the sidebar.

Outputs are stored under `outputs/transcripts/` and `outputs/summaries/`.

## Project layout

```
app.py                   Streamlit UI
src/
  audio.py                Microphone capture (auto-detects a working device)
  transcription.py        faster-whisper speech-to-text
  inference.py             transformers summarization
  topics.py                Keyword/topic extraction
  decisions.py             Rule-based decision extraction
  action_items.py          Rule-based action-item extraction (owner + deadline)
  realtime.py              LiveMeetingEngine: wires the above together
  final_summary.py         Builds and formats the end-of-meeting report
test_*.py                 Standalone microphone diagnostics
meeting_summarizer.ipynb  Notebook to sanity-check each component
```

## What changed from the previous version

This pass fixed the bugs that were breaking the live meeting run and
brought the codebase to one consistent style:

- **Fixed the crash** `transcribe_audio_array() got an unexpected keyword
  argument 'model_name'` — `realtime.py` was calling the transcription
  function with argument names (`model_name`, `compute_type`) that don't
  match its real signature (`model_size`, and no `compute_type` at all —
  compute type is derived automatically from the device inside
  `transcription.py`).
- **Fixed the notebook** — it imported a `load_whisper` function that
  doesn't exist; the real function is `get_whisper_model(model_size, device)`.
- **Unified topic/decision/action-item extraction.** Previously `realtime.py`
  had its own inline, slightly different copy of this logic, so the live
  view and the final report could disagree. Both now call the same
  `topics.py` / `decisions.py` / `action_items.py` functions.
- **Wired up `microphone_device`.** It was accepted as a parameter but
  silently ignored (always forced to auto-detect); it's now actually
  passed through to the recorder.
- **Fixed a sample-rate bug in `test_recorder.py`** — it always saved the
  WAV file tagged as 48 kHz regardless of the device's actual capture
  rate, which plays back at the wrong speed/pitch. It now uses the
  recorder's real selected sample rate.
- **General cleanup** — consistent docstrings, type hints, named
  constants instead of magic numbers, and normal (non one-argument-per-line)
  formatting throughout.

## Limitations

This is a working local prototype, not a production meeting platform.
Speaker diarization is not included. Action/decision extraction is
lightweight rule-based NLP, not ML-based structured extraction. For
production use, add speaker diarization, stronger structured extraction,
authentication, persistence, encryption, real-meeting evaluation data, and
stronger long-context summarization.
