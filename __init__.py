"""AI Meeting Assistant — core package.

Modules:
    audio            Microphone capture (sounddevice).
    transcription    Speech-to-text via faster-whisper.
    inference        Abstractive summarization via transformers.
    topics           Lightweight keyword/topic extraction.
    action_items     Rule-based action item extraction.
    decisions        Rule-based decision extraction.
    realtime         Live meeting engine tying the above together.
    final_summary    Builds and formats the end-of-meeting report.
"""
