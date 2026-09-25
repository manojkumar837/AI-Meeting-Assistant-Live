"""AI Meeting Assistant — Streamlit UI.

Live transcription, periodic live summary, topic/action/decision
extraction, and a final downloadable meeting report.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from src.final_summary import build_final_report, report_to_text
from src.inference import DEFAULT_MODEL
from src.realtime import LiveMeetingEngine

ROOT = Path(__file__).resolve().parent
TRANSCRIPTS_DIR = ROOT / "outputs" / "transcripts"
SUMMARIES_DIR = ROOT / "outputs" / "summaries"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

MEETING_TYPES = [
    "General Meeting", "Project Meeting", "Technical Meeting",
    "Client Meeting", "Team Discussion", "Interview", "Sales Meeting",
    "HR Meeting", "Class / Lecture", "Brainstorming",
]
LANGUAGES = ["Auto", "en", "ta", "hi", "te", "ml", "kn"]
WHISPER_MODELS = ["tiny", "base", "small"]

st.set_page_config(page_title="AI Meeting Assistant", page_icon="🎙️", layout="wide")
st.title("🎙️ AI Meeting Assistant")
st.caption("Live transcription • real-time summary • final meeting report")

# -- sidebar: configuration ---------------------------------------------------

with st.sidebar:
    meeting_type = st.selectbox("Meeting Type", MEETING_TYPES)
    language = st.selectbox("Speech Language", LANGUAGES)
    whisper_model = st.selectbox("Whisper Model", WHISPER_MODELS, index=1)
    whisper_device = st.selectbox("Whisper Device", ["cpu", "cuda"])
    compute = "float16" if whisper_device == "cuda" else "int8"
    summary_interval = st.slider("Live summary interval (seconds)", 20, 120, 30, 10)
    model_path = st.text_input("Summarization model", DEFAULT_MODEL)

# -- session state: (re)create the engine if settings changed ----------------

st.session_state.setdefault("engine", None)
st.session_state.setdefault("report", None)

engine = st.session_state.engine
settings_signature = (model_path, whisper_model, whisper_device, compute, summary_interval, language)

if engine is None or not engine.state.active:
    if engine is None or getattr(engine, "signature", None) != settings_signature:
        engine = LiveMeetingEngine(
            model_path,
            whisper_model,
            whisper_device,
            compute,
            chunk_seconds=5,
            summary_interval=summary_interval,
            language=language,
            microphone_device=None,  # auto-detect
        )
        engine.signature = settings_signature
        st.session_state.engine = engine

# -- controls ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔴 START MEETING", type="primary", use_container_width=True,
                 disabled=engine.state.active):
        try:
            engine.start()
            st.rerun()
        except Exception as e:
            st.error(f"Microphone error: {e}")

with col2:
    if st.button("⏹️ END MEETING", use_container_width=True,
                 disabled=not engine.state.active):
        engine.stop()
        transcript = engine.get_transcript()
        if transcript.strip():
            with st.spinner("Generating final overall summary..."):
                try:
                    st.session_state.report = build_final_report(transcript, model_path, meeting_type)
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    (TRANSCRIPTS_DIR / f"meeting_{stamp}.txt").write_text(transcript, encoding="utf-8")
                    (SUMMARIES_DIR / f"meeting_{stamp}.txt").write_text(
                        report_to_text(st.session_state.report), encoding="utf-8"
                    )
                except Exception as e:
                    st.error(f"Final summary error: {e}")
        st.rerun()

with col3:
    if st.button("🗑️ RESET", use_container_width=True, disabled=engine.state.active):
        st.session_state.engine = None
        st.session_state.report = None
        st.rerun()

if engine.state.active:
    st.warning("🔴 LIVE MEETING RUNNING")

# -- live view ------------------------------------------------------------

snapshot = engine.snapshot()
left, right = st.columns(2)

with left:
    st.subheader("🎙️ Live Transcript")
    st.text_area("Transcript", snapshot["transcript"], height=450, label_visibility="collapsed")

with right:
    st.subheader("🧠 Live Summary")
    st.info(snapshot["live_summary"] or "Waiting for the first live summary...")

    st.subheader("📌 Topics")
    for topic in snapshot["topics"]:
        st.write("• " + topic)

    st.subheader("✅ Action Items")
    for item in snapshot["actions"]:
        st.write(f"**{item.get('person') or 'Unassigned'}** — {item['action']} — "
                  f"Deadline: {item.get('deadline') or 'Not detected'}")

    st.subheader("📋 Decisions")
    for decision in snapshot["decisions"]:
        st.write("• " + decision)

if snapshot["error"]:
    st.error(snapshot["error"])

# -- final report -----------------------------------------------------------

st.divider()
st.subheader("📄 Final Overall Meeting Summary")

if st.session_state.report:
    report = st.session_state.report

    st.markdown("### Executive Summary")
    st.write(report["executive_summary"])

    st.markdown("### Key Topics")
    for topic in report["topics"]:
        st.write("• " + topic)

    st.markdown("### Decisions")
    for decision in report["decisions"]:
        st.write("• " + decision)

    st.markdown("### Action Items")
    for item in report["action_items"]:
        st.write(f"**{item.get('person') or 'Unassigned'}** — {item['action']} — "
                  f"Deadline: {item.get('deadline') or 'Not detected'}")

    st.download_button("⬇️ Download Meeting Report", report_to_text(report),
                        "meeting_report.txt", "text/plain", use_container_width=True)
    st.download_button("⬇️ Download Transcript", report["transcript"],
                        "meeting_transcript.txt", "text/plain", use_container_width=True)
else:
    st.info("End the meeting to generate the final overall report.")

# -- live auto-refresh --------------------------------------------------------

if engine.state.active:
    @st.fragment(run_every="1s")
    def refresh_live():
        live_snapshot = engine.snapshot()
        peak = live_snapshot["last_peak"]
        chunk_count = len(live_snapshot["transcript_chunks"])

        # 0.003 is the threshold below which audio is treated as silence
        # and dropped before transcription (see src/transcription.py).
        if peak >= 0.003:
            level_note = f"🟢 mic level OK ({peak:.4f})"
        elif peak > 0:
            level_note = f"🟡 mic level very low ({peak:.4f}) — speak louder or raise mic volume in Windows"
        else:
            level_note = "🔴 no audio detected yet — check mic permissions/selected device"

        st.caption(f"Live chunks transcribed: {chunk_count} · {level_note}")
        if live_snapshot["live_summary"]:
            st.info("Latest live summary: " + live_snapshot["live_summary"])

    refresh_live()
