"""Abstractive summarization via a transformers seq2seq pipeline."""

from __future__ import annotations

from functools import lru_cache

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

DEFAULT_MODEL = "facebook/bart-large-cnn"
CHUNK_WORD_LIMIT = 650
MIN_WORDS_TO_SUMMARIZE = 12
RESUMMARIZE_WORD_THRESHOLD = 80


@lru_cache(maxsize=2)
def load_summarizer(model_path: str = DEFAULT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return pipeline("summarization", model=model, tokenizer=tokenizer)


def chunk_text(text: str, max_words: int = CHUNK_WORD_LIMIT) -> list[str]:
    words = (text or "").split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def summarize(
    text: str,
    model_path: str = DEFAULT_MODEL,
    max_length: int = 130,
    min_length: int = 30,
) -> str:
    """Summarize text of any length: chunk it, summarize each chunk, then
    (if needed) summarize the concatenation of chunk summaries."""
    text = (text or "").strip()
    if not text:
        return ""

    summarizer = load_summarizer(model_path)
    chunks = chunk_text(text)

    summaries = []
    for chunk in chunks:
        if len(chunk.split()) < MIN_WORDS_TO_SUMMARIZE:
            summaries.append(chunk)
            continue
        try:
            result = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(result[0]["summary_text"].strip())
        except Exception:
            summaries.append(chunk)

    combined = " ".join(summaries)
    if len(summaries) > 1 and len(combined.split()) > RESUMMARIZE_WORD_THRESHOLD:
        result = summarizer(combined, max_length=max_length, min_length=min_length, do_sample=False)
        return result[0]["summary_text"].strip()

    return combined.strip()
