from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from faster_whisper import WhisperModel  # type: ignore

from ml.src.config import ATTENTION_CALL_PHRASES, LOOK_PHRASES


TASK_TO_COL = {
    "joint_attention": "joint_attention_path",
    "imitation": "imitation_path",
    "free_play": "free_play_path",
}


@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float


Event = Tuple[str, float, float, float, str]
# (event_type, t_start, t_end, confidence, matched_phrase)


def _run_ffmpeg_extract_wav(video_path: Path, wav_path: Path) -> None:
    """
    Extract mono 16k wav for whisper.
    Requires: ffmpeg available in PATH.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(wav_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def _normalize_token(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9']+", "", s)
    return s


def _flatten_words(segments) -> List[Word]:
    """
    Convert faster-whisper segments to a flat list of Word objects.
    Falls back to segment timestamps if word timestamps are missing.
    """
    words: List[Word] = []
    for seg in segments:
        if getattr(seg, "words", None):
            for w in seg.words:
                words.append(
                    Word(
                        start=float(w.start),
                        end=float(w.end),
                        text=_normalize_token(w.word),
                        prob=float(getattr(w, "probability", 0.0) or 0.0),
                    )
                )
        else:
            # Segment fallback: treat whole segment as one "blob"
            words.append(
                Word(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=_normalize_token(seg.text),
                    prob=0.0,
                )
            )
    return words


def transcribe_words(model: WhisperModel, wav_path: Path, language: Optional[str] = "en") -> List[Word]:
    segments, _info = model.transcribe(
        str(wav_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )
    return _flatten_words(list(segments))


def _find_phrase_events(
    words: List[Word],
    phrases: List[str],
    event_type: str,
    min_conf: float = 0.0,
) -> List[Event]:
    """
    Find phrase occurrences in the word token stream.
    Returns (event_type, t_start, t_end, confidence, matched_phrase).
    Confidence = avg word probability over matched tokens.
    """
    phrase_tokens: List[Tuple[str, List[str]]] = []
    for p in phrases:
        toks = [_normalize_token(t) for t in p.split()]
        toks = [t for t in toks if t]
        if toks:
            phrase_tokens.append((p, toks))

    tokens = [w.text for w in words]
    probs = [w.prob for w in words]

    events: List[Event] = []
    for phrase, ptoks in phrase_tokens:
        k = len(ptoks)
        for i in range(0, len(tokens) - k + 1):
            if tokens[i : i + k] == ptoks:
                t_start = words[i].start
                t_end = words[i + k - 1].end
                conf = float(sum(probs[i : i + k]) / max(k, 1))
                if conf >= min_conf:
                    events.append((event_type, t_start, t_end, conf, phrase))
    return events


def dedupe_keep_longest(events: List[Event], overlap_sec: float = 0.10) -> List[Event]:
    """
    Deduplicate overlapping phrase matches of the SAME event_type.
    Keep the longest duration (then higher confidence).
    """
    if not events:
        return events

    # sort by start time, prefer longer and higher confidence first
    events_sorted = sorted(events, key=lambda x: (x[1], -(x[2] - x[1]), -x[3]))

    kept: List[Event] = []
    for e in events_sorted:
        etype, s, t, conf, phrase = e

        too_much_overlap = False
        for k in kept:
            ktype, ks, kt, *_ = k
            if ktype != etype:
                continue
            overlap = max(0.0, min(t, kt) - max(s, ks))
            if overlap >= overlap_sec:
                too_much_overlap = True
                break

        if not too_much_overlap:
            kept.append(e)

    return sorted(kept, key=lambda x: x[1])


def main():
    ap = argparse.ArgumentParser(description="Extract audio events (CALL_ATTENTION + LOOK) with timestamps.")
    ap.add_argument("--manifest", required=True, help="CSV manifest (ideally data/derived/manifest_usable.csv)")
    ap.add_argument("--task", default="joint_attention", choices=list(TASK_TO_COL.keys()))
    ap.add_argument("--out", default="data/derived/audio_events.csv")
    ap.add_argument("--model", default="base", help="faster-whisper model: tiny/base/small/medium/large-v3")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--compute_type", default="int8", help="cpu best: int8")
    ap.add_argument("--limit", type=int, default=0, help="Process first N rows (0 = all)")
    ap.add_argument("--min_conf", type=float, default=0.0, help="Min avg word prob to keep event")
    ap.add_argument("--overlap_sec", type=float, default=0.10, help="Overlap window for dedupe (sec)")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    col = TASK_TO_COL[args.task]
    if col not in df.columns:
        raise ValueError(f"Manifest missing column '{col}'. Present: {list(df.columns)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize whisper model once
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    fieldnames = ["child_id", "task_type", "event_type", "t_start", "t_end", "confidence", "matched_phrase"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

    n = len(df) if args.limit <= 0 else min(args.limit, len(df))
    for i in range(n):
        child_id = str(df.iloc[i]["child_id"])
        video_path = Path(str(df.iloc[i][col]))
        if not video_path.exists():
            continue

        print(f"[{i+1}/{n}] child={child_id} task={args.task} video={video_path.name}", flush=True)

        # Extract audio -> wav -> transcribe
        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "audio.wav"
            try:
                _run_ffmpeg_extract_wav(video_path, wav_path)
            except Exception:
                # No audio stream or ffmpeg failed
                continue

            try:
                words = transcribe_words(model, wav_path, language="en")
            except Exception:
                continue

        # Extract events (non-overlapping phrase sets)
        call_events = _find_phrase_events(words, ATTENTION_CALL_PHRASES, "CALL_ATTENTION", min_conf=args.min_conf)
        look_events = _find_phrase_events(words, LOOK_PHRASES, "LOOK", min_conf=args.min_conf)

        # Deduplicate overlaps within each type
        call_events = dedupe_keep_longest(call_events, overlap_sec=args.overlap_sec)
        look_events = dedupe_keep_longest(look_events, overlap_sec=args.overlap_sec)

        # Write events incrementally
        with open(out_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            for (etype, ts, te, conf, phrase) in call_events + look_events:
                w.writerow(
                    {
                        "child_id": child_id,
                        "task_type": args.task,
                        "event_type": etype,
                        "t_start": round(ts, 3),
                        "t_end": round(te, 3),
                        "confidence": round(conf, 4),
                        "matched_phrase": phrase,
                    }
                )

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
