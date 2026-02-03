from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import wavfile  # type: ignore
from faster_whisper import WhisperModel  # type: ignore

from ml.src.config import ATTENTION_CALL_PHRASES, LOOK_PHRASES


TASK_TO_COL = {
    "joint_attention": "joint_attention_path",
    "imitation": "imitation_path",
    "free_play": "free_play_path",
}


# ---------------------------
# Data structures
# ---------------------------
@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float


@dataclass
class AudioEvent:
    """Enhanced audio event with multi-source confidence."""
    event_type: str  # "CALL_ATTENTION", "LOOK"
    t_start: float  # start time (sec)
    t_end: float  # end time (sec)
    matched_phrase: str  # the phrase that matched
    stt_confidence: float  # STT word probability (0-1)
    energy_confidence: float  # normalized energy in region (0-1)
    combined_confidence: float  # weighted combination
    is_vad_confirmed: bool  # whether VAD detected speech here


@dataclass
class AudioQualityReport:
    """Per-video audio quality metrics."""
    duration_sec: float  # audio duration
    speech_ratio: float  # % of audio with detected speech (VAD)
    mean_energy_db: float  # mean RMS energy in dB
    snr_estimate: float  # signal-to-noise ratio estimate
    n_speech_segments: int  # number of speech segments from VAD
    stt_word_count: int  # total words transcribed
    has_audio: bool  # whether audio track exists


@dataclass
class ParentPromptSummary:
    """Per-video summary of parent calling attempts."""
    child_id: str
    task_type: str
    attempts_count: int  # total parent calling attempts
    attempt_timestamps: List[float] = field(default_factory=list)  # list of t_start
    prompt_confidence_mean: float = 0.0  # mean confidence across attempts
    prompt_confidence_min: float = 0.0  # min confidence (weakest detection)
    call_attention_count: int = 0  # CALL_ATTENTION events
    look_count: int = 0  # LOOK events
    # Quality metrics (flattened for CSV)
    audio_duration_sec: float = 0.0
    speech_ratio: float = 0.0
    snr_estimate: float = 0.0


# Legacy tuple format for backward compatibility
Event = Tuple[str, float, float, float, str]
# (event_type, t_start, t_end, confidence, matched_phrase)


# ---------------------------
# Energy-based VAD
# ---------------------------
def compute_energy_envelope(
    audio: np.ndarray,
    sr: int = 16000,
    frame_ms: int = 25,
    hop_ms: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RMS energy envelope of audio signal.

    Args:
        audio: 1D audio signal (mono)
        sr: sample rate
        frame_ms: frame length in milliseconds
        hop_ms: hop length in milliseconds

    Returns:
        (times, energy_db) arrays where energy_db is in decibels
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    # Pad audio to ensure we can compute frames
    n_frames = max(1, (len(audio) - frame_len) // hop_len + 1)

    energy = np.zeros(n_frames, dtype=np.float32)
    times = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_len
        end = min(start + frame_len, len(audio))
        frame = audio[start:end].astype(np.float32)

        # RMS energy
        rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        energy[i] = rms
        times[i] = (start + end) / 2 / sr

    # Convert to dB (reference: max possible amplitude = 1.0 for normalized audio)
    energy_db = 20 * np.log10(energy + 1e-10)

    return times, energy_db


def estimate_adaptive_threshold(
    energy_db: np.ndarray,
    noise_percentile: float = 10.0,
    threshold_above_noise_db: float = 10.0,
) -> float:
    """
    Estimate an adaptive energy threshold based on noise floor.

    Args:
        energy_db: energy envelope in dB
        noise_percentile: percentile to estimate noise floor (default 10th)
        threshold_above_noise_db: dB above noise floor for speech threshold

    Returns:
        Adaptive threshold in dB
    """
    if len(energy_db) == 0:
        return -40.0  # fallback

    noise_floor = float(np.percentile(energy_db, noise_percentile))
    adaptive_threshold = noise_floor + threshold_above_noise_db
    return adaptive_threshold


def detect_speech_regions_energy(
    energy_db: np.ndarray,
    times: np.ndarray,
    threshold_db: Optional[float] = None,
    min_speech_sec: float = 0.1,
    min_silence_sec: float = 0.3,
    use_adaptive_threshold: bool = True,
    noise_percentile: float = 10.0,
    threshold_above_noise_db: float = 10.0,
) -> List[Tuple[float, float]]:
    """
    Detect speech regions using energy thresholding.

    Args:
        energy_db: energy envelope in dB
        times: time stamps for each energy frame
        threshold_db: fixed energy threshold (used if use_adaptive_threshold=False)
        min_speech_sec: minimum duration for a speech region
        min_silence_sec: minimum silence gap to split regions
        use_adaptive_threshold: if True, compute threshold relative to noise floor
        noise_percentile: percentile for noise floor estimation
        threshold_above_noise_db: dB above noise floor for adaptive threshold

    Returns:
        List of (start, end) time tuples for speech regions
    """
    if len(energy_db) == 0:
        return []

    # Determine threshold
    if use_adaptive_threshold:
        effective_threshold = estimate_adaptive_threshold(
            energy_db, noise_percentile, threshold_above_noise_db
        )
    else:
        effective_threshold = threshold_db if threshold_db is not None else -40.0

    # Binary speech/silence decision
    is_speech = energy_db >= effective_threshold

    # Find speech region boundaries
    regions: List[Tuple[float, float]] = []
    in_speech = False
    region_start = 0.0

    for t, speech in zip(times, is_speech):
        if speech and not in_speech:
            in_speech = True
            region_start = t
        elif not speech and in_speech:
            in_speech = False
            region_end = t
            if region_end - region_start >= min_speech_sec:
                regions.append((region_start, region_end))

    # Handle region at end
    if in_speech and len(times) > 0:
        region_end = times[-1]
        if region_end - region_start >= min_speech_sec:
            regions.append((region_start, region_end))

    # Merge regions separated by short silences
    if len(regions) < 2:
        return regions

    merged: List[Tuple[float, float]] = [regions[0]]
    for start, end in regions[1:]:
        prev_end = merged[-1][1]
        if start - prev_end < min_silence_sec:
            # Merge with previous
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    return merged


def is_time_in_speech_regions(
    t_start: float,
    t_end: float,
    speech_regions: List[Tuple[float, float]],
    min_overlap_ratio: float = 0.3,
) -> bool:
    """
    Check if a time interval overlaps with detected speech regions.

    Args:
        t_start, t_end: interval to check
        speech_regions: list of (start, end) speech regions
        min_overlap_ratio: minimum overlap ratio to count as confirmed

    Returns:
        True if interval overlaps sufficiently with speech
    """
    if not speech_regions:
        return False

    interval_dur = max(t_end - t_start, 0.001)
    total_overlap = 0.0

    for s_start, s_end in speech_regions:
        overlap_start = max(t_start, s_start)
        overlap_end = min(t_end, s_end)
        if overlap_end > overlap_start:
            total_overlap += overlap_end - overlap_start

    overlap_ratio = total_overlap / interval_dur
    return overlap_ratio >= min_overlap_ratio


# ---------------------------
# Confidence scoring
# ---------------------------
def compute_region_energy_confidence(
    energy_db: np.ndarray,
    times: np.ndarray,
    t_start: float,
    t_end: float,
    noise_floor_db: float = -50.0,
    speech_ceiling_db: float = -20.0,
    energy_percentile: float = 90.0,
) -> float:
    """
    Compute normalized energy confidence for a time region.
    Maps energy to 0-1 range where higher energy = higher confidence.

    Uses percentile energy (default 90th) instead of mean to avoid
    dilution by silence within the phrase region.

    Args:
        energy_db: energy envelope in dB
        times: timestamps
        t_start, t_end: region to analyze
        noise_floor_db: energy level considered silence (maps to 0)
        speech_ceiling_db: energy level considered loud speech (maps to 1)
        energy_percentile: percentile of energy to use (default 90th)

    Returns:
        Confidence score in [0, 1]
    """
    if len(energy_db) == 0:
        return 0.0

    # Find frames in the region
    mask = (times >= t_start) & (times <= t_end)
    if not mask.any():
        return 0.0

    region_energy = energy_db[mask]
    # Use percentile instead of mean - more stable when region contains silence
    peak_energy = float(np.percentile(region_energy, energy_percentile))

    # Normalize to 0-1
    if peak_energy <= noise_floor_db:
        return 0.0
    if peak_energy >= speech_ceiling_db:
        return 1.0

    confidence = (peak_energy - noise_floor_db) / (speech_ceiling_db - noise_floor_db)
    return float(np.clip(confidence, 0.0, 1.0))


def compute_combined_confidence(
    stt_conf: float,
    energy_conf: float,
    is_vad_confirmed: bool,
    stt_weight: float = 0.6,
    vad_bonus: float = 0.1,
) -> float:
    """
    Combine STT and acoustic confidence scores.

    Args:
        stt_conf: STT word probability (0-1)
        energy_conf: energy-based confidence (0-1)
        is_vad_confirmed: whether VAD detected speech in region
        stt_weight: weight for STT confidence (energy gets 1-stt_weight)
        vad_bonus: bonus added when VAD confirms speech

    Returns:
        Combined confidence in [0, 1]
    """
    energy_weight = 1.0 - stt_weight
    combined = stt_weight * stt_conf + energy_weight * energy_conf

    if is_vad_confirmed:
        combined = min(1.0, combined + vad_bonus)

    return float(np.clip(combined, 0.0, 1.0))


# ---------------------------
# Audio quality computation
# ---------------------------
def compute_audio_quality(
    audio: np.ndarray,
    duration_sec: float,
    speech_regions: List[Tuple[float, float]],
    energy_db: np.ndarray,
    energy_times: np.ndarray,
    words: List["Word"],
) -> AudioQualityReport:
    """
    Compute audio quality metrics for downstream QC.

    Args:
        audio: audio signal
        duration_sec: total audio duration
        speech_regions: detected speech regions
        energy_db: energy envelope
        energy_times: timestamps for energy frames (from compute_energy_envelope)
        words: transcribed words

    Returns:
        AudioQualityReport with quality metrics
    """
    # Speech ratio
    total_speech_dur = sum(end - start for start, end in speech_regions)
    speech_ratio = total_speech_dur / max(duration_sec, 0.001)

    # Mean energy
    mean_energy_db = float(np.mean(energy_db)) if len(energy_db) > 0 else -60.0

    # SNR estimate: speech energy vs non-speech energy
    if len(speech_regions) > 0 and len(energy_db) > 0 and len(energy_times) > 0:
        # Use actual times from energy envelope (not linspace)
        speech_mask = np.zeros(len(energy_db), dtype=bool)
        for s_start, s_end in speech_regions:
            speech_mask |= (energy_times >= s_start) & (energy_times <= s_end)

        if speech_mask.any() and (~speech_mask).any():
            speech_energy = np.mean(energy_db[speech_mask])
            noise_energy = np.mean(energy_db[~speech_mask])
            snr_estimate = speech_energy - noise_energy
        else:
            snr_estimate = 0.0
    else:
        snr_estimate = 0.0

    return AudioQualityReport(
        duration_sec=duration_sec,
        speech_ratio=float(speech_ratio),
        mean_energy_db=mean_energy_db,
        snr_estimate=float(snr_estimate),
        n_speech_segments=len(speech_regions),
        stt_word_count=len(words),
        has_audio=len(audio) > 0,
    )


# ---------------------------
# Summary creation
# ---------------------------
def create_prompt_summary(
    child_id: str,
    task_type: str,
    events: List[AudioEvent],
    quality: AudioQualityReport,
) -> ParentPromptSummary:
    """
    Aggregate events into per-video summary.

    Args:
        child_id: child identifier
        task_type: task type
        events: list of detected audio events
        quality: audio quality report

    Returns:
        ParentPromptSummary with aggregated metrics
    """
    if not events:
        return ParentPromptSummary(
            child_id=child_id,
            task_type=task_type,
            attempts_count=0,
            attempt_timestamps=[],
            prompt_confidence_mean=0.0,
            prompt_confidence_min=0.0,
            call_attention_count=0,
            look_count=0,
            audio_duration_sec=quality.duration_sec,
            speech_ratio=quality.speech_ratio,
            snr_estimate=quality.snr_estimate,
        )

    # Count by type
    call_attention_count = sum(1 for e in events if e.event_type == "CALL_ATTENTION")
    look_count = sum(1 for e in events if e.event_type == "LOOK")

    # Timestamps (sorted)
    attempt_timestamps = sorted([e.t_start for e in events])

    # Confidence stats
    confidences = [e.combined_confidence for e in events]
    conf_mean = float(np.mean(confidences))
    conf_min = float(np.min(confidences))

    return ParentPromptSummary(
        child_id=child_id,
        task_type=task_type,
        attempts_count=len(events),
        attempt_timestamps=attempt_timestamps,
        prompt_confidence_mean=conf_mean,
        prompt_confidence_min=conf_min,
        call_attention_count=call_attention_count,
        look_count=look_count,
        audio_duration_sec=quality.duration_sec,
        speech_ratio=quality.speech_ratio,
        snr_estimate=quality.snr_estimate,
    )


class AudioExtractionError(Exception):
    """Custom exception for audio extraction failures with specific cause."""
    def __init__(self, message: str, cause: str):
        super().__init__(message)
        self.cause = cause  # "no_ffmpeg", "no_audio_stream", "decode_error"


def _run_ffmpeg_extract_wav(video_path: Path, wav_path: Path) -> None:
    """
    Extract mono 16k wav for whisper.
    Requires: ffmpeg available in PATH.

    Raises:
        AudioExtractionError: with specific cause for different failure modes
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

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
    except FileNotFoundError:
        raise AudioExtractionError(
            "ffmpeg not found in PATH", cause="no_ffmpeg"
        )

    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="replace").lower()

        # Check for specific error patterns
        if "does not contain any stream" in stderr_text or "no audio stream" in stderr_text:
            raise AudioExtractionError(
                f"No audio stream in {video_path.name}", cause="no_audio_stream"
            )
        elif "invalid data found" in stderr_text or "error while decoding" in stderr_text:
            raise AudioExtractionError(
                f"Audio decode error in {video_path.name}", cause="decode_error"
            )
        elif "no such file" in stderr_text or "does not exist" in stderr_text:
            raise AudioExtractionError(
                f"Video file not found: {video_path}", cause="file_not_found"
            )
        else:
            # Generic ffmpeg error
            raise AudioExtractionError(
                f"ffmpeg failed for {video_path.name}: {stderr_text[:200]}", cause="ffmpeg_error"
            )


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


def merge_close_events(
    events: List[AudioEvent],
    merge_gap_sec: float = 0.7,
) -> List[AudioEvent]:
    """
    Merge events that are close together into a single "attempt".

    When multiple events occur within merge_gap_sec of each other,
    they are merged into one event spanning from the first t_start
    to the last t_end, with confidence scores combined.

    Args:
        events: list of AudioEvent sorted by t_start
        merge_gap_sec: max gap between events to merge (default 0.7s)

    Returns:
        Merged list of AudioEvent
    """
    if len(events) <= 1:
        return events

    # Ensure sorted by start time
    events = sorted(events, key=lambda e: e.t_start)

    merged: List[AudioEvent] = []
    current_group: List[AudioEvent] = [events[0]]

    for e in events[1:]:
        # Check if this event is close to the last in current group
        last_end = current_group[-1].t_end
        if e.t_start - last_end <= merge_gap_sec:
            # Add to current group
            current_group.append(e)
        else:
            # Finalize current group and start new one
            merged.append(_merge_event_group(current_group))
            current_group = [e]

    # Finalize last group
    if current_group:
        merged.append(_merge_event_group(current_group))

    return merged


def _merge_event_group(group: List[AudioEvent]) -> AudioEvent:
    """Merge a group of events into a single event."""
    if len(group) == 1:
        return group[0]

    # Use the most common event type, or first if tied
    type_counts: dict = {}
    for e in group:
        type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
    event_type = max(type_counts.keys(), key=lambda k: type_counts[k])

    # Span from first to last
    t_start = group[0].t_start
    t_end = group[-1].t_end

    # Combine phrases
    phrases = [e.matched_phrase for e in group]
    unique_phrases = list(dict.fromkeys(phrases))  # preserve order, remove dupes
    matched_phrase = " + ".join(unique_phrases)

    # Average confidences (or max - user preference)
    stt_confidence = float(np.mean([e.stt_confidence for e in group]))
    energy_confidence = float(np.mean([e.energy_confidence for e in group]))
    combined_confidence = float(np.mean([e.combined_confidence for e in group]))

    # VAD confirmed if any in group was confirmed
    is_vad_confirmed = any(e.is_vad_confirmed for e in group)

    return AudioEvent(
        event_type=event_type,
        t_start=t_start,
        t_end=t_end,
        matched_phrase=matched_phrase,
        stt_confidence=stt_confidence,
        energy_confidence=energy_confidence,
        combined_confidence=combined_confidence,
        is_vad_confirmed=is_vad_confirmed,
    )


def _load_wav_audio(wav_path: Path) -> Tuple[np.ndarray, int]:
    """Load WAV file and return (audio, sample_rate)."""
    sr, audio = wavfile.read(str(wav_path))
    # Convert to float32 normalized to [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128.0
    else:
        audio = audio.astype(np.float32)
    return audio, int(sr)


def process_video_audio(
    video_path: Path,
    child_id: str,
    task_type: str,
    model: WhisperModel,
    energy_threshold_db: Optional[float] = None,
    use_adaptive_threshold: bool = True,
    threshold_above_noise_db: float = 10.0,
    stt_weight: float = 0.6,
    min_stt_conf: float = 0.0,
    overlap_sec: float = 0.10,
) -> Tuple[List[AudioEvent], ParentPromptSummary, AudioQualityReport]:
    """
    Process a single video's audio track.

    Args:
        video_path: path to video file
        child_id: child identifier
        task_type: task type string
        model: WhisperModel instance
        energy_threshold_db: fixed dB threshold (used if use_adaptive_threshold=False)
        use_adaptive_threshold: if True, compute threshold relative to noise floor
        threshold_above_noise_db: dB above noise floor for adaptive threshold
        stt_weight: weight for STT confidence in combined score
        min_stt_conf: minimum STT confidence to keep event
        overlap_sec: overlap window for deduplication

    Returns:
        (events, summary, quality) tuple
    """
    empty_quality = AudioQualityReport(
        duration_sec=0.0,
        speech_ratio=0.0,
        mean_energy_db=-60.0,
        snr_estimate=0.0,
        n_speech_segments=0,
        stt_word_count=0,
        has_audio=False,
    )
    empty_summary = ParentPromptSummary(
        child_id=child_id,
        task_type=task_type,
        attempts_count=0,
    )

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "audio.wav"

        # Extract audio
        try:
            _run_ffmpeg_extract_wav(video_path, wav_path)
        except AudioExtractionError as e:
            # Log specific cause for debugging
            print(f"  [audio extraction] {e.cause}: {e}", flush=True)
            return [], empty_summary, empty_quality
        except Exception as e:
            print(f"  [audio extraction] unexpected error: {e}", flush=True)
            return [], empty_summary, empty_quality

        # Load audio for energy analysis
        try:
            audio, sr = _load_wav_audio(wav_path)
        except Exception:
            return [], empty_summary, empty_quality

        if len(audio) == 0:
            return [], empty_summary, empty_quality

        duration_sec = len(audio) / sr

        # Compute energy envelope and VAD
        times, energy_db = compute_energy_envelope(audio, sr=sr)
        speech_regions = detect_speech_regions_energy(
            energy_db,
            times,
            threshold_db=energy_threshold_db,
            use_adaptive_threshold=use_adaptive_threshold,
            threshold_above_noise_db=threshold_above_noise_db,
        )

        # Transcribe with Whisper
        try:
            words = transcribe_words(model, wav_path, language="en")
        except Exception:
            words = []

        # Compute audio quality
        quality = compute_audio_quality(
            audio=audio,
            duration_sec=duration_sec,
            speech_regions=speech_regions,
            energy_db=energy_db,
            energy_times=times,
            words=words,
        )

        if not words:
            summary = create_prompt_summary(child_id, task_type, [], quality)
            return [], summary, quality

        # Find phrase events (legacy format)
        call_events_raw = _find_phrase_events(
            words, ATTENTION_CALL_PHRASES, "CALL_ATTENTION", min_conf=min_stt_conf
        )
        look_events_raw = _find_phrase_events(
            words, LOOK_PHRASES, "LOOK", min_conf=min_stt_conf
        )

        # Deduplicate
        call_events_raw = dedupe_keep_longest(call_events_raw, overlap_sec=overlap_sec)
        look_events_raw = dedupe_keep_longest(look_events_raw, overlap_sec=overlap_sec)

        # Convert to AudioEvent with enhanced confidence
        all_events: List[AudioEvent] = []

        for etype, t_start, t_end, stt_conf, phrase in call_events_raw + look_events_raw:
            # Compute energy confidence for this region
            energy_conf = compute_region_energy_confidence(
                energy_db, times, t_start, t_end
            )

            # Check VAD confirmation
            is_vad_confirmed = is_time_in_speech_regions(
                t_start, t_end, speech_regions
            )

            # Compute combined confidence
            combined_conf = compute_combined_confidence(
                stt_conf, energy_conf, is_vad_confirmed, stt_weight=stt_weight
            )

            event = AudioEvent(
                event_type=etype,
                t_start=t_start,
                t_end=t_end,
                matched_phrase=phrase,
                stt_confidence=stt_conf,
                energy_confidence=energy_conf,
                combined_confidence=combined_conf,
                is_vad_confirmed=is_vad_confirmed,
            )
            all_events.append(event)

        # Sort by time
        all_events.sort(key=lambda e: e.t_start)

        # Create summary
        summary = create_prompt_summary(child_id, task_type, all_events, quality)

        return all_events, summary, quality


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract audio events (CALL_ATTENTION + LOOK) with timestamps and confidence scoring."
    )
    # Existing args
    ap.add_argument("--manifest", required=True, help="CSV manifest")
    ap.add_argument("--task", default="joint_attention", choices=list(TASK_TO_COL.keys()))
    ap.add_argument("--out", default="data/derived/audio_events.csv")
    ap.add_argument("--model", default="base", help="faster-whisper model: tiny/base/small/medium/large-v3")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--compute_type", default="int8", help="cpu best: int8")
    ap.add_argument("--limit", type=int, default=0, help="Process first N rows (0 = all)")
    ap.add_argument("--min_conf", type=float, default=0.0, help="Min STT word prob to keep event")
    ap.add_argument("--overlap_sec", type=float, default=0.10, help="Overlap window for dedupe (sec)")

    # New args for enhanced features
    ap.add_argument("--summary_out", default=None, help="Output CSV for per-video summaries")
    ap.add_argument("--quality_out", default=None, help="Output CSV for audio quality reports")
    ap.add_argument("--energy_threshold", type=float, default=None,
                    help="Fixed energy threshold (dB) for VAD (overrides adaptive)")
    ap.add_argument("--no_adaptive_threshold", action="store_true",
                    help="Disable adaptive threshold (use fixed --energy_threshold)")
    ap.add_argument("--threshold_above_noise", type=float, default=10.0,
                    help="dB above noise floor for adaptive threshold")
    ap.add_argument("--stt_weight", type=float, default=0.6,
                    help="Weight for STT confidence in combined score (0-1)")
    ap.add_argument("--min_combined_conf", type=float, default=0.0,
                    help="Filter events below this combined confidence")
    ap.add_argument("--merge_gap", type=float, default=0.0,
                    help="Merge events within this gap (sec) into single attempt. 0=disabled, typical=0.7")

    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    col = TASK_TO_COL[args.task]
    if col not in df.columns:
        raise ValueError(f"Manifest missing column '{col}'. Present: {list(df.columns)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize whisper model once
    print(f"Loading Whisper model: {args.model} ({args.device}, {args.compute_type})")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    # Enhanced fieldnames (backward compatible - 'confidence' now maps to combined_confidence)
    fieldnames = [
        "child_id", "task_type", "event_type", "t_start", "t_end",
        "confidence",  # combined_confidence for backward compat
        "matched_phrase",
        "stt_confidence", "energy_confidence", "is_vad_confirmed",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

    # Collect summaries and quality reports
    summaries: List[ParentPromptSummary] = []
    quality_reports: List[Tuple[str, str, AudioQualityReport]] = []

    n = len(df) if args.limit <= 0 else min(args.limit, len(df))
    print(f"Processing {n} videos for task={args.task}")

    for i in range(n):
        child_id = str(df.iloc[i]["child_id"])
        video_path = Path(str(df.iloc[i][col]))
        if not video_path.exists():
            print(f"[{i+1}/{n}] child={child_id} - video not found, skipping")
            continue

        print(f"[{i+1}/{n}] child={child_id} task={args.task}", flush=True)

        events, summary, quality = process_video_audio(
            video_path=video_path,
            child_id=child_id,
            task_type=args.task,
            model=model,
            energy_threshold_db=args.energy_threshold,
            use_adaptive_threshold=not args.no_adaptive_threshold,
            threshold_above_noise_db=args.threshold_above_noise,
            stt_weight=args.stt_weight,
            min_stt_conf=args.min_conf,
            overlap_sec=args.overlap_sec,
        )

        # Filter by combined confidence
        events = [e for e in events if e.combined_confidence >= args.min_combined_conf]

        # Merge close events into single attempts (if enabled)
        if args.merge_gap > 0:
            events = merge_close_events(events, merge_gap_sec=args.merge_gap)

        # Write events incrementally
        with open(out_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            for e in events:
                w.writerow({
                    "child_id": child_id,
                    "task_type": args.task,
                    "event_type": e.event_type,
                    "t_start": round(e.t_start, 3),
                    "t_end": round(e.t_end, 3),
                    "confidence": round(e.combined_confidence, 4),
                    "matched_phrase": e.matched_phrase,
                    "stt_confidence": round(e.stt_confidence, 4),
                    "energy_confidence": round(e.energy_confidence, 4),
                    "is_vad_confirmed": e.is_vad_confirmed,
                })

        # Update summary with filtered event counts (fix stale counts after filtering)
        summary.attempts_count = len(events)
        summary.attempt_timestamps = [e.t_start for e in events]
        summary.call_attention_count = sum(1 for e in events if e.event_type == "CALL_ATTENTION")
        summary.look_count = sum(1 for e in events if e.event_type == "LOOK")
        if events:
            summary.prompt_confidence_mean = float(np.mean([e.combined_confidence for e in events]))
            summary.prompt_confidence_min = float(np.min([e.combined_confidence for e in events]))
        summaries.append(summary)
        quality_reports.append((child_id, args.task, quality))

    print(f"Wrote events: {out_path}")

    # Write summary CSV
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_fieldnames = [
            "child_id", "task_type", "attempts_count", "attempt_timestamps",
            "prompt_confidence_mean", "prompt_confidence_min",
            "call_attention_count", "look_count",
            "audio_duration_sec", "speech_ratio", "snr_estimate",
        ]
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_fieldnames)
            w.writeheader()
            for s in summaries:
                w.writerow({
                    "child_id": s.child_id,
                    "task_type": s.task_type,
                    "attempts_count": s.attempts_count,
                    "attempt_timestamps": ";".join(f"{t:.3f}" for t in s.attempt_timestamps),
                    "prompt_confidence_mean": round(s.prompt_confidence_mean, 4),
                    "prompt_confidence_min": round(s.prompt_confidence_min, 4),
                    "call_attention_count": s.call_attention_count,
                    "look_count": s.look_count,
                    "audio_duration_sec": round(s.audio_duration_sec, 2),
                    "speech_ratio": round(s.speech_ratio, 4),
                    "snr_estimate": round(s.snr_estimate, 2),
                })
        print(f"Wrote summary: {summary_path}")

    # Write quality CSV
    if args.quality_out:
        quality_path = Path(args.quality_out)
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        quality_fieldnames = [
            "child_id", "task_type", "duration_sec", "speech_ratio",
            "mean_energy_db", "snr_estimate", "n_speech_segments",
            "stt_word_count", "has_audio",
        ]
        with open(quality_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=quality_fieldnames)
            w.writeheader()
            for child_id, task_type, q in quality_reports:
                w.writerow({
                    "child_id": child_id,
                    "task_type": task_type,
                    "duration_sec": round(q.duration_sec, 2),
                    "speech_ratio": round(q.speech_ratio, 4),
                    "mean_energy_db": round(q.mean_energy_db, 2),
                    "snr_estimate": round(q.snr_estimate, 2),
                    "n_speech_segments": q.n_speech_segments,
                    "stt_word_count": q.stt_word_count,
                    "has_audio": q.has_audio,
                })
        print(f"Wrote quality: {quality_path}")

    print("Done.")


if __name__ == "__main__":
    main()
