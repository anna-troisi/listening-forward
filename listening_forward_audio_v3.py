from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Tuple
import hashlib
import random

import librosa
import numpy as np
import soundfile as sf


TARGET_SR = 48_000
AUDIBLE_WINDOW_SECONDS = 0.10
AUDIBLE_LOOKAHEAD_SECONDS = 12.0
INITIAL_AUDIBILITY_THRESHOLD = 1e-4
AUDIBLE_RELATIVE_THRESHOLD = 0.12

VALID_PLAYBACK_CHARACTERS = {"continuous", "event_rich", "discrete_event"}

EVENT_LIKE_TERMS: Set[str] = {
    "horn", "clacson", "car horn", "beep", "beeps", "bell", "bells",
    "impact", "impacts", "saw", "helicopter", "shout", "shouts",
    "alarm", "alarms", "knock", "knocks", "monkey", "monkeys"
}


def deterministic_rng(session_seed: int, filekey: str, state_index: int = 0) -> random.Random:
    payload = f"{session_seed}:{filekey}:{state_index}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return random.Random(int(digest[:16], 16))


def normalise_playback_character(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    return text if text in VALID_PLAYBACK_CHARACTERS else ""


def infer_playback_character(
    function_name: str,
    elements: Set[str],
    duration_seconds: float,
) -> str:
    function_name = function_name.strip().lower()

    if elements & EVENT_LIKE_TERMS:
        return "discrete_event"

    if function_name in {"foreground", "accent"} or duration_seconds < 25.0:
        return "discrete_event"

    if function_name == "texture" and duration_seconds >= 45.0:
        return "continuous"

    return "event_rich"


def resolve_playback_character(
    explicit_value: object,
    function_name: str,
    elements: Set[str],
    duration_seconds: float,
) -> str:
    explicit = normalise_playback_character(explicit_value)
    if explicit:
        return explicit
    return infer_playback_character(function_name, elements, duration_seconds)


def load_audio(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    audio, sample_rate = sf.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sample_rate != target_sr:
        audio = librosa.resample(
            audio.astype(np.float32),
            orig_sr=sample_rate,
            target_sr=target_sr,
        )

    return audio.astype(np.float32)


def find_audible_start_sample(
    audio: np.ndarray,
    sample_rate: int = TARGET_SR,
    max_lookahead_seconds: float = AUDIBLE_LOOKAHEAD_SECONDS,
) -> int:
    if len(audio) == 0:
        return 0

    window = max(1, int(AUDIBLE_WINDOW_SECONDS * sample_rate))
    lookahead = min(len(audio), int(max_lookahead_seconds * sample_rate))
    section = audio[:lookahead]

    if len(section) < window:
        return 0

    usable = len(section) - (len(section) % window)
    if usable <= 0:
        return 0

    frames = section[:usable].reshape(-1, window)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    positive = rms[rms > 0]

    if positive.size == 0:
        return 0

    reference = float(np.percentile(positive, 90))
    threshold = max(
        INITIAL_AUDIBILITY_THRESHOLD,
        reference * AUDIBLE_RELATIVE_THRESHOLD,
    )

    audible_frames = np.flatnonzero(rms >= threshold)
    if audible_frames.size == 0:
        return 0

    frame_index = max(0, int(audible_frames[0]) - 1)
    return frame_index * window


def prepare_clip(
    audio: np.ndarray,
    target_samples: int,
    rng: random.Random,
    *,
    function_name: str,
    elements: Set[str],
    playback_character: object = "",
    prefer_audible_opening: bool = False,
) -> Tuple[np.ndarray, int, str, str]:
    """
    Prepare a recording according to its explicit playback character.

    Returns:
        clip
        source_start_sample
        playback_mode: crop | loop | one_shot | silent
        resolved_character
    """
    if len(audio) == 0:
        return np.zeros(target_samples, dtype=np.float32), 0, "silent", "continuous"

    audible_start = find_audible_start_sample(audio, TARGET_SR)
    usable_audio = audio[audible_start:]

    if len(usable_audio) == 0:
        usable_audio = audio
        audible_start = 0

    duration_seconds = len(usable_audio) / TARGET_SR
    character = resolve_playback_character(
        playback_character,
        function_name,
        elements,
        duration_seconds,
    )

    if len(usable_audio) >= target_samples:
        if prefer_audible_opening:
            start = audible_start
        else:
            max_start = len(usable_audio) - target_samples
            relative_start = rng.randint(0, max_start) if max_start > 0 else 0
            start = audible_start + relative_start

        return (
            audio[start:start + target_samples].astype(np.float32),
            start,
            "crop",
            character,
        )

    if character in {"event_rich", "discrete_event"}:
        layer = np.zeros(target_samples, dtype=np.float32)
        max_offset = max(0, target_samples - len(usable_audio))
        offset = 0 if prefer_audible_opening else (
            rng.randint(0, max_offset) if max_offset > 0 else 0
        )
        usable = min(len(usable_audio), target_samples - offset)
        layer[offset:offset + usable] = usable_audio[:usable]
        return layer, audible_start, "one_shot", character

    repeats = int(np.ceil(target_samples / max(1, len(usable_audio))))
    tiled = np.tile(usable_audio, repeats)[:target_samples]
    return tiled.astype(np.float32), audible_start, "loop", character



def smooth_activity_edges(
    audio: np.ndarray,
    sample_rate: int = TARGET_SR,
    *,
    fade_in_s: float = 2.0,
    fade_out_s: float = 2.0,
    threshold: float = 1e-5,
) -> np.ndarray:
    """
    Smooth the beginning and end of the audible region.

    This is especially important for one-shot and event-rich clips placed
    inside a longer state: their internal onset and release receive fades,
    preventing sudden water, traffic, voice, or mechanical entries.
    """
    output = audio.copy()
    audible = np.flatnonzero(np.abs(output) >= threshold)

    if audible.size == 0:
        return output

    start = int(audible[0])
    end = int(audible[-1]) + 1

    fade_in_len = min(end - start, int(fade_in_s * sample_rate))
    fade_out_len = min(end - start, int(fade_out_s * sample_rate))

    if fade_in_len > 0:
        output[start:start + fade_in_len] *= np.linspace(0.0, 1.0, fade_in_len)

    if fade_out_len > 0:
        output[end - fade_out_len:end] *= np.linspace(1.0, 0.0, fade_out_len)

    return output

def apply_fade(
    audio: np.ndarray,
    sample_rate: int,
    fade_in_s: float,
    fade_out_s: float,
) -> np.ndarray:
    output = audio.copy()
    fade_in_len = min(len(output), int(fade_in_s * sample_rate))
    fade_out_len = min(len(output), int(fade_out_s * sample_rate))

    if fade_in_len:
        output[:fade_in_len] *= np.linspace(0.0, 1.0, fade_in_len)

    if fade_out_len:
        output[-fade_out_len:] *= np.linspace(1.0, 0.0, fade_out_len)

    return output
