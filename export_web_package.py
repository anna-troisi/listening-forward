from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

import librosa
import numpy as np
import soundfile as sf

import generate_soundscape_v12 as base
import listening_forward_audio_v3 as audio_core


ORIENTATIONS = ["repair", "automation", "habitation", "biodiversity"]
WEB_SAMPLE_RATE = 32_000
CONTINUOUS_SECONDS = 48.0
EVENT_RICH_SECONDS = 16.0
DISCRETE_SECONDS = 18.0
EVENT_RICH_VARIANTS = 3
TOP_NEIGHBOURS = 8


def safe_name(value: str) -> str:
    allowed = []
    for char in value.lower():
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("-")
    text = "".join(allowed)
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-")


def role_for(function_name: str) -> str:
    function_name = str(function_name).strip().lower()
    if function_name in {"base", "texture", "foreground", "accent"}:
        return function_name
    if function_name == "transition":
        return "texture"
    return "texture"


def gentle_normalise(audio: np.ndarray) -> np.ndarray:
    if len(audio) == 0:
        return audio.astype(np.float32)

    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms > 0:
        target_rms = 0.065
        factor = max(0.65, min(2.0, target_rms / rms))
        audio = audio * factor

    peak = float(np.max(np.abs(audio)))
    if peak > 0.96:
        audio = audio / peak * 0.94

    return audio.astype(np.float32)


def fade_edges(audio: np.ndarray, fade_in_s: float, fade_out_s: float) -> np.ndarray:
    out = audio.copy()
    fade_in = min(len(out), int(fade_in_s * WEB_SAMPLE_RATE))
    fade_out = min(len(out), int(fade_out_s * WEB_SAMPLE_RATE))

    if fade_in > 0:
        out[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out > 0:
        out[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    return out


def crossfade_extend(audio: np.ndarray, target_samples: int, crossfade_seconds: float = 1.5) -> np.ndarray:
    if len(audio) == 0:
        return np.zeros(target_samples, dtype=np.float32)
    if len(audio) >= target_samples:
        return audio[:target_samples].astype(np.float32)

    crossfade = min(
        int(crossfade_seconds * WEB_SAMPLE_RATE),
        max(1, len(audio) // 4),
    )

    result = audio.astype(np.float32).copy()
    while len(result) < target_samples:
        overlap = min(crossfade, len(result), len(audio))
        if overlap <= 1:
            result = np.concatenate([result, audio])
            continue

        theta = np.linspace(0.0, np.pi / 2.0, overlap, dtype=np.float32)
        blend = result[-overlap:] * np.cos(theta) + audio[:overlap] * np.sin(theta)
        result = np.concatenate([result[:-overlap], blend, audio[overlap:]])

    return result[:target_samples].astype(np.float32)


def resample_for_web(audio: np.ndarray, source_sr: int) -> np.ndarray:
    if source_sr == WEB_SAMPLE_RATE:
        return audio.astype(np.float32)
    return librosa.resample(
        audio.astype(np.float32),
        orig_sr=source_sr,
        target_sr=WEB_SAMPLE_RATE,
    ).astype(np.float32)


def audible_audio(path: Path) -> np.ndarray:
    audio = audio_core.load_audio(path)
    start = audio_core.find_audible_start_sample(audio, audio_core.TARGET_SR)
    if start < len(audio):
        audio = audio[start:]
    return resample_for_web(audio, audio_core.TARGET_SR)


def evenly_spaced_starts(length: int, clip_samples: int, variants: int) -> List[int]:
    if length <= clip_samples:
        return [0]
    maximum = length - clip_samples
    if variants <= 1:
        return [maximum // 2]
    return [int(round(value)) for value in np.linspace(0, maximum, variants)]


def export_audio_variants(
    row,
    source_path: Path,
    audio_dir: Path,
) -> Tuple[List[Dict], str]:
    audio = audible_audio(source_path)
    elements = set(row["elements_set"])
    character = audio_core.resolve_playback_character(
        row.get("playback_character_norm", ""),
        str(row["function_norm"]),
        elements,
        float(row.get("duration_sec", 0)),
    )

    file_id = safe_name(str(row["filekey"]))
    files: List[Dict] = []

    if character == "continuous":
        target = int(CONTINUOUS_SECONDS * WEB_SAMPLE_RATE)
        clip = crossfade_extend(audio, target)
        clip = gentle_normalise(clip)
        output = audio_dir / f"{file_id}--continuous.wav"
        sf.write(output, clip, WEB_SAMPLE_RATE, subtype="PCM_16")
        files.append({
            "path": f"audio/{output.name}",
            "variant": 0,
            "duration": round(len(clip) / WEB_SAMPLE_RATE, 3),
        })

    elif character == "event_rich":
        target = int(EVENT_RICH_SECONDS * WEB_SAMPLE_RATE)
        starts = evenly_spaced_starts(len(audio), target, EVENT_RICH_VARIANTS)
        for variant, start in enumerate(starts):
            clip = audio[start:start + target]
            if len(clip) < target:
                padded = np.zeros(target, dtype=np.float32)
                padded[:len(clip)] = clip
                clip = padded
            clip = fade_edges(clip, 2.0, 2.0)
            clip = gentle_normalise(clip)
            output = audio_dir / f"{file_id}--event-{variant + 1}.wav"
            sf.write(output, clip, WEB_SAMPLE_RATE, subtype="PCM_16")
            files.append({
                "path": f"audio/{output.name}",
                "variant": variant,
                "duration": round(len(clip) / WEB_SAMPLE_RATE, 3),
            })

    else:
        target = min(len(audio), int(DISCRETE_SECONDS * WEB_SAMPLE_RATE))
        clip = audio[:target]
        clip = fade_edges(clip, 0.5, 1.2)
        clip = gentle_normalise(clip)
        output = audio_dir / f"{file_id}--discrete.wav"
        sf.write(output, clip, WEB_SAMPLE_RATE, subtype="PCM_16")
        files.append({
            "path": f"audio/{output.name}",
            "variant": 0,
            "duration": round(len(clip) / WEB_SAMPLE_RATE, 3),
        })

    return files, character


def normalise_scenario_scores(raw_scores: Dict[str, List[float]]) -> Dict[str, List[float]]:
    normalised: Dict[str, List[float]] = {}
    for orientation, values in raw_scores.items():
        array = np.asarray(values, dtype=np.float32)
        minimum = float(np.min(array))
        maximum = float(np.max(array))
        if math.isclose(minimum, maximum):
            normalised[orientation] = [0.5 for _ in values]
        else:
            normalised[orientation] = [
                round(float((value - minimum) / (maximum - minimum)), 6)
                for value in array
            ]
    return normalised


def export_package(
    metadata_file: Path,
    embeddings_file: Path,
    sounds_dir: Path,
    docs_dir: Path,
    clean: bool,
) -> None:
    bundle = base.load_data(metadata_file, embeddings_file)
    dataframe = bundle.merged.copy().reset_index(drop=True)
    embedding_columns = base.get_embedding_columns(dataframe)
    audio_index = base.build_audio_index(sounds_dir)

    audio_dir = docs_dir / "audio"
    data_dir = docs_dir / "data"

    if clean:
        shutil.rmtree(audio_dir, ignore_errors=True)
        shutil.rmtree(data_dir, ignore_errors=True)

    audio_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_scores: Dict[str, List[float]] = {name: [] for name in ORIENTATIONS}
    for _, row in dataframe.iterrows():
        for orientation in ORIENTATIONS:
            raw_scores[orientation].append(base.row_score(row, orientation))
    scores = normalise_scenario_scores(raw_scores)

    vectors = np.vstack([
        base.get_embedding_vector(row, embedding_columns)
        for _, row in dataframe.iterrows()
    ])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_vectors = vectors / norms
    similarity_matrix = unit_vectors @ unit_vectors.T

    sounds: List[Dict] = []
    skipped: List[str] = []

    for index, row in dataframe.iterrows():
        source_path = audio_index.get(row["filekey"])
        if source_path is None:
            skipped.append(str(row["filename_original"]))
            continue

        files, character = export_audio_variants(row, source_path, audio_dir)

        similarities = similarity_matrix[index]
        neighbour_indices = np.argsort(similarities)[::-1]
        neighbours = []
        for neighbour_index in neighbour_indices:
            if int(neighbour_index) == index:
                continue
            neighbours.append({
                "id": str(dataframe.iloc[int(neighbour_index)]["filekey"]),
                "similarity": round(float(similarities[int(neighbour_index)]), 6),
            })
            if len(neighbours) >= TOP_NEIGHBOURS:
                break

        elements = sorted(str(value) for value in row["elements_set"])
        sound = {
            "id": str(row["filekey"]),
            "label": str(row["filename_original"]),
            "role": role_for(str(row["function_norm"])),
            "playbackCharacter": character,
            "category": str(row["category_norm"]),
            "environment": str(row["environment_norm"]),
            "elements": elements,
            "humanPresence": str(row["human_presence_norm"]),
            "durationSeconds": int(row.get("duration_sec", 0)),
            "scores": {
                orientation: scores[orientation][index]
                for orientation in ORIENTATIONS
            },
            "files": files,
            "neighbours": neighbours,
        }
        sounds.append(sound)
        print(f"Exported {sound['id']} ({character}, {len(files)} file(s))")

    manifest = {
        "version": 1,
        "sampleRate": WEB_SAMPLE_RATE,
        "orientations": ORIENTATIONS,
        "soundCount": len(sounds),
        "sounds": sounds,
    }

    manifest_path = data_dir / "sounds.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nWeb package export complete.")
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Audio folder: {audio_dir.resolve()}")
    print(f"Exported sounds: {len(sounds)}")
    if skipped:
        print("Missing source audio files:")
        for name in skipped:
            print(f"  - {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export browser-ready WAV excerpts and a compact Listening Forward "
            "manifest for the GitHub Pages website."
        )
    )
    parser.add_argument("--metadata", type=Path, default=Path("Soundscape_Metadata2.xlsx"))
    parser.add_argument("--embeddings", type=Path, default=Path("sound_embeddings.csv"))
    parser.add_argument("--sounds-dir", type=Path, default=Path("sounds"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the existing docs/audio and docs/data folders before exporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_package(
        metadata_file=args.metadata,
        embeddings_file=args.embeddings,
        sounds_dir=args.sounds_dir,
        docs_dir=args.docs_dir,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
