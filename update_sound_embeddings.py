#!/usr/bin/env python3
"""Audit, verify, and incrementally update OpenL3 audio embeddings.

This script reproduces the extraction settings previously used for the
Listening Forward / Speculative Soundscapes dataset:

- stereo/multichannel audio averaged to mono
- resampled to 48 kHz
- first 60 seconds used
- OpenL3 content_type="env"
- input_repr="mel256"
- embedding_size=512
- default 0.1-second hop
- temporal embeddings averaged into one vector per recording

Recommended workflow:

1. Verify that the current environment reproduces existing vectors:
   python3 update_sound_embeddings.py \
       --metadata Soundscape_Metadata2.xlsx \
       --embeddings sound_embeddings.csv \
       --sounds-dir sounds \
       --verify-existing 3

2. If the cosine similarities are close to 1.0, update the CSV:
   python3 update_sound_embeddings.py \
       --metadata Soundscape_Metadata2.xlsx \
       --embeddings sound_embeddings.csv \
       --sounds-dir sounds \
       --update

The update operation makes a timestamped backup, removes duplicate/stale
embedding rows not represented in the metadata, calculates only missing
embeddings, and writes the final CSV in metadata order. Audio files are never
deleted or moved.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


TARGET_SR = 48_000
MAX_DURATION_SECONDS = 60
CONTENT_TYPE = "env"
INPUT_REPR = "mel256"
EMBEDDING_SIZE = 512
HOP_SIZE = 0.1
FRONTEND = "kapre"
BATCH_SIZE = 32


def canonical_filekey(value: object) -> str:
    """Return a stable key from a filename or path."""
    if pd.isna(value):
        return ""
    stem = Path(str(value).strip()).stem.lower()
    return re.sub(r"\s*\(\d+\)$", "", stem)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    metadata = pd.read_excel(path)
    if "filename" not in metadata.columns:
        raise ValueError("The metadata workbook must contain a 'filename' column.")

    metadata = metadata.copy()
    metadata["filekey"] = metadata["filename"].apply(canonical_filekey)
    metadata = metadata[metadata["filekey"] != ""].copy()

    duplicate_keys = (
        metadata.loc[metadata["filekey"].duplicated(keep=False), "filekey"]
        .drop_duplicates()
        .tolist()
    )
    if duplicate_keys:
        formatted = "\n".join(f"  - {key}" for key in duplicate_keys)
        raise ValueError(
            "Duplicate canonical filenames remain in the metadata:\n"
            f"{formatted}\n"
            "Correct these before updating embeddings."
        )

    return metadata


def load_embeddings(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings CSV not found: {path}")

    embeddings = pd.read_csv(path)
    required = {"filepath", "filename", "folder"}
    missing = required.difference(embeddings.columns)
    if missing:
        raise ValueError(
            "Embeddings CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    expected_vector_columns = [str(i) for i in range(EMBEDDING_SIZE)]
    absent = [col for col in expected_vector_columns if col not in embeddings.columns]
    if absent:
        raise ValueError(
            f"Expected {EMBEDDING_SIZE} embedding columns named 0–511; "
            f"{len(absent)} are missing."
        )

    embeddings = embeddings.copy()
    embeddings["filekey"] = embeddings["filename"].apply(canonical_filekey)
    return embeddings


def build_audio_index(sounds_dir: Path) -> Dict[str, List[Path]]:
    if not sounds_dir.exists():
        raise FileNotFoundError(f"Sounds directory not found: {sounds_dir}")

    index: Dict[str, List[Path]] = {}
    for path in sounds_dir.rglob("*"):
        if not path.is_file():
            continue
        key = canonical_filekey(path.name)
        if key:
            index.setdefault(key, []).append(path)
    return index


def choose_audio_path(
    metadata_filename: str,
    key: str,
    audio_index: Dict[str, List[Path]],
) -> Path:
    matches = audio_index.get(key, [])
    if not matches:
        raise FileNotFoundError(
            f"No audio file found under the sounds directory for: {metadata_filename}"
        )

    if len(matches) == 1:
        return matches[0]

    # Prefer an exact basename match before declaring ambiguity.
    requested_name = Path(str(metadata_filename)).name.lower()
    exact = [path for path in matches if path.name.lower() == requested_name]
    if len(exact) == 1:
        return exact[0]

    listed = "\n".join(f"    - {path}" for path in matches)
    raise RuntimeError(
        f"Multiple audio files match '{metadata_filename}':\n{listed}\n"
        "Keep one canonical copy or make the metadata filename exact."
    )


def import_openl3():
    try:
        import openl3  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "OpenL3 could not be imported in this Python environment.\n"
            "Activate the environment previously used for the project, then retry.\n"
            f"Original import error: {exc}"
        ) from exc
    return openl3


def load_openl3_model(openl3):
    print(
        "Loading OpenL3 model "
        f"(content_type={CONTENT_TYPE}, input_repr={INPUT_REPR}, "
        f"embedding_size={EMBEDDING_SIZE})..."
    )
    return openl3.models.load_audio_embedding_model(
        input_repr=INPUT_REPR,
        content_type=CONTENT_TYPE,
        embedding_size=EMBEDDING_SIZE,
        frontend=FRONTEND,
    )


def load_and_prepare_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = np.asarray(audio, dtype=np.float32)

    if sample_rate != TARGET_SR:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=TARGET_SR,
        ).astype(np.float32)

    max_samples = TARGET_SR * MAX_DURATION_SECONDS
    return audio[:max_samples]


def extract_mean_embedding(path: Path, openl3, model) -> np.ndarray:
    audio = load_and_prepare_audio(path)
    if audio.size < TARGET_SR:
        raise ValueError(
            f"Audio is shorter than one second and cannot be embedded safely: {path}"
        )

    temporal_embeddings, _timestamps = openl3.get_audio_embedding(
        audio,
        TARGET_SR,
        model=model,
        center=True,
        hop_size=HOP_SIZE,
        batch_size=BATCH_SIZE,
        frontend=FRONTEND,
        verbose=False,
    )

    temporal_embeddings = np.asarray(temporal_embeddings, dtype=np.float32)
    if temporal_embeddings.ndim != 2 or temporal_embeddings.shape[1] != EMBEDDING_SIZE:
        raise ValueError(
            f"Unexpected OpenL3 output shape {temporal_embeddings.shape} for {path}"
        )

    return temporal_embeddings.mean(axis=0)


def cleaned_embedding_rows(
    metadata: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Keep one row per metadata key and remove stale/unmatched rows."""
    metadata_keys = set(metadata["filekey"])

    duplicate_keys = (
        embeddings.loc[embeddings["filekey"].duplicated(keep=False), "filekey"]
        .drop_duplicates()
        .tolist()
    )
    stale_keys = sorted(set(embeddings["filekey"]) - metadata_keys)

    clean = embeddings[embeddings["filekey"].isin(metadata_keys)].copy()
    clean = clean.drop_duplicates(subset="filekey", keep="first")
    return clean, duplicate_keys, stale_keys


def print_audit(
    metadata: pd.DataFrame,
    embeddings: pd.DataFrame,
    audio_index: Dict[str, List[Path]],
) -> None:
    clean, duplicate_keys, stale_keys = cleaned_embedding_rows(metadata, embeddings)

    metadata_keys = set(metadata["filekey"])
    clean_keys = set(clean["filekey"])
    missing_embedding_keys = [
        key for key in metadata["filekey"].tolist() if key not in clean_keys
    ]
    missing_audio_keys = [
        key for key in metadata["filekey"].tolist() if key not in audio_index
    ]

    filename_by_key = dict(zip(metadata["filekey"], metadata["filename"]))

    print(f"Metadata rows: {len(metadata)}")
    print(f"Embedding rows before synchronization: {len(embeddings)}")
    print(f"Embedding rows after synchronization: {len(clean)}")
    print(f"Metadata recordings missing embeddings: {len(missing_embedding_keys)}")

    if duplicate_keys:
        print("\nDuplicate embedding keys that will be reduced to one row:")
        for key in duplicate_keys:
            print(f"  - {key}")

    if stale_keys:
        print("\nEmbedding keys absent from metadata that will be removed from the CSV:")
        for key in stale_keys:
            print(f"  - {key}")

    if missing_embedding_keys:
        print("\nRecordings requiring new embeddings:")
        for key in missing_embedding_keys:
            print(f"  - {filename_by_key[key]}")

    if missing_audio_keys:
        print("\nMetadata rows with no matching physical audio file:")
        for key in missing_audio_keys:
            print(f"  - {filename_by_key[key]}")


def verify_existing(
    metadata: pd.DataFrame,
    embeddings: pd.DataFrame,
    audio_index: Dict[str, List[Path]],
    count: int,
) -> None:
    clean, _duplicates, _stale = cleaned_embedding_rows(metadata, embeddings)
    vector_columns = [str(i) for i in range(EMBEDDING_SIZE)]
    clean_by_key = clean.set_index("filekey", drop=False)

    candidates: List[Tuple[str, str, Path]] = []
    for _, row in metadata.iterrows():
        key = row["filekey"]
        if key not in clean_by_key.index or key not in audio_index:
            continue
        try:
            path = choose_audio_path(str(row["filename"]), key, audio_index)
        except RuntimeError:
            continue
        candidates.append((key, str(row["filename"]), path))

    if not candidates:
        raise RuntimeError(
            "No uniquely matched existing audio/embedding pair is available for verification."
        )

    openl3 = import_openl3()
    model = load_openl3_model(openl3)

    print(
        "\nVerifying extraction compatibility against existing embeddings.\n"
        "Cosine similarity should be extremely close to 1.0.\n"
    )

    similarities: List[float] = []
    for key, metadata_filename, path in candidates[:count]:
        recomputed = extract_mean_embedding(path, openl3, model)
        stored = (
            clean_by_key.loc[key, vector_columns]
            .to_numpy(dtype=np.float32)
            .reshape(-1)
        )
        similarity = cosine_similarity(recomputed, stored)
        mean_abs_error = float(np.mean(np.abs(recomputed - stored)))
        similarities.append(similarity)

        print(f"{metadata_filename}")
        print(f"  audio: {path}")
        print(f"  cosine similarity: {similarity:.8f}")
        print(f"  mean absolute difference: {mean_abs_error:.8f}")

    minimum = min(similarities)
    print(f"\nLowest verification similarity: {minimum:.8f}")
    if minimum < 0.995:
        print(
            "\nSTOP: the current environment does not reproduce the stored "
            "embedding space closely enough. Do not run --update yet."
        )
        sys.exit(2)

    print(
        "\nVerification passed. You can now run the same command with --update."
    )


def update_embeddings(
    metadata_path: Path,
    embeddings_path: Path,
    sounds_dir: Path,
    metadata: pd.DataFrame,
    embeddings: pd.DataFrame,
    audio_index: Dict[str, List[Path]],
) -> None:
    clean, duplicate_keys, stale_keys = cleaned_embedding_rows(metadata, embeddings)
    vector_columns = [str(i) for i in range(EMBEDDING_SIZE)]
    clean_by_key = clean.set_index("filekey", drop=False)

    missing_rows = [
        row for _, row in metadata.iterrows()
        if row["filekey"] not in clean_by_key.index
    ]

    # Resolve every missing audio file before doing expensive computation.
    resolved: List[Tuple[pd.Series, Path]] = []
    for row in missing_rows:
        path = choose_audio_path(
            str(row["filename"]),
            str(row["filekey"]),
            audio_index,
        )
        resolved.append((row, path))

    openl3 = None
    model = None
    if resolved:
        openl3 = import_openl3()
        model = load_openl3_model(openl3)

    new_rows: Dict[str, Dict[str, object]] = {}
    for index, (metadata_row, path) in enumerate(resolved, start=1):
        filename = str(metadata_row["filename"])
        print(f"[{index}/{len(resolved)}] Embedding {filename}")
        vector = extract_mean_embedding(path, openl3, model)

        try:
            relative_path = path.resolve().relative_to(Path.cwd().resolve())
            filepath_text = relative_path.as_posix()
        except ValueError:
            filepath_text = path.as_posix()

        row_data: Dict[str, object] = {
            "filepath": filepath_text,
            "filename": path.name,
            "folder": path.parent.name,
        }
        row_data.update({
            str(i): float(vector[i]) for i in range(EMBEDDING_SIZE)
        })
        new_rows[str(metadata_row["filekey"])] = row_data

    # Rebuild rows in metadata order, using cleaned old rows or new vectors.
    clean_lookup = {
        str(row["filekey"]): row
        for _, row in clean.iterrows()
    }

    output_rows: List[Dict[str, object]] = []
    output_columns = ["filepath", "filename", "folder"] + vector_columns

    for _, metadata_row in metadata.iterrows():
        key = str(metadata_row["filekey"])
        if key in new_rows:
            output_rows.append(new_rows[key])
            continue

        old_row = clean_lookup.get(key)
        if old_row is None:
            raise RuntimeError(f"Internal error: no embedding available for {key}")

        output_rows.append({
            column: old_row[column] for column in output_columns
        })

    output = pd.DataFrame(output_rows, columns=output_columns)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = embeddings_path.with_name(
        f"{embeddings_path.stem}_before_update_{timestamp}{embeddings_path.suffix}"
    )
    embeddings_path.replace(backup_path)

    temp_path = embeddings_path.with_name(
        f"{embeddings_path.stem}.updating.tmp{embeddings_path.suffix}"
    )
    output.to_csv(temp_path, index=False)
    temp_path.replace(embeddings_path)

    print("\nEmbedding update completed.")
    print(f"Backup: {backup_path}")
    print(f"Updated CSV: {embeddings_path}")
    print(f"Final embedding rows: {len(output)}")
    print(f"New embeddings added: {len(new_rows)}")
    print(f"Duplicate keys removed: {len(duplicate_keys)}")
    print(f"Stale keys removed: {len(stale_keys)}")
    print("No audio files were deleted or moved.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit, verify, and update OpenL3 embeddings incrementally."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("Soundscape_Metadata2.xlsx"),
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("sound_embeddings.csv"),
    )
    parser.add_argument(
        "--sounds-dir",
        type=Path,
        default=Path("sounds"),
    )

    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--verify-existing",
        type=int,
        metavar="N",
        help="Recompute N existing recordings and compare them with stored vectors.",
    )
    action.add_argument(
        "--update",
        action="store_true",
        help="Synchronize the CSV to metadata and calculate missing embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata = load_metadata(args.metadata)
    embeddings = load_embeddings(args.embeddings)
    audio_index = build_audio_index(args.sounds_dir)

    print_audit(metadata, embeddings, audio_index)

    if args.verify_existing:
        verify_existing(
            metadata,
            embeddings,
            audio_index,
            max(1, args.verify_existing),
        )
    elif args.update:
        update_embeddings(
            args.metadata,
            args.embeddings,
            args.sounds_dir,
            metadata,
            embeddings,
            audio_index,
        )
    else:
        print(
            "\nAudit only. Use --verify-existing 3 before running --update."
        )


if __name__ == "__main__":
    main()
