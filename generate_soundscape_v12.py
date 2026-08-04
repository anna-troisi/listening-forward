from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import argparse
import csv
import random
import re

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import listening_forward_audio_v3 as audio_core


DEFAULT_METADATA_FILE = Path("Soundscape_Metadata2.xlsx")
DEFAULT_EMBEDDINGS_FILE = Path("sound_embeddings.csv")
DEFAULT_SOUNDS_DIR = Path("sounds")
DEFAULT_OUTPUT_DIR = Path("generated")

TARGET_SR = audio_core.TARGET_SR
TARGET_DURATION = 60
MAX_INITIAL_SILENCE_SECONDS = 1.5
INITIAL_AUDIBILITY_THRESHOLD = 1e-4
AUDIBLE_WINDOW_SECONDS = 0.10
AUDIBLE_LOOKAHEAD_SECONDS = 12.0
AUDIBLE_RELATIVE_THRESHOLD = 0.12


SCENARIO_PROFILES: Dict[str, Dict] = {
    "repair": {
        "preferred_moods": {"calm", "open", "fragile", "inhabited"},
        "preferred_environments": {"coastal", "natural", "birds"},
        "preferred_categories": {"birds", "water", "coastal", "wind"},
        "avoided_categories": {"machines"},
        "preferred_functions": ["base", "texture", "foreground", "accent"],
    },
    "erosion": {
        "preferred_moods": {"exposed", "unstable", "tense", "open"},
        "preferred_environments": {"coastal", "infrastructural"},
        "preferred_categories": {"wind", "coastal", "water"},
        "avoided_categories": {"urban", "machines"},
        "preferred_functions": ["base", "texture", "foreground", "accent"],
    },
    "automation": {
        "preferred_moods": {"tense", "inhabited", "pressure"},
        "preferred_environments": {"urban", "infrastructural"},
        "preferred_categories": {"machines", "urban"},
        "avoided_categories": set(),
        "preferred_functions": ["base", "texture", "foreground", "accent"],
    },
    "habitation": {
        "preferred_moods": {"inhabited", "open", "calm", "liminal"},
        "preferred_environments": {"urban", "infrastructural"},
        "preferred_categories": {"urban", "human", "people", "machines"},
        "preferred_elements": {
            "people", "person", "voices", "voice", "children", "kids",
            "footsteps", "walking", "walkers", "boat", "boats",
            "engine", "engines", "dogs", "dog", "tools", "machinery",
            "helicopter", "cars", "car", "work", "workers", "swimming"
        },
        "preferred_human_presence": {"direct", "indirect"},
        "downweighted_elements_when_alone": {"water", "waves", "rain", "wind"},
        "avoided_categories": set(),
        "preferred_functions": ["base", "foreground", "texture", "accent"],
    },
    "biodiversity": {
        "preferred_moods": {"open", "calm", "fragile"},
        "preferred_environments": {"coastal", "natural", "birds"},
        "preferred_categories": {"birds", "water", "wind"},
        "avoided_categories": {"machines"},
        "preferred_functions": ["base", "texture", "foreground", "accent"],
    },
}


@dataclass
class DatasetBundle:
    metadata: pd.DataFrame
    embeddings: pd.DataFrame
    merged: pd.DataFrame
    missing_embeddings: List[str]
    embeddings_without_metadata: List[str]
    duplicate_metadata_keys: List[str]
    duplicate_embedding_keys: List[str]


def canonical_filekey(name: object) -> str:
    if pd.isna(name):
        return ""
    stem = Path(str(name)).stem.strip().lower()
    return re.sub(r"\s*\(\d+\)$", "", stem)


def parse_duration(value: object) -> int:
    if pd.isna(value):
        return 0
    try:
        return max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return 0


def parse_tags(value: object) -> Set[str]:
    if pd.isna(value):
        return set()
    return {item.strip().lower() for item in str(value).split(",") if item.strip()}


def parse_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def choose_column(df: pd.DataFrame, *candidates: str, required: bool = True) -> Optional[str]:
    lookup = {str(col).strip().lower(): str(col) for col in df.columns}
    for candidate in candidates:
        if candidate.strip().lower() in lookup:
            return lookup[candidate.strip().lower()]
    if required:
        raise ValueError(
            f"Required column not found. Expected one of: {', '.join(candidates)}. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )
    return None


def load_data(metadata_file: Path, embeddings_file: Path) -> DatasetBundle:
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    metadata = pd.read_excel(metadata_file)
    embeddings = pd.read_csv(embeddings_file)

    filename_col = choose_column(metadata, "filename")
    duration_col = choose_column(metadata, "duration_seconds", "duration")
    category_col = choose_column(metadata, "category")
    environment_col = choose_column(metadata, "environment")
    mood_col = choose_column(metadata, "mood", "perceptual_quality")
    scenario_col = choose_column(metadata, "scenario_tags")
    function_col = choose_column(metadata, "function")
    elements_col = choose_column(metadata, "elements")
    human_presence_col = choose_column(metadata, "human presence", "human_presence")
    playback_character_col = choose_column(metadata, "playback_character", required=False)
    embedding_filename_col = choose_column(embeddings, "filename")

    metadata = metadata.copy()
    embeddings = embeddings.copy()

    metadata["filekey"] = metadata[filename_col].apply(canonical_filekey)
    embeddings["filekey"] = embeddings[embedding_filename_col].apply(canonical_filekey)

    metadata["duration_sec"] = metadata[duration_col].apply(parse_duration)
    metadata["scenario_set"] = metadata[scenario_col].apply(parse_tags)
    metadata["mood_norm"] = metadata[mood_col].apply(parse_text)
    metadata["environment_norm"] = metadata[environment_col].apply(parse_text)
    metadata["category_norm"] = metadata[category_col].apply(parse_text)
    metadata["function_norm"] = metadata[function_col].apply(parse_text)
    metadata["elements_set"] = metadata[elements_col].apply(parse_tags)
    metadata["human_presence_norm"] = metadata[human_presence_col].apply(parse_text)
    metadata["playback_character_norm"] = (
        metadata[playback_character_col].apply(parse_text)
        if playback_character_col is not None
        else ""
    )
    metadata["filename_original"] = metadata[filename_col]
    metadata["scenario_tags_original"] = metadata[scenario_col]

    duplicate_metadata_keys = sorted(
        metadata.loc[
            metadata["filekey"].duplicated(keep=False) & (metadata["filekey"] != ""),
            "filekey",
        ].unique().tolist()
    )
    duplicate_embedding_keys = sorted(
        embeddings.loc[
            embeddings["filekey"].duplicated(keep=False) & (embeddings["filekey"] != ""),
            "filekey",
        ].unique().tolist()
    )

    metadata_keys = set(metadata.loc[metadata["filekey"] != "", "filekey"])
    embedding_keys = set(embeddings.loc[embeddings["filekey"] != "", "filekey"])

    missing_key_set = metadata_keys - embedding_keys
    extra_key_set = embedding_keys - metadata_keys

    missing_embeddings = sorted(
        metadata.loc[metadata["filekey"].isin(missing_key_set), "filename_original"]
        .astype(str).tolist()
    )
    embeddings_without_metadata = sorted(set(
        embeddings.loc[embeddings["filekey"].isin(extra_key_set), embedding_filename_col]
        .astype(str).tolist()
    ))

    merged = metadata.merge(
        embeddings,
        on="filekey",
        how="inner",
        suffixes=("_meta", "_emb"),
    )

    print(f"Spreadsheet rows: {len(metadata)}")
    print(f"Embedding rows: {len(embeddings)}")
    print(f"Matched rows: {len(merged)}")

    if duplicate_metadata_keys:
        print("\nDuplicate canonical keys in metadata:")
        for key in duplicate_metadata_keys:
            print(f"  - {key}")
    if duplicate_embedding_keys:
        print("\nDuplicate canonical keys in embeddings:")
        for key in duplicate_embedding_keys:
            print(f"  - {key}")
    if missing_embeddings:
        print("\nMetadata rows missing embeddings:")
        for name in missing_embeddings:
            print(f"  - {name}")
    if embeddings_without_metadata:
        print("\nEmbedding rows without metadata:")
        for name in embeddings_without_metadata:
            print(f"  - {name}")
    if merged.empty:
        raise ValueError("No matched rows found between metadata and embeddings.")

    return DatasetBundle(
        metadata=metadata,
        embeddings=embeddings,
        merged=merged,
        missing_embeddings=missing_embeddings,
        embeddings_without_metadata=embeddings_without_metadata,
        duplicate_metadata_keys=duplicate_metadata_keys,
        duplicate_embedding_keys=duplicate_embedding_keys,
    )


def get_embedding_columns(df: pd.DataFrame) -> List[str]:
    emb_cols = [str(col) for col in df.columns if str(col).isdigit()]
    if not emb_cols:
        raise ValueError("No numeric embedding columns found in the embeddings CSV.")
    return sorted(emb_cols, key=int)


def get_embedding_vector(row: pd.Series, emb_cols: List[str]) -> np.ndarray:
    return row[emb_cols].to_numpy(dtype=np.float32)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denominator == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denominator)


def scenario_match(row: pd.Series, scenario: str) -> bool:
    return scenario.lower() in row["scenario_set"]


def profile_for(scenario: str) -> Dict:
    return SCENARIO_PROFILES[scenario.lower()]


def row_score(row: pd.Series, scenario: str) -> float:
    profile = profile_for(scenario)
    score = 0.0

    if scenario_match(row, scenario):
        score += 10.0

    if row["mood_norm"] in profile["preferred_moods"]:
        score += 3.0

    if row["environment_norm"] in profile["preferred_environments"]:
        score += 2.5

    if row["category_norm"] in profile["preferred_categories"]:
        score += 2.0

    if row["category_norm"] in profile["avoided_categories"]:
        score -= 5.0

    # Habitation is scored through audible occupation and activity.
    # Water is not excluded: a recording such as "people, water" can score
    # strongly, while a recording containing only water/waves/rain/wind is
    # downweighted.
    preferred_elements = profile.get("preferred_elements", set())
    if preferred_elements:
        matching_elements = row["elements_set"] & preferred_elements
        score += min(len(matching_elements) * 1.25, 4.0)

    preferred_human_presence = profile.get("preferred_human_presence", set())
    if row["human_presence_norm"] in preferred_human_presence:
        if row["human_presence_norm"] == "direct":
            score += 3.0
        else:
            score += 1.5

    downweighted_alone = profile.get("downweighted_elements_when_alone", set())
    if (
        downweighted_alone
        and row["elements_set"]
        and row["elements_set"].issubset(downweighted_alone)
    ):
        score -= 4.0

    # In habitation, bird-only recordings can remain as ecological detail,
    # but they should not dominate the scenario or become its main foundation.
    if scenario.lower() == "habitation":
        bird_terms = {"bird", "birds", "birdsong"}
        human_activity_terms = profile.get("preferred_elements", set())
        has_birds = bool(row["elements_set"] & bird_terms)
        has_human_activity = bool(row["elements_set"] & human_activity_terms)

        if has_birds and not has_human_activity:
            score -= 5.0

    function_order = {name: i for i, name in enumerate(profile["preferred_functions"])}
    if row["function_norm"] in function_order:
        score += max(0.0, 2.0 - function_order[row["function_norm"]] * 0.4)

    score += min(row["duration_sec"] / 300.0, 3.0)
    return score


def choose_seed(
    df: pd.DataFrame,
    scenario: str,
    rng: random.Random,
) -> pd.Series:
    candidates = df[df.apply(lambda row: scenario_match(row, scenario), axis=1)].copy()
    if candidates.empty:
        raise ValueError(f"No matched sounds found for scenario '{scenario}'.")

    if scenario.lower() == "habitation":
        bird_terms = {"bird", "birds", "birdsong"}
        human_terms = profile_for("habitation").get("preferred_elements", set())

        candidates["bird_only_habitation"] = candidates.apply(
            lambda row: (
                bool(row["elements_set"] & bird_terms)
                and not bool(row["elements_set"] & human_terms)
            ),
            axis=1,
        )

        non_bird_candidates = candidates[~candidates["bird_only_habitation"]].copy()
        if not non_bird_candidates.empty:
            candidates = non_bird_candidates

    candidates["seed_score"] = candidates.apply(
        lambda row: row_score(row, scenario), axis=1
    )
    candidates = candidates.sort_values("seed_score", ascending=False)

    top_n = min(len(candidates), 5)
    return candidates.iloc[rng.randrange(top_n)]


def candidate_pool(
    df: pd.DataFrame,
    seed_row: pd.Series,
    scenario: str,
    emb_cols: List[str],
    fallback: bool,
) -> pd.DataFrame:
    profile = profile_for(scenario)
    seed_vec = get_embedding_vector(seed_row, emb_cols)

    if fallback:
        pool = df[df["filekey"] != seed_row["filekey"]].copy()
        pool = pool[
            (pool["environment_norm"] == seed_row["environment_norm"])
            | (pool["category_norm"] == seed_row["category_norm"])
        ].copy()
        similarity_weight = 0.75
        metadata_weight = 0.25
    else:
        pool = df[df.apply(lambda row: scenario_match(row, scenario), axis=1)].copy()
        pool = pool[pool["filekey"] != seed_row["filekey"]].copy()
        similarity_weight = 0.70
        metadata_weight = 0.30

    if pool.empty:
        return pool

    if profile["avoided_categories"]:
        strict_pool = pool[
            ~pool["category_norm"].isin(profile["avoided_categories"])
        ].copy()
        if not strict_pool.empty:
            pool = strict_pool

    pool["similarity"] = pool.apply(
        lambda row: cosine_similarity(seed_vec, get_embedding_vector(row, emb_cols)),
        axis=1,
    )
    pool["row_score"] = pool.apply(lambda row: row_score(row, scenario), axis=1)
    pool["combined_score"] = (
        pool["similarity"] * similarity_weight
        + pool["row_score"] * metadata_weight / 10.0
    )
    pool["pool_type"] = "fallback" if fallback else "same_scenario"
    return pool.sort_values("combined_score", ascending=False)


def select_layers(
    df: pd.DataFrame,
    seed_row: pd.Series,
    scenario: str,
    emb_cols: List[str],
    rng: random.Random,
) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = [{
        "row": seed_row,
        "selection_role": "seed",
        "similarity": 1.0,
        "row_score": row_score(seed_row, scenario),
        "combined_score": row_score(seed_row, scenario),
        "pool_type": "seed",
    }]
    used_keys = {seed_row["filekey"]}

    same_pool = candidate_pool(df, seed_row, scenario, emb_cols, fallback=False)
    fallback_pool = candidate_pool(df, seed_row, scenario, emb_cols, fallback=True)

    target_order = ["base", "texture", "foreground", "accent"]
    if seed_row["function_norm"] in target_order:
        target_order.remove(seed_row["function_norm"])

    def try_pick(pool: pd.DataFrame, wanted_function: str) -> Optional[pd.Series]:
        if pool.empty:
            return None

        candidates = pool[
            (~pool["filekey"].isin(used_keys))
            & (pool["function_norm"] == wanted_function)
        ].copy()

        if candidates.empty and wanted_function == "base":
            candidates = pool[
                (~pool["filekey"].isin(used_keys))
                & (pool["function_norm"] == "texture")
            ].copy()

        if scenario.lower() == "habitation" and wanted_function == "base" and not candidates.empty:
            bird_terms = {"bird", "birds", "birdsong"}
            human_terms = profile_for("habitation").get("preferred_elements", set())

            candidates["bird_only_habitation"] = candidates.apply(
                lambda row: (
                    bool(row["elements_set"] & bird_terms)
                    and not bool(row["elements_set"] & human_terms)
                ),
                axis=1,
            )
            non_bird_candidates = candidates[~candidates["bird_only_habitation"]].copy()
            if not non_bird_candidates.empty:
                candidates = non_bird_candidates

        if candidates.empty:
            return None

        # Controlled variation: select among the top three candidates,
        # weighted in favour of the strongest-ranked candidate.
        top = candidates.head(3)
        weights = [3, 2, 1][:len(top)]
        chosen_index = rng.choices(list(top.index), weights=weights, k=1)[0]
        return top.loc[chosen_index]

    for target in target_order:
        row = try_pick(same_pool, target)
        if row is None and target in {"base", "texture", "foreground"}:
            row = try_pick(fallback_pool, target)

        if row is not None:
            selected.append({
                "row": row,
                "selection_role": target,
                "similarity": float(row.get("similarity", 0.0)),
                "row_score": float(row.get("row_score", 0.0)),
                "combined_score": float(row.get("combined_score", 0.0)),
                "pool_type": str(row.get("pool_type", "")),
            })
            used_keys.add(row["filekey"])

        if len(selected) >= 4:
            break

    return selected


def build_audio_index(sounds_dir: Path) -> Dict[str, Path]:
    if not sounds_dir.exists():
        raise FileNotFoundError(f"Sounds directory not found: {sounds_dir}")

    index: Dict[str, Path] = {}
    for path in sounds_dir.rglob("*"):
        if path.is_file():
            key = canonical_filekey(path.name)
            if key and key not in index:
                index[key] = path
    return index


def layer_settings(function_name: str) -> Dict[str, object]:
    settings = {
        "base": {"gain": 0.42, "offset_range": (0, 0), "fade_in": 3.0, "fade_out": 5.0},
        "texture": {"gain": 0.22, "offset_range": (3, 15), "fade_in": 4.0, "fade_out": 5.0},
        "foreground": {"gain": 0.18, "offset_range": (8, 20), "fade_in": 2.0, "fade_out": 4.0},
        "accent": {"gain": 0.14, "offset_range": (15, 35), "fade_in": 1.0, "fade_out": 3.0},
    }
    return settings.get(
        function_name.lower(),
        {"gain": 0.18, "offset_range": (5, 15), "fade_in": 2.0, "fade_out": 4.0},
    )


def first_audible_time(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = INITIAL_AUDIBILITY_THRESHOLD,
) -> Optional[float]:
    if len(audio) == 0:
        return None

    window = max(1, int(AUDIBLE_WINDOW_SECONDS * sample_rate))
    usable = len(audio) - (len(audio) % window)
    if usable <= 0:
        return None

    frames = audio[:usable].reshape(-1, window)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))

    positive = rms[rms > 0]
    if positive.size == 0:
        return None

    reference = float(np.percentile(positive, 90))
    adaptive_threshold = max(threshold, reference * 0.05)
    audible_frames = np.flatnonzero(rms >= adaptive_threshold)

    if audible_frames.size == 0:
        return None

    return float(audible_frames[0] * window / sample_rate)


def build_mix(
    selected: List[Dict[str, object]],
    output_path: Path,
    audio_index: Dict[str, Path],
    rng: random.Random,
    scenario_name: str,
) -> Tuple[Path, List[Dict[str, object]]]:
    target_len = TARGET_DURATION * TARGET_SR
    master = np.zeros(target_len, dtype=np.float32)
    layer_log: List[Dict[str, object]] = []

    # Guarantee an audible opening:
    # - the selected seed always starts at 0 seconds;
    # - the seed uses a short fade-in;
    # - if the seed is an accent/foreground, the earliest base or texture also starts at 0.
    for layer_index, item in enumerate(selected):
        row = item["row"]
        path = audio_index.get(row["filekey"])
        if path is None:
            print(f"Skipping missing audio file: {row['filename_original']}")
            continue

        audio = audio_core.load_audio(path)
        settings = layer_settings(row["function_norm"])

        is_seed = item["selection_role"] == "seed"
        is_base_role = item["selection_role"] == "base"
        is_opening_layer = is_seed or is_base_role

        clip, crop_start_sample, playback_mode, resolved_character = audio_core.prepare_clip(
            audio,
            target_len,
            rng,
            function_name=row["function_norm"],
            elements=set(row["elements_set"]),
            playback_character=row["playback_character_norm"],
            prefer_audible_opening=is_opening_layer,
        )

        if is_seed:
            offset_sec = 0
            fade_in_s = 0.75
        elif is_base_role:
            offset_sec = 0
            fade_in_s = 1.50
        else:
            offset_min, offset_max = settings["offset_range"]
            offset_sec = rng.randint(int(offset_min), int(offset_max))
            fade_in_s = float(settings["fade_in"])

        character_fades = {
            "continuous": (3.0, 3.0),
            "event_rich": (2.0, 2.0),
            "discrete_event": (0.5, 1.0),
        }
        activity_fade_in, activity_fade_out = character_fades.get(
            resolved_character, (2.0, 2.0)
        )
        clip = audio_core.smooth_activity_edges(
            clip,
            TARGET_SR,
            fade_in_s=activity_fade_in,
            fade_out_s=activity_fade_out,
        )
        clip = audio_core.apply_fade(
            clip,
            TARGET_SR,
            fade_in_s,
            float(settings["fade_out"]),
        )

        offset_samples = offset_sec * TARGET_SR
        layer = np.zeros(target_len, dtype=np.float32)
        usable = min(len(clip), target_len - offset_samples)

        if usable > 0:
            layer[offset_samples:offset_samples + usable] = clip[:usable]

        gain = float(settings["gain"])

        if scenario_name == "habitation":
            bird_terms = {"bird", "birds", "birdsong"}
            human_terms = profile_for("habitation").get("preferred_elements", set())
            bird_only = (
                bool(row["elements_set"] & bird_terms)
                and not bool(row["elements_set"] & human_terms)
            )
            if bird_only:
                gain *= 0.45

        master += layer * gain

        layer_log.append({
            "filename": row["filename_original"],
            "function": row["function_norm"],
            "selection_role": item["selection_role"],
            "category": row["category_norm"],
            "environment": row["environment_norm"],
            "mood": row["mood_norm"],
            "elements": ", ".join(sorted(row["elements_set"])),
            "human_presence": row["human_presence_norm"],
            "scenario_tags": row["scenario_tags_original"],
            "similarity": item["similarity"],
            "metadata_score": item["row_score"],
            "combined_score": item["combined_score"],
            "pool_type": item["pool_type"],
            "gain": gain,
            "offset_seconds": offset_sec,
            "fade_in_seconds": fade_in_s,
            "crop_start_seconds": crop_start_sample / TARGET_SR,
            "playback_mode": playback_mode,
            "playback_character": resolved_character,
            "source_path": str(path),
        })

    peak = float(np.max(np.abs(master)))
    if peak > 0.99:
        master = master / peak * 0.95

    opening_time = first_audible_time(master, TARGET_SR)
    if opening_time is None:
        print("WARNING: rendered output is silent.")
    elif opening_time > MAX_INITIAL_SILENCE_SECONDS:
        print(
            f"WARNING: first audible sample occurs at {opening_time:.2f}s "
            f"(target maximum: {MAX_INITIAL_SILENCE_SECONDS:.2f}s)."
        )
    else:
        print(f"First audible sample: {opening_time:.2f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, master, TARGET_SR)
    return output_path, layer_log


def append_run_log(log_path: Path, rows: List[Dict[str, object]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp", "scenario", "run_number", "seed", "output_file",
        "filename", "function", "selection_role", "category", "environment",
        "mood", "elements", "human_presence", "scenario_tags",
        "similarity", "metadata_score",
        "combined_score", "pool_type", "gain", "offset_seconds",
        "fade_in_seconds", "crop_start_seconds", "playback_mode", "playback_character", "source_path",
    ]
    exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def generate_runs(
    scenario: str,
    metadata_file: Path,
    embeddings_file: Path,
    sounds_dir: Path,
    output_dir: Path,
    runs: int,
    base_seed: int,
) -> List[Path]:
    bundle = load_data(metadata_file, embeddings_file)
    df = bundle.merged
    emb_cols = get_embedding_columns(df)
    audio_index = build_audio_index(sounds_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"generation_log_{scenario}_{timestamp}.csv"
    outputs: List[Path] = []

    for run_number in range(1, runs + 1):
        run_seed = base_seed + run_number - 1
        rng = random.Random(run_seed)
        np.random.seed(run_seed)

        seed_row = choose_seed(df, scenario, rng)
        selected = select_layers(df, seed_row, scenario, emb_cols, rng)

        output_path = output_dir / (
            f"generated_{scenario}_v12_seed{run_seed}_run{run_number:02d}.wav"
        )

        print(f"\nRun {run_number}/{runs}")
        print(f"Scenario: {scenario}")
        print(f"Seed value: {run_seed}")
        print(f"Seed recording: {seed_row['filename_original']}")
        print("Selected layers:")
        for item in selected:
            row = item["row"]
            print(
                f"  - {row['filename_original']} | "
                f"function={row['function_norm']} | "
                f"similarity={float(item['similarity']):.4f} | "
                f"combined={float(item['combined_score']):.4f}"
            )

        output_path, layer_log = build_mix(selected, output_path, audio_index, rng, scenario)
        outputs.append(output_path)

        timestamp_iso = datetime.now().isoformat(timespec="seconds")
        log_rows = []
        for layer in layer_log:
            log_rows.append({
                "timestamp": timestamp_iso,
                "scenario": scenario,
                "run_number": run_number,
                "seed": run_seed,
                "output_file": str(output_path),
                **layer,
            })
        append_run_log(log_path, log_rows)
        print(f"Generated file: {output_path.resolve()}")

    print(f"\nGeneration log: {log_path.resolve()}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the dataset and generate reproducible scenario-based soundscapes with spreadsheet-defined playback characters and smoothed internal activity edges."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_FILE)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS_FILE)
    parser.add_argument("--sounds-dir", type=Path, default=DEFAULT_SOUNDS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario", choices=sorted(SCENARIO_PROFILES), default="repair")
    parser.add_argument("--runs", type=int, default=1, help="Number of outputs to generate.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base reproducibility seed. Each additional run increments it by one.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check metadata/embedding alignment without rendering audio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")

    if args.validate_only:
        load_data(args.metadata, args.embeddings)
        return

    generate_runs(
        scenario=args.scenario,
        metadata_file=args.metadata,
        embeddings_file=args.embeddings,
        sounds_dir=args.sounds_dir,
        output_dir=args.output_dir,
        runs=args.runs,
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()
