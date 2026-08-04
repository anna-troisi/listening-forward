from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import argparse
import hashlib
import math
import random

import numpy as np
import soundfile as sf

import generate_soundscape_v12 as base
import listening_forward_audio_v3 as audio_core


# Temporary prototype orientations.
# These let us test continuous tuning before fixing the final public vocabulary.
ORIENTATIONS = ["repair", "automation", "habitation", "biodiversity"]

ROLE_ORDER = ["base", "texture", "foreground", "accent"]
ROLE_GAINS = {
    "base": 0.42,
    "texture": 0.24,
    "foreground": 0.18,
    "accent": 0.13,
}


@dataclass
class DialState:
    index: int
    position: float
    weights: Dict[str, float]
    selected_keys: List[str]
    replaced_key: str = ""
    introduced_key: str = ""


def orientation_weights(position: float) -> Dict[str, float]:
    """
    Convert a circular dial position in [0, 1) into weights for two adjacent
    orientations. Example: 0.125 lies halfway between repair and automation.
    """
    position = position % 1.0
    scaled = position * len(ORIENTATIONS)
    left_index = int(math.floor(scaled)) % len(ORIENTATIONS)
    fraction = scaled - math.floor(scaled)
    right_index = (left_index + 1) % len(ORIENTATIONS)

    weights = {name: 0.0 for name in ORIENTATIONS}
    weights[ORIENTATIONS[left_index]] = 1.0 - fraction
    weights[ORIENTATIONS[right_index]] = fraction
    return weights


def blended_score(row, weights: Dict[str, float]) -> float:
    return sum(
        weight * base.row_score(row, scenario)
        for scenario, weight in weights.items()
        if weight > 0
    )


def candidate_table(df, weights: Dict[str, float]):
    table = df.copy()
    table["dial_score"] = table.apply(lambda row: blended_score(row, weights), axis=1)
    return table.sort_values("dial_score", ascending=False)


def choose_initial_layers(df, weights: Dict[str, float], rng: random.Random) -> List[str]:
    candidates = candidate_table(df, weights)
    selected: List[str] = []

    for role in ROLE_ORDER:
        role_rows = candidates[candidates["function_norm"] == role].head(5)
        if role_rows.empty and role == "base":
            role_rows = candidates[candidates["function_norm"] == "texture"].head(5)
        if role_rows.empty:
            continue

        top = role_rows.head(3)
        chosen_index = rng.choices(
            list(top.index),
            weights=[3, 2, 1][:len(top)],
            k=1,
        )[0]
        key = str(top.loc[chosen_index, "filekey"])
        if key not in selected:
            selected.append(key)

    for key in candidates["filekey"]:
        if len(selected) >= 4:
            break
        if key not in selected:
            selected.append(str(key))

    return selected[:4]


def update_layers(
    df,
    current_keys: List[str],
    weights: Dict[str, float],
    rng: random.Random,
    force_replace_keys: Optional[Set[str]] = None,
    excluded_keys: Optional[Set[str]] = None,
) -> Tuple[List[str], str, str]:
    """
    Preserve continuity by replacing at most one layer per dial state.

    Event-like one-shots are forced out after they have sounded once, so a horn,
    bell, beep, saw, or other discrete event is not replayed at every state.
    """
    force_replace_keys = force_replace_keys or set()
    excluded_keys = excluded_keys or set()
    candidates = candidate_table(df, weights)
    if excluded_keys:
        candidates = candidates[~candidates["filekey"].isin(excluded_keys)].copy()
    score_lookup = dict(zip(candidates["filekey"], candidates["dial_score"]))

    available = candidates[~candidates["filekey"].isin(current_keys)].head(8)
    if available.empty:
        return current_keys, "", ""

    top = available.head(3)
    candidate_index = rng.choices(
        list(top.index),
        weights=[3, 2, 1][:len(top)],
        k=1,
    )[0]
    candidate_key = str(top.loc[candidate_index, "filekey"])
    candidate_score = float(top.loc[candidate_index, "dial_score"])

    forced_candidates = [key for key in current_keys if key in force_replace_keys]
    if forced_candidates:
        replace_key = forced_candidates[0]
        updated = list(current_keys)
        updated[updated.index(replace_key)] = candidate_key
        return updated, replace_key, candidate_key

    current_scores = [(key, float(score_lookup.get(key, -999.0))) for key in current_keys]
    weakest_key, weakest_score = min(current_scores, key=lambda item: item[1])

    # Hysteresis prevents restless switching for tiny score differences.
    if candidate_score < weakest_score + 1.25:
        return current_keys, "", ""

    updated = list(current_keys)
    updated[updated.index(weakest_key)] = candidate_key
    return updated, weakest_key, candidate_key


def load_stable_excerpt(
    row,
    audio_index: Dict[str, Path],
    session_seed: int,
    state_index: int,
    target_samples: int,
) -> Tuple[np.ndarray, str]:
    path = audio_index.get(row["filekey"])
    if path is None:
        return np.zeros(target_samples, dtype=np.float32), "continuous"

    audio = audio_core.load_audio(path)
    character = playback_character_for_row(row)

    # Continuous layers retain a stable excerpt. Event-rich layers receive a
    # new deterministic excerpt at each state. Discrete events are handled by
    # the state scheduler and triggered only when introduced.
    excerpt_state = 0 if character == "continuous" else state_index
    local_rng = audio_core.deterministic_rng(
        session_seed,
        str(row["filekey"]),
        excerpt_state,
    )

    clip, _, _, resolved_character = audio_core.prepare_clip(
        audio,
        target_samples,
        local_rng,
        function_name=str(row["function_norm"]),
        elements=set(row["elements_set"]),
        playback_character=character,
        prefer_audible_opening=False,
    )

    character_fades = {
        "continuous": (4.0, 4.0),
        "event_rich": (2.5, 2.5),
        "discrete_event": (0.6, 1.2),
    }
    fade_in_s, fade_out_s = character_fades.get(
        resolved_character, (2.5, 2.5)
    )
    clip = audio_core.smooth_activity_edges(
        clip,
        base.TARGET_SR,
        fade_in_s=fade_in_s,
        fade_out_s=fade_out_s,
    )
    return clip, resolved_character


def role_for_row(row) -> str:
    function_name = str(row["function_norm"]).lower()
    return function_name if function_name in ROLE_GAINS else "texture"


def playback_character_for_row(row) -> str:
    return audio_core.resolve_playback_character(
        row.get("playback_character_norm", ""),
        str(row["function_norm"]),
        set(row["elements_set"]),
        float(row.get("duration_sec", 0.0)),
    )



def render_state(
    df,
    selected_keys: List[str],
    weights: Dict[str, float],
    audio_index: Dict[str, Path],
    session_seed: int,
    state_samples: int,
    state_index: int,
    active_discrete_keys: Set[str],
) -> np.ndarray:
    state = np.zeros(state_samples, dtype=np.float32)
    rows_by_key = {str(row["filekey"]): row for _, row in df.iterrows()}

    raw_scores = []
    for key in selected_keys:
        row = rows_by_key[key]
        raw_scores.append(max(0.0, blended_score(row, weights)))

    max_score = max(raw_scores) if raw_scores else 1.0

    for key, raw_score in zip(selected_keys, raw_scores):
        row = rows_by_key[key]
        role = role_for_row(row)
        character = playback_character_for_row(row)

        # Discrete events sound only in the state in which they are introduced.
        if character == "discrete_event" and key not in active_discrete_keys:
            continue

        clip, character = load_stable_excerpt(
            row,
            audio_index,
            session_seed=session_seed,
            state_index=state_index,
            target_samples=state_samples,
        )

        relevance = 0.35 + 0.65 * (raw_score / max_score if max_score else 0.0)
        gain = ROLE_GAINS[role] * relevance
        state += clip * gain

    peak = float(np.max(np.abs(state)))
    if peak > 0.95:
        state = state / peak * 0.92

    return state


def equal_power_crossfade(
    left: np.ndarray,
    right: np.ndarray,
    crossfade_samples: int,
) -> np.ndarray:
    if crossfade_samples <= 0:
        return np.concatenate([left, right])

    crossfade_samples = min(crossfade_samples, len(left), len(right))
    theta = np.linspace(0.0, np.pi / 2.0, crossfade_samples)
    fade_out = np.cos(theta)
    fade_in = np.sin(theta)

    overlap = (
        left[-crossfade_samples:] * fade_out
        + right[:crossfade_samples] * fade_in
    )
    return np.concatenate([
        left[:-crossfade_samples],
        overlap,
        right[crossfade_samples:],
    ])


def build_radio_trajectory(
    metadata_file: Path,
    embeddings_file: Path,
    sounds_dir: Path,
    output_file: Path,
    states: int,
    state_duration: float,
    crossfade_duration: float,
    session_seed: int,
) -> Tuple[Path, Path]:
    bundle = base.load_data(metadata_file, embeddings_file)
    df = bundle.merged
    audio_index = base.build_audio_index(sounds_dir)

    rng = random.Random(session_seed)
    state_samples = int(state_duration * base.TARGET_SR)
    crossfade_samples = int(crossfade_duration * base.TARGET_SR)

    positions = [index / states for index in range(states)]
    positions.append(1.0)  # Return to repair and close the circle.

    all_states: List[DialState] = []
    rendered_states: List[np.ndarray] = []

    current_keys: List[str] = []
    expired_event_keys: Set[str] = set()
    cooldown_until: Dict[str, int] = {}

    for index, position in enumerate(positions):
        weights = orientation_weights(position)

        if not current_keys:
            current_keys = choose_initial_layers(df, weights, rng)
            replaced = ""
            introduced = ""
            active_event_keys = {
                key for key in current_keys
                if playback_character_for_row(df[df["filekey"] == key].iloc[0])
                == "discrete_event"
            }
        else:
            current_keys, replaced, introduced = update_layers(
                df,
                current_keys,
                weights,
                rng,
                force_replace_keys=expired_event_keys,
                excluded_keys={
                    key for key, until_state in cooldown_until.items()
                    if index < until_state
                },
            )

            # Only a newly introduced event is triggered in this state.
            active_event_keys = set()
            if introduced:
                introduced_row = df[df["filekey"] == introduced].iloc[0]
                if playback_character_for_row(introduced_row) == "discrete_event":
                    active_event_keys.add(introduced)

        # Discrete events expire immediately and cannot return for three states.
        expired_event_keys = set(active_event_keys)
        for key in active_event_keys:
            cooldown_until[key] = index + 3

        state = DialState(
            index=index,
            position=position % 1.0,
            weights=weights,
            selected_keys=list(current_keys),
            replaced_key=replaced,
            introduced_key=introduced,
        )
        all_states.append(state)

        rendered = render_state(
            df,
            current_keys,
            weights,
            audio_index,
            session_seed=session_seed,
            state_samples=state_samples,
            state_index=index,
            active_discrete_keys=active_event_keys,
        )
        rendered_states.append(rendered)

        readable_weights = ", ".join(
            f"{name}={weight:.2f}"
            for name, weight in weights.items()
            if weight > 0
        )
        print(f"State {index:02d}: {readable_weights}")
        if replaced:
            print(f"  replaced: {replaced}")
            print(f"  introduced: {introduced}")
        for key in current_keys:
            print(f"  - {key}")

    master = rendered_states[0]
    for next_state in rendered_states[1:]:
        master = equal_power_crossfade(master, next_state, crossfade_samples)

    peak = float(np.max(np.abs(master)))
    if peak > 0.98:
        master = master / peak * 0.95

    output_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_file, master, base.TARGET_SR)

    manifest_file = output_file.with_suffix(".csv")
    with manifest_file.open("w", encoding="utf-8") as handle:
        handle.write(
            "state,position,repair,automation,habitation,biodiversity,"
            "selected_keys,replaced_key,introduced_key\n"
        )
        for state in all_states:
            handle.write(
                f"{state.index},{state.position:.6f},"
                f"{state.weights['repair']:.6f},"
                f"{state.weights['automation']:.6f},"
                f"{state.weights['habitation']:.6f},"
                f"{state.weights['biodiversity']:.6f},"
                f"\"{' | '.join(state.selected_keys)}\","
                f"\"{state.replaced_key}\","
                f"\"{state.introduced_key}\"\n"
            )

    print(f"\nRadio trajectory: {output_file.resolve()}")
    print(f"Transition manifest: {manifest_file.resolve()}")
    return output_file, manifest_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render an offline stateful Listening Forward dial trajectory using smoothed layer entrances, longer state crossfades, and spreadsheet-defined playback behaviour. "
            "The temporary circle is repair → automation → habitation → "
            "biodiversity → repair."
        )
    )
    parser.add_argument("--metadata", type=Path, default=Path("Soundscape_Metadata2.xlsx"))
    parser.add_argument("--embeddings", type=Path, default=Path("sound_embeddings.csv"))
    parser.add_argument("--sounds-dir", type=Path, default=Path("sounds"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated/listening_forward_radio_v6.wav"),
    )
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--state-duration", type=float, default=12.0)
    parser.add_argument("--crossfade-duration", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.states < 4:
        raise ValueError("--states must be at least 4.")
    if args.state_duration <= 0:
        raise ValueError("--state-duration must be positive.")
    if not 0 <= args.crossfade_duration < args.state_duration:
        raise ValueError(
            "--crossfade-duration must be non-negative and shorter than state duration."
        )

    build_radio_trajectory(
        metadata_file=args.metadata,
        embeddings_file=args.embeddings,
        sounds_dir=args.sounds_dir,
        output_file=args.output,
        states=args.states,
        state_duration=args.state_duration,
        crossfade_duration=args.crossfade_duration,
        session_seed=args.seed,
    )


if __name__ == "__main__":
    main()
