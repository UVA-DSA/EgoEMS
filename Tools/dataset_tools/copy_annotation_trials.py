#!/usr/bin/env python3
"""
Copy selected trial streams from an EgoExoEMS annotation JSON.

The annotation stores stream paths relative to the dataset root. This script
copies only the stream names you request and preserves that relative structure
under the destination directory.

Example:
    python Tools/dataset_tools/copy_annotation_trials.py \
        --annotation Annotations/splits/trials/aaai26_test_split_classification.json \
        --source-root /path/to/full/dataset/root \
        --output-dir /path/to/copied/test_split \
        --streams ego smartwatch_data distance_sensor_data \
        --ignore-subject P \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any


# === DEFAULT CONFIGURATION ===
# These can be edited directly, or overridden with command-line arguments.
ANNOTATION_JSON = Path("Annotations/splits/trials/aaai26_test_split_classification.json")
SOURCE_ROOT: Path | str | None = None
OUTPUT_DIR: Path | str | None = None
STREAMS_TO_COPY: list[str] = []  # Example: ["ego", "smartwatch_data"]
IGNORE_SUBJECT: str | list[str] | None = None  # Example: "P"
DRY_RUN = False
OVERWRITE = False


@dataclass(frozen=True)
class CopyItem:
    subject_id: str
    scenario_id: str
    trial_id: str
    stream_name: str
    relative_path: Path
    source_path: Path
    destination_path: Path


@dataclass
class CopyStats:
    subjects_seen: int = 0
    subjects_skipped: int = 0
    scenarios_seen: int = 0
    trials_seen: int = 0
    trials_skipped_by_subject: int = 0
    trials_considered: int = 0
    trials_with_selected_streams: int = 0
    trials_without_selected_streams: int = 0
    selected_stream_entries: int = 0
    requested_streams_missing: int = 0
    missing_file_path_fields: int = 0
    invalid_relative_paths: int = 0
    duplicate_relative_paths: int = 0
    source_files_missing: int = 0
    destination_exists_same_size: int = 0
    destination_exists_different_size: int = 0
    files_would_copy: int = 0
    files_would_overwrite: int = 0
    files_copied: int = 0
    files_overwritten: int = 0
    bytes_would_copy: int = 0
    bytes_copied: int = 0
    entries_by_stream: Counter[str] = field(default_factory=Counter)
    missing_requested_by_stream: Counter[str] = field(default_factory=Counter)
    missing_sources_by_stream: Counter[str] = field(default_factory=Counter)
    copied_by_stream: Counter[str] = field(default_factory=Counter)
    samples: dict[str, list[str]] = field(
        default_factory=lambda: {
            "missing_requested_streams": [],
            "missing_sources": [],
            "invalid_relative_paths": [],
            "different_size_destinations": [],
        }
    )


def parse_list(values: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]

    parsed: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                parsed.append(item)
    return parsed


def as_path(value: Path | str) -> Path:
    if isinstance(value, Path):
        return value
    return Path(value)


def normalize_cli_path(value: Path | str, label: str) -> tuple[Path, str | None]:
    raw_value = str(value)
    normalized = raw_value.replace("\\ ", " ")
    if normalized == raw_value:
        return as_path(value), None

    warning = (
        f"Converted shell-escaped spaces in {label}: "
        f"{raw_value!r} -> {normalized!r}"
    )
    return Path(normalized), warning


def safe_relative_path(raw_path: str) -> Path:
    normalized = raw_path.replace("\\", "/")
    posix_path = PurePosixPath(normalized)

    if posix_path.is_absolute():
        raise ValueError("absolute paths are not allowed")

    bad_parts = {"", ".", ".."}
    if any(part in bad_parts for part in posix_path.parts):
        raise ValueError("path must be a clean relative path")

    return Path(*posix_path.parts)


def add_sample(stats: CopyStats, key: str, value: str, limit: int) -> None:
    if len(stats.samples[key]) < limit:
        stats.samples[key].append(value)


def load_annotation(annotation_path: Path) -> dict[str, Any]:
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict) or not isinstance(data.get("subjects"), list):
        raise ValueError("Annotation JSON must contain a top-level 'subjects' list.")
    return data


def collect_copy_items(
    data: dict[str, Any],
    source_root: Path,
    output_dir: Path,
    streams_to_copy: list[str],
    ignore_subject_prefixes: list[str],
    sample_limit: int,
) -> tuple[list[CopyItem], CopyStats]:
    stats = CopyStats()
    copy_items: list[CopyItem] = []
    seen_relative_paths: set[Path] = set()

    for subject in data["subjects"]:
        stats.subjects_seen += 1
        subject_id = str(subject.get("subject_id", ""))
        scenarios = subject.get("scenarios", [])
        subject_is_ignored = any(
            subject_id.startswith(prefix) for prefix in ignore_subject_prefixes
        )

        if subject_is_ignored:
            stats.subjects_skipped += 1

        for scenario in scenarios:
            stats.scenarios_seen += 1
            scenario_id = str(scenario.get("scenario_id", ""))
            trials = scenario.get("trials", [])

            for trial in trials:
                stats.trials_seen += 1
                trial_id = str(trial.get("trial_id", ""))

                if subject_is_ignored:
                    stats.trials_skipped_by_subject += 1
                    continue

                stats.trials_considered += 1
                trial_streams = trial.get("streams", {})
                selected_in_trial = 0

                for stream_name in streams_to_copy:
                    stream_info = trial_streams.get(stream_name)
                    if not stream_info:
                        stats.requested_streams_missing += 1
                        stats.missing_requested_by_stream[stream_name] += 1
                        add_sample(
                            stats,
                            "missing_requested_streams",
                            f"{subject_id}/{scenario_id}/{trial_id}: {stream_name}",
                            sample_limit,
                        )
                        continue

                    raw_file_path = stream_info.get("file_path")
                    if not raw_file_path:
                        stats.missing_file_path_fields += 1
                        continue

                    try:
                        relative_path = safe_relative_path(str(raw_file_path))
                    except ValueError as exc:
                        stats.invalid_relative_paths += 1
                        add_sample(
                            stats,
                            "invalid_relative_paths",
                            f"{subject_id}/{scenario_id}/{trial_id}/{stream_name}: "
                            f"{raw_file_path} ({exc})",
                            sample_limit,
                        )
                        continue

                    if relative_path in seen_relative_paths:
                        stats.duplicate_relative_paths += 1
                        continue

                    seen_relative_paths.add(relative_path)
                    selected_in_trial += 1
                    stats.selected_stream_entries += 1
                    stats.entries_by_stream[stream_name] += 1

                    copy_items.append(
                        CopyItem(
                            subject_id=subject_id,
                            scenario_id=scenario_id,
                            trial_id=trial_id,
                            stream_name=stream_name,
                            relative_path=relative_path,
                            source_path=source_root / relative_path,
                            destination_path=output_dir / relative_path,
                        )
                    )

                if selected_in_trial:
                    stats.trials_with_selected_streams += 1
                else:
                    stats.trials_without_selected_streams += 1

    return copy_items, stats


def copy_or_plan_items(
    copy_items: list[CopyItem],
    stats: CopyStats,
    dry_run: bool,
    overwrite: bool,
    sample_limit: int,
) -> None:
    for item in copy_items:
        source_path = item.source_path
        destination_path = item.destination_path

        if not source_path.exists():
            stats.source_files_missing += 1
            stats.missing_sources_by_stream[item.stream_name] += 1
            add_sample(stats, "missing_sources", str(source_path), sample_limit)
            continue

        source_size = source_path.stat().st_size

        if destination_path.exists():
            destination_size = destination_path.stat().st_size
            if overwrite:
                if dry_run:
                    stats.files_would_overwrite += 1
                    stats.bytes_would_copy += source_size
                else:
                    destination_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, destination_path)
                    stats.files_overwritten += 1
                    stats.bytes_copied += source_size
                    stats.copied_by_stream[item.stream_name] += 1
                continue

            if destination_size == source_size:
                stats.destination_exists_same_size += 1
            else:
                stats.destination_exists_different_size += 1
                add_sample(
                    stats,
                    "different_size_destinations",
                    f"{destination_path} (src={source_size}, dst={destination_size})",
                    sample_limit,
                )
            continue

        if dry_run:
            stats.files_would_copy += 1
            stats.bytes_would_copy += source_size
            continue

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        stats.files_copied += 1
        stats.bytes_copied += source_size
        stats.copied_by_stream[item.stream_name] += 1


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def print_counter(title: str, counter: Counter[str]) -> None:
    print(title)
    if not counter:
        print("  none")
        return
    for key in sorted(counter):
        print(f"  {key}: {counter[key]}")


def print_samples(title: str, samples: list[str]) -> None:
    if not samples:
        return
    print(title)
    for sample in samples:
        print(f"  {sample}")


def print_summary(
    annotation_path: Path,
    source_root: Path,
    output_dir: Path,
    streams_to_copy: list[str],
    ignore_subject_prefixes: list[str],
    dry_run: bool,
    overwrite: bool,
    stats: CopyStats,
) -> None:
    mode = "DRY RUN" if dry_run else "COPY"
    print("=" * 72)
    print(f"Annotation trial stream copier ({mode})")
    print("=" * 72)
    print(f"Annotation: {annotation_path}")
    print(f"Source root: {source_root}")
    print(f"Output dir:  {output_dir}")
    print(f"Streams:     {', '.join(streams_to_copy)}")
    print(f"Ignore subject prefixes: {', '.join(ignore_subject_prefixes) or 'none'}")
    print(f"Overwrite existing files: {'yes' if overwrite else 'no'}")
    print()

    print("Selection")
    print(f"  Subjects seen:                {stats.subjects_seen}")
    print(f"  Subjects skipped:             {stats.subjects_skipped}")
    print(f"  Scenarios seen:               {stats.scenarios_seen}")
    print(f"  Trials seen:                  {stats.trials_seen}")
    print(f"  Trials skipped by subject:    {stats.trials_skipped_by_subject}")
    print(f"  Trials considered:            {stats.trials_considered}")
    print(f"  Trials with selected streams: {stats.trials_with_selected_streams}")
    print(f"  Trials without selected data: {stats.trials_without_selected_streams}")
    print(f"  Selected stream entries:      {stats.selected_stream_entries}")
    print(f"  Requested streams missing:    {stats.requested_streams_missing}")
    print(f"  Missing file_path fields:     {stats.missing_file_path_fields}")
    print(f"  Invalid relative paths:       {stats.invalid_relative_paths}")
    print(f"  Duplicate relative paths:     {stats.duplicate_relative_paths}")
    print()

    print("Copy Status")
    print(f"  Source files missing:                 {stats.source_files_missing}")
    print(f"  Destination exists, same size:        {stats.destination_exists_same_size}")
    print(f"  Destination exists, different size:   {stats.destination_exists_different_size}")
    if dry_run:
        print(f"  Files that would be copied:           {stats.files_would_copy}")
        print(f"  Files that would be overwritten:      {stats.files_would_overwrite}")
        print(f"  Bytes that would be copied:           {format_bytes(stats.bytes_would_copy)}")
    else:
        print(f"  Files copied:                         {stats.files_copied}")
        print(f"  Files overwritten:                    {stats.files_overwritten}")
        print(f"  Bytes copied:                         {format_bytes(stats.bytes_copied)}")
    print()

    print_counter("Selected entries by stream", stats.entries_by_stream)
    print_counter("Missing requested streams by stream", stats.missing_requested_by_stream)
    print_counter("Missing source files by stream", stats.missing_sources_by_stream)
    if not dry_run:
        print_counter("Copied files by stream", stats.copied_by_stream)
    print()

    print_samples("Sample missing requested streams", stats.samples["missing_requested_streams"])
    print_samples("Sample missing source files", stats.samples["missing_sources"])
    print_samples("Sample invalid relative paths", stats.samples["invalid_relative_paths"])
    print_samples(
        "Sample existing destinations with different size",
        stats.samples["different_size_destinations"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy selected streams for every trial listed in an annotation JSON, "
            "preserving the annotation's relative folder structure."
        )
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=ANNOTATION_JSON,
        help=f"Annotation JSON path. Default: {ANNOTATION_JSON}",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=SOURCE_ROOT,
        help="Dataset root that the annotation file paths are relative to.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Destination root where matching files will be copied.",
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        default=None,
        help=(
            "Stream names to copy. Accepts spaces or commas, e.g. "
            "--streams ego smartwatch_data or --streams ego,smartwatch_data."
        ),
    )
    parser.add_argument(
        "--ignore-subject",
        nargs="+",
        default=None,
        help=(
            "Subject ID prefixes to skip. Example: --ignore-subject P skips P0, "
            "P1, etc. Accepts spaces or commas."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=DRY_RUN,
        help="Print the copy plan and stats without copying files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE,
        help="Overwrite destination files that already exist.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="Maximum number of sample warnings to print for each warning type.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with a non-zero status if requested streams or source files are missing.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    streams_to_copy = parse_list(args.streams) or parse_list(STREAMS_TO_COPY)
    if not streams_to_copy:
        parser.error(
            "No streams specified. Use --streams or edit STREAMS_TO_COPY in the script."
        )

    ignore_subject_prefixes = parse_list(args.ignore_subject)
    if args.ignore_subject is None:
        ignore_subject_prefixes = parse_list(IGNORE_SUBJECT)

    if args.source_root is None:
        parser.error("No source root specified. Use --source-root or edit SOURCE_ROOT.")
    if args.output_dir is None:
        parser.error("No output directory specified. Use --output-dir or edit OUTPUT_DIR.")

    annotation_path = as_path(args.annotation).expanduser().resolve()
    source_root_path, source_root_warning = normalize_cli_path(
        args.source_root, "--source-root"
    )
    output_dir_path, output_dir_warning = normalize_cli_path(
        args.output_dir, "--output-dir"
    )
    source_root = source_root_path.expanduser().resolve()
    output_dir = output_dir_path.expanduser().resolve()
    path_warnings = [
        warning for warning in (source_root_warning, output_dir_warning) if warning
    ]

    data = load_annotation(annotation_path)
    copy_items, stats = collect_copy_items(
        data=data,
        source_root=source_root,
        output_dir=output_dir,
        streams_to_copy=streams_to_copy,
        ignore_subject_prefixes=ignore_subject_prefixes,
        sample_limit=args.sample_limit,
    )
    copy_or_plan_items(
        copy_items=copy_items,
        stats=stats,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        sample_limit=args.sample_limit,
    )
    print_summary(
        annotation_path=annotation_path,
        source_root=source_root,
        output_dir=output_dir,
        streams_to_copy=streams_to_copy,
        ignore_subject_prefixes=ignore_subject_prefixes,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        stats=stats,
    )
    if path_warnings:
        print()
        print("Path Notes")
        for warning in path_warnings:
            print(f"  {warning}")

    if args.strict and (
        stats.requested_streams_missing
        or stats.source_files_missing
        or stats.invalid_relative_paths
        or stats.missing_file_path_fields
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
