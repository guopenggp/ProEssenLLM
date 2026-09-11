#!/usr/bin/env python
"""Build a training-compatible LMDB from protein sequences and ESM embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


DEFAULT_MODEL = "./pretrained_model/esm2_t33_650M_UR50D.pt"
DEFAULT_ALLOWED_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYBXZUO")
FASTA_EXTENSIONS = {".fa", ".fasta", ".faa", ".fas"}
RESERVED_LMDB_KEYS = {"__keys__", "__metadata__"}


def parse_fasta(path: Path):
    """Yield ``(header, sequence)`` pairs from a FASTA file."""
    header = None
    sequence_parts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(sequence_parts)
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Empty FASTA header at line {line_number}")
                sequence_parts = []
            else:
                if header is None:
                    raise ValueError(
                        "Found sequence text before the first FASTA header "
                        f"at line {line_number}"
                    )
                sequence_parts.append("".join(line.split()))
    if header is not None:
        yield header, "".join(sequence_parts)


def parse_fasta_header(header: str) -> tuple[str, str, str, int]:
    """Parse the historical ``index_ID_group_label`` FASTA header format."""
    first_token = header.split()[0]
    parts = first_token.split("_")
    if len(parts) < 4:
        raise ValueError(
            "Invalid FASTA header format. Expected at least index_ID_group_label, "
            f"got: {header}"
        )
    label_text = str(parts[3])
    if not label_text or label_text[0] not in {"0", "1"}:
        raise ValueError(f"FASTA label must start with 0 or 1, got: {parts[3]}")
    return parts[0], parts[1], parts[2], int(label_text[0])


def read_fasta_as_table(path: Path) -> pd.DataFrame:
    records = []
    for auto_index, (header, sequence) in enumerate(parse_fasta(path)):
        original_index, protein_id, group, essential = parse_fasta_header(header)
        records.append(
            {
                "index": auto_index,
                "original_fasta_index": original_index,
                "ID": protein_id,
                "group": group,
                "essential": essential,
                "sequence": sequence,
                "fasta_header": header,
            }
        )
    return pd.DataFrame(
        records,
        columns=[
            "index",
            "original_fasta_index",
            "ID",
            "group",
            "essential",
            "sequence",
            "fasta_header",
        ],
    )


def read_input_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in FASTA_EXTENSIONS:
        return read_fasta_as_table(path)
    if suffix in {".pkl", ".pickle"}:
        loaded = pd.read_pickle(path)
    elif suffix == ".csv":
        loaded = pd.read_csv(path)
    elif suffix in {".tsv", ".txt"}:
        loaded = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unsupported data file suffix: {path.suffix}")
    if not isinstance(loaded, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame, got {type(loaded).__name__}")
    return loaded.copy()


def to_python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def parse_group_value(value: Any) -> str:
    value = to_python_scalar(value)
    if pd.isna(value) or not str(value).strip():
        raise ValueError("Species/group values must not be empty")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def parse_label_value(value: Any) -> int:
    value = to_python_scalar(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Label cannot be converted to 0/1: {value!r}") from exc
    if numeric not in {0.0, 1.0}:
        raise ValueError(f"Label must be 0 or 1, got: {value!r}")
    return int(numeric)


def sequence_hash(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()


def normalize_sequence(
    sequence: Any,
    allowed_residues: set[str],
    invalid_policy: str,
    internal_stop_policy: str,
) -> tuple[str, bool, Counter, dict[str, int]]:
    """Normalize a protein sequence and report all modifications."""
    if not isinstance(sequence, str):
        sequence = "" if pd.isna(sequence) else str(sequence)
    original = sequence
    sequence = re.sub(r"\s+", "", sequence).upper()

    terminal_stop_count = len(sequence) - len(sequence.rstrip("*"))
    if terminal_stop_count:
        sequence = sequence.rstrip("*")

    internal_stop_count = sequence.count("*")
    if internal_stop_count:
        if internal_stop_policy == "skip":
            return "", True, Counter({"*": internal_stop_count}), {
                "terminal_stop_removed": terminal_stop_count,
                "internal_stop_count": internal_stop_count,
            }
        if internal_stop_policy == "remove":
            sequence = sequence.replace("*", "")
        elif internal_stop_policy == "mask":
            sequence = sequence.replace("*", "X")
        else:
            raise ValueError(f"Unsupported internal stop policy: {internal_stop_policy}")

    invalid_characters = Counter(
        character for character in sequence if character not in allowed_residues
    )
    if invalid_characters:
        if invalid_policy == "skip":
            return "", True, invalid_characters, {
                "terminal_stop_removed": terminal_stop_count,
                "internal_stop_count": internal_stop_count,
            }
        if invalid_policy == "remove":
            sequence = "".join(
                character for character in sequence if character in allowed_residues
            )
        elif invalid_policy == "mask":
            sequence = "".join(
                character if character in allowed_residues else "X"
                for character in sequence
            )
        else:
            raise ValueError(f"Unsupported invalid policy: {invalid_policy}")

    return sequence, sequence != original, invalid_characters, {
        "terminal_stop_removed": terminal_stop_count,
        "internal_stop_count": internal_stop_count,
    }


def validate_columns(table: pd.DataFrame, args: argparse.Namespace) -> None:
    if table.empty:
        raise ValueError("Input data contains no records")
    required = [
        args.id_column,
        args.group_column,
        args.label_column,
        args.sequence_column,
    ]
    if args.key_source == "column":
        required.append(args.index_column)
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_arguments(args: argparse.Namespace) -> None:
    if args.truncation_seq_length <= 0:
        raise ValueError("truncation_seq_length must be positive")
    if args.toks_per_batch <= 0:
        raise ValueError("toks_per_batch must be positive")
    if args.commit_every <= 0:
        raise ValueError("commit_every must be positive")
    if args.map_size_gb <= 0:
        raise ValueError("map_size_gb must be positive")


def choose_allowed_residues(alphabet) -> set[str]:
    alphabet_tokens = set(getattr(alphabet, "tok_to_idx", {}).keys())
    residue_tokens = {
        token for token in alphabet_tokens if len(token) == 1 and token.isalpha()
    }
    allowed = residue_tokens.intersection(DEFAULT_ALLOWED_RESIDUES)
    return allowed or DEFAULT_ALLOWED_RESIDUES


def resolve_repr_layer(requested_layer: int, model_num_layers: int) -> int:
    return (requested_layer + model_num_layers + 1) % (model_num_layers + 1)


def make_batches(
    items: list[tuple[str, str]], toks_per_batch: int
) -> list[list[tuple[str, str]]]:
    """Group similar-length sequences under an approximate token budget."""
    batches: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_max_length = 0
    for label, sequence in sorted(items, key=lambda item: len(item[1])):
        sequence_length = len(sequence) + 2
        next_max_length = max(current_max_length, sequence_length)
        next_size = next_max_length * (len(current) + 1)
        if current and next_size > toks_per_batch:
            batches.append(current)
            current = []
            current_max_length = 0
        current.append((label, sequence))
        current_max_length = max(current_max_length, sequence_length)
    if current:
        batches.append(current)
    return batches


def remove_existing_lmdb(path: Path) -> None:
    """Remove only the exact output file and its LMDB lock file."""
    lock_path = Path(str(path) + "-lock")
    if path.exists():
        if path.is_dir():
            raise ValueError(
                f"{path} is a directory; this script writes a single-file LMDB"
            )
        path.unlink()
    if lock_path.exists():
        lock_path.unlink()


def build_records(
    table: pd.DataFrame,
    args: argparse.Namespace,
    allowed_residues: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], list[Any]]:
    sequence_to_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_counter: Counter = Counter()
    observed_keys: set[str] = set()
    kept_indices: list[Any] = []
    skipped = 0
    modified = 0
    terminal_stop_removed_records = 0
    terminal_stop_removed_total = 0
    internal_stop_records = 0
    internal_stop_total = 0

    for source_position, (dataframe_index, row) in enumerate(table.iterrows()):
        raw_key = (
            row[args.index_column]
            if args.key_source == "column"
            else dataframe_index
        )
        key = str(to_python_scalar(raw_key))
        if not key:
            raise ValueError(f"Empty LMDB key at source position {source_position}")
        if key in RESERVED_LMDB_KEYS:
            raise ValueError(f"LMDB key {key!r} is reserved for project metadata")
        if key in observed_keys:
            raise ValueError(f"Duplicate LMDB key after string conversion: {key!r}")
        observed_keys.add(key)

        normalized, changed, invalid_characters, stop_stats = normalize_sequence(
            row[args.sequence_column],
            allowed_residues=allowed_residues,
            invalid_policy=args.invalid_policy,
            internal_stop_policy=args.internal_stop_policy,
        )
        invalid_counter.update(invalid_characters)
        if stop_stats["terminal_stop_removed"]:
            terminal_stop_removed_records += 1
            terminal_stop_removed_total += stop_stats["terminal_stop_removed"]
        if stop_stats["internal_stop_count"]:
            internal_stop_records += 1
            internal_stop_total += stop_stats["internal_stop_count"]
        if changed or invalid_characters:
            modified += 1
        if not normalized:
            skipped += 1
            continue

        record = {
            "key": key,
            "source_position": int(source_position),
            "source_dataframe_index": to_python_scalar(dataframe_index),
            "row_index_column": to_python_scalar(
                row.get(args.index_column, dataframe_index)
            ),
            "id": str(row[args.id_column]),
            "label": parse_label_value(row[args.label_column]),
            "group": parse_group_value(row[args.group_column]),
            "sequence_hash": sequence_hash(normalized),
            "sequence_length": int(len(normalized)),
            "sequence_modified": bool(changed or invalid_characters),
            "invalid_chars": dict(invalid_characters),
            "terminal_stop_removed": int(stop_stats["terminal_stop_removed"]),
            "internal_stop_count": int(stop_stats["internal_stop_count"]),
        }
        sequence_to_records[normalized].append(record)
        kept_indices.append(dataframe_index)

    stats = {
        "total_rows": int(len(table)),
        "usable_rows": int(
            sum(len(records) for records in sequence_to_records.values())
        ),
        "skipped_rows": int(skipped),
        "rows_with_sequence_changes_or_invalid_chars": int(modified),
        "unique_sequences_to_embed": int(len(sequence_to_records)),
        "invalid_char_counts": dict(sorted(invalid_counter.items())),
        "terminal_stop_removed_records": int(terminal_stop_removed_records),
        "terminal_stop_removed_total": int(terminal_stop_removed_total),
        "internal_stop_records": int(internal_stop_records),
        "internal_stop_total": int(internal_stop_total),
    }
    if not sequence_to_records:
        raise ValueError("No usable protein sequences remain after normalization")
    return sequence_to_records, stats, kept_indices


def build_companion_dataframe(
    table: pd.DataFrame,
    kept_indices: list[Any],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Return metadata with an explicit key matching each LMDB record."""
    filtered = table.loc[kept_indices].copy()
    if args.key_source == "column":
        filtered["lmdb_key"] = filtered[args.index_column].map(
            lambda value: str(to_python_scalar(value))
        )
    else:
        filtered["lmdb_key"] = [str(to_python_scalar(value)) for value in filtered.index]
    return filtered


def write_companion_tables(table: pd.DataFrame, output_lmdb: Path) -> None:
    output_prefix = output_lmdb.with_suffix("")
    pickle_path = output_prefix.with_suffix(".metadata_table.pkl")
    csv_path = output_prefix.with_suffix(".metadata_table.csv")
    table.to_pickle(pickle_path)
    table.to_csv(csv_path, index=False)
    print(f"Wrote companion metadata table: {pickle_path}")
    print(f"Wrote companion metadata table: {csv_path}")


def write_metadata_file(
    output_lmdb: Path,
    stats: dict[str, Any],
    duplicate_summary: dict[str, Any],
    feature_length: int,
    resolved_layer: int,
    args: argparse.Namespace,
) -> Path:
    metadata_path = output_lmdb.with_suffix(output_lmdb.suffix + ".metadata.json")
    metadata = {
        "output_lmdb": str(output_lmdb.resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "model_name": args.model_name,
        "feature_length": int(feature_length),
        "label_column": args.label_column,
        "group_column": args.group_column,
        "sequence_column": args.sequence_column,
        "id_column": args.id_column,
        "index_column": args.index_column,
        "key_source": args.key_source,
        "invalid_policy": args.invalid_policy,
        "internal_stop_policy": args.internal_stop_policy,
        "requested_repr_layer": int(args.repr_layer),
        "resolved_repr_layer": int(resolved_layer),
        "truncation_seq_length": int(args.truncation_seq_length),
        "stats": stats,
        "duplicate_summary": duplicate_summary,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata_path


def load_esm_model(model_name: str):
    """Load fair-esm lazily so ``--help`` works without optional dependencies."""
    try:
        from esm import MSATransformer, pretrained
    except ImportError as exc:
        raise ImportError(
            "fair-esm is required to build features; install "
            "requirements-feature-builder.txt"
        ) from exc
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    if isinstance(model, MSATransformer):
        raise ValueError("MSA Transformer models are not supported")
    return model, alphabet


def summarize_duplicates(
    sequence_to_records: dict[str, list[dict[str, Any]]],
    usable_rows: int,
) -> dict[str, int]:
    duplicate_groups = [
        records for records in sequence_to_records.values() if len(records) > 1
    ]
    label_conflicts = [
        records
        for records in duplicate_groups
        if len({record["label"] for record in records}) > 1
    ]
    cross_species = [
        records
        for records in duplicate_groups
        if len({record["group"] for record in records}) > 1
    ]
    return {
        "duplicate_sequence_groups": int(len(duplicate_groups)),
        "duplicate_records_extra": int(usable_rows - len(sequence_to_records)),
        "duplicate_groups_with_label_conflict": int(len(label_conflicts)),
        "duplicate_groups_across_groups": int(len(cross_species)),
    }


def write_lmdb(
    output_lmdb: Path,
    sequence_to_records: dict[str, list[dict[str, Any]]],
    stats: dict[str, Any],
    duplicate_summary: dict[str, Any],
    model,
    alphabet,
    device: torch.device,
    resolved_layer: int,
    feature_length: int,
    args: argparse.Namespace,
) -> int:
    map_size = int(args.map_size_gb * 1024**3)
    environment = lmdb.open(str(output_lmdb), subdir=False, map_size=map_size)
    transaction = environment.begin(write=True)
    keys_written: list[str] = []
    records_written = 0

    unique_items = [
        (f"seq_{index}", sequence)
        for index, sequence in enumerate(sequence_to_records)
    ]
    label_to_sequence = dict(unique_items)
    batches = make_batches(unique_items, args.toks_per_batch)
    batch_converter = alphabet.get_batch_converter(
        truncation_seq_length=args.truncation_seq_length
    )

    print(f"Writing LMDB: {output_lmdb}")
    try:
        with torch.no_grad():
            for batch in tqdm(batches, desc="Embedding batches"):
                labels = [label for label, _sequence in batch]
                sequences = [sequence for _label, sequence in batch]
                batch_labels, batch_strings, tokens = batch_converter(
                    list(zip(labels, sequences))
                )
                tokens = tokens.to(device=device, non_blocking=True)
                output = model(
                    tokens,
                    repr_layers=[resolved_layer],
                    return_contacts=False,
                )
                representations = output["representations"][resolved_layer].detach().cpu()
                if representations.size(-1) != feature_length:
                    raise ValueError(
                        "Model representation width changed unexpectedly: "
                        f"expected {feature_length}, got {representations.size(-1)}"
                    )

                for batch_index, batch_label in enumerate(batch_labels):
                    sequence = label_to_sequence[batch_label]
                    retained_length = min(
                        args.truncation_seq_length, len(batch_strings[batch_index])
                    )
                    feature = (
                        representations[batch_index, 1 : retained_length + 1]
                        .contiguous()
                        .numpy()
                        .astype(np.float32)
                    )
                    if not feature.shape[0]:
                        raise ValueError(f"ESM returned an empty feature for {batch_label}")

                    for record in sequence_to_records[sequence]:
                        sample = {
                            "feature": feature,
                            "label": record["label"],
                            "group": record["group"],
                            "id": record["id"],
                            "index": record["key"],
                            "source_position": record["source_position"],
                            "source_dataframe_index": record[
                                "source_dataframe_index"
                            ],
                            "row_index_column": record["row_index_column"],
                            "sequence_hash": record["sequence_hash"],
                            "sequence_length": record["sequence_length"],
                            "stored_feature_length": int(feature.shape[0]),
                            "sequence_modified": record["sequence_modified"],
                            "invalid_chars": record["invalid_chars"],
                            "terminal_stop_removed": record[
                                "terminal_stop_removed"
                            ],
                            "internal_stop_count": record["internal_stop_count"],
                        }
                        key_bytes = record["key"].encode("utf-8", errors="strict")
                        if not transaction.put(
                            key_bytes,
                            pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL),
                            overwrite=False,
                        ):
                            raise ValueError(f"Duplicate LMDB key: {record['key']!r}")
                        keys_written.append(record["key"])
                        records_written += 1
                        if records_written % args.commit_every == 0:
                            transaction.commit()
                            transaction = environment.begin(write=True)

        transaction.put(
            b"__keys__",
            pickle.dumps(keys_written, protocol=pickle.HIGHEST_PROTOCOL),
        )
        transaction.put(
            b"__metadata__",
            pickle.dumps(
                {
                    "stats": stats,
                    "duplicate_summary": duplicate_summary,
                    "repr_layer": int(resolved_layer),
                    "feature_length": int(feature_length),
                    "truncation_seq_length": int(args.truncation_seq_length),
                    "model_name": args.model_name,
                },
                protocol=pickle.HIGHEST_PROTOCOL,
            ),
        )
        transaction.commit()
        environment.sync()
    except Exception:
        transaction.abort()
        raise
    finally:
        environment.close()
    return records_written


def build_lmdb(args: argparse.Namespace) -> None:
    validate_arguments(args)
    data_path = Path(args.data_path)
    output_lmdb = Path(args.output_lmdb)
    output_lmdb.parent.mkdir(parents=True, exist_ok=True)

    lock_path = Path(str(output_lmdb) + "-lock")
    if args.overwrite:
        remove_existing_lmdb(output_lmdb)
    elif output_lmdb.exists() or lock_path.exists():
        raise FileExistsError(
            f"Output LMDB already exists: {output_lmdb}. "
            "Use --overwrite to replace it."
        )

    print(f"Reading input data: {data_path}")
    table = read_input_table(data_path)
    validate_columns(table, args)
    print(f"Rows: {len(table)}")
    print(f"Columns: {list(table.columns)}")

    print(f"Loading ESM model: {args.model_name}")
    model, alphabet = load_esm_model(args.model_name)
    model.eval()
    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is unavailable; falling back from {requested_device} to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)
    model = model.to(device=device)
    resolved_layer = resolve_repr_layer(args.repr_layer, int(model.num_layers))
    feature_length = int(model.embed_dim)
    allowed_residues = choose_allowed_residues(alphabet)
    print(f"Device: {device}")
    print(f"Using representation layer: {resolved_layer}")
    print(f"Feature dimension: {feature_length}")
    print(f"Allowed residue tokens: {''.join(sorted(allowed_residues))}")

    sequence_to_records, stats, kept_indices = build_records(
        table, args, allowed_residues
    )
    duplicate_summary = summarize_duplicates(
        sequence_to_records, int(stats["usable_rows"])
    )
    print("Input sequence summary")
    for name in (
        "usable_rows",
        "skipped_rows",
        "unique_sequences_to_embed",
        "terminal_stop_removed_records",
        "internal_stop_records",
        "invalid_char_counts",
    ):
        print(f"  {name}: {stats[name]}")
    for name, value in duplicate_summary.items():
        print(f"  {name}: {value}")

    records_written = write_lmdb(
        output_lmdb=output_lmdb,
        sequence_to_records=sequence_to_records,
        stats=stats,
        duplicate_summary=duplicate_summary,
        model=model,
        alphabet=alphabet,
        device=device,
        resolved_layer=resolved_layer,
        feature_length=feature_length,
        args=args,
    )
    stats["records_written"] = int(records_written)
    companion_table = build_companion_dataframe(table, kept_indices, args)
    if args.write_companion_table:
        write_companion_tables(companion_table, output_lmdb)
    metadata_path = write_metadata_file(
        output_lmdb,
        stats,
        duplicate_summary,
        feature_length,
        resolved_layer,
        args,
    )
    print("Done")
    print(f"  records_written: {records_written}")
    print(f"  output_lmdb: {output_lmdb}")
    print(f"  metadata_json: {metadata_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a ProEssenLLM-compatible LMDB from FASTA or tabular "
            "protein sequences using a pretrained ESM model."
        )
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Input FASTA, pickle, CSV, TSV, or TXT file",
    )
    parser.add_argument(
        "--output_lmdb",
        required=True,
        help="Output single-file LMDB path",
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL,
        help="Local ESM checkpoint path or fair-esm model name",
    )
    parser.add_argument("--sequence_column", default="sequence")
    parser.add_argument("--label_column", default="essential")
    parser.add_argument("--group_column", default="group")
    parser.add_argument("--id_column", default="ID")
    parser.add_argument("--index_column", default="index")
    parser.add_argument(
        "--key_source",
        choices=["dataframe_index", "column"],
        default="dataframe_index",
        help="Use the DataFrame index or --index_column as each LMDB key",
    )
    parser.add_argument(
        "--invalid_policy",
        choices=["skip", "remove", "mask"],
        default="mask",
    )
    parser.add_argument(
        "--internal_stop_policy",
        choices=["skip", "remove", "mask"],
        default="mask",
    )
    parser.add_argument(
        "--repr_layer",
        type=int,
        default=-1,
        help="Representation layer; negative values count from the end",
    )
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1000,
        help="Maximum number of residues embedded per protein",
    )
    parser.add_argument(
        "--toks_per_batch",
        type=int,
        default=4096,
        help="Approximate padded-token budget per inference batch",
    )
    parser.add_argument("--map_size_gb", type=float, default=64.0)
    parser.add_argument("--commit_every", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--write_companion_table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write filtered metadata as pickle and CSV next to the LMDB",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the exact output LMDB and lock file if they exist",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    build_lmdb(parse_args())
