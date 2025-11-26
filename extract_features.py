#!/usr/bin/env python3
"""
LLM-powered feature extraction pipeline for follicular lymphoma cohorts.

Enhancements over the baseline version:
- Groups multiple visit JSON files by patient, ordered by visit time.
- Calls DeepSeek for every visit and appends visit index suffixes (T1/T2/...) per feature.
- Emits both a consolidated JSON artifact and a wide CSV table that analysts can consume directly.
- Feature names stay separated from downstream column aliases so you can map `SUVmax` -> `Suvmax_Tn`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


DEFAULT_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-chat"
DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d",
]
ALIAS_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass
class FeatureSpec:
    """Normalized representation of a feature definition."""

    name: str
    alias: str
    kind: str  # "continuous" or "categorical"


@dataclass
class VisitMeta:
    """Metadata describing a single visit JSON file."""

    patient_id: str
    patient_name: str
    visit_id: str
    enter_time_raw: str
    enter_time: datetime
    file_path: Path
    record: Optional[Dict[str, Any]]


def read_json(path: Path) -> Any:
    """Read a JSON document from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: Any, path: Path) -> None:
    """Persist a JSON-serializable object to disk."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=4)


def write_csv(fieldnames: List[str], rows: List[Dict[str, Any]], path: Path) -> None:
    """Serialize `rows` into a CSV file with UTF-8 encoding."""
    if not fieldnames:
        raise ValueError("CSV requires at least one column.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def merge_text(record: Dict[str, Any]) -> str:
    """Flatten nested medical records into a long text block."""

    def walk(node: Any, prefix: str = "") -> List[str]:
        lines: List[str] = []
        if node is None:
            return lines

        if isinstance(node, dict):
            for key, value in node.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                lines.extend(walk(value, new_prefix))
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                new_prefix = f"{prefix}[{idx}]"
                lines.extend(walk(value, new_prefix))
        else:
            lines.append(f"{prefix}: {node}")
        return lines

    return "\n".join(walk(record))


def build_prompt(features: Dict[str, List[FeatureSpec]], narrative: str) -> str:
    """Create the user prompt that instructs DeepSeek to extract the target features."""
    cont = "\n".join(f"- {item.name}" for item in features["continuous"])
    cate = "\n".join(f"- {item.name}" for item in features["categorical"])
    template = f"""
You are an information extraction assistant for follicular lymphoma patient timelines.
Use ONLY the provided clinical narrative to extract the requested features.

Continuous features:
{cont}

Categorical features:
{cate}

Return strictly valid JSON using the schema:
{{
  "continuous_features": {{
    "<feature name>": {{"value": <float or null>, "evidence": "<direct quote>"}}
  }},
  "categorical_features": {{
    "<feature name>": {{"value": "<string or null>", "evidence": "<direct quote>"}}
  }}
}}

Rules:
- Keep every feature in the response even if the evidence is missing (use null).
- Evidence must be a verbatim excerpt, trimmed to <= 40 characters when possible.
- If multiple values appear, prefer the one closest to the visit start unless another timepoint is requested.
- Do NOT add extra commentary or Markdown, JSON only.

Clinical narrative:
\"\"\"{narrative}\"\"\"
"""
    return template.strip()


def call_deepseek(
    api_key: str,
    model: str,
    prompt: str,
    api_url: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Send a chat completion request to DeepSeek and return the parsed JSON payload."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You extract medical features and always answer with compact JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    response = requests.post(api_url, headers=headers, json=body, timeout=120)
    response.raise_for_status()
    payload = response.json()
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"DeepSeek payload missing choices: {payload}") from exc

    structured = json.loads(content)
    structured.setdefault("continuous_features", {})
    structured.setdefault("categorical_features", {})
    return structured


def extract_for_visit(
    visit: VisitMeta,
    features: Dict[str, List[FeatureSpec]],
    api_key: str,
    model: str,
    api_url: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_backoff: float,
) -> Dict[str, Any]:
    """Run the end-to-end extraction for a single visit file."""
    raw_record = visit.record if visit.record is not None else read_json(visit.file_path)
    narrative = merge_text(raw_record)
    prompt = build_prompt(features, narrative)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            structured = call_deepseek(
                api_key=api_key,
                model=model,
                prompt=prompt,
                api_url=api_url,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return structured
        except (requests.HTTPError, requests.ConnectionError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep_for = retry_backoff * attempt
            time.sleep(sleep_for)

    raise RuntimeError(f"Extraction failed for {visit.file_path}: {last_error}") from last_error


def collect_input_paths(files: Sequence[str] | None, dirs: Sequence[str] | None) -> List[Path]:
    """Gather a deduplicated, sorted list of JSON files coming from CLI arguments."""
    candidates: List[Path] = []
    for item in files or []:
        path = Path(item).expanduser()
        if path.is_file():
            candidates.append(path)
    for directory in dirs or []:
        base = Path(directory).expanduser()
        if not base.is_dir():
            continue
        for json_path in base.rglob("*.json"):
            if json_path.is_file():
                candidates.append(json_path)
    unique: List[Path] = []
    seen: set[str] = set()
    for path in sorted(candidates):
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def parse_visit_time(value: Optional[str]) -> datetime:
    """Parse multiple date formats, falling back to datetime.min."""
    if not value:
        return datetime.min
    stripped = value.strip()
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(stripped, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(stripped)
    except ValueError:
        return datetime.min


def first_non_empty(*values: Optional[str]) -> str:
    """Return the first value that is truthy once stripped."""
    for val in values:
        if val:
            stripped = str(val).strip()
            if stripped:
                return stripped
    return ""


def extract_visit_metadata(path: Path) -> VisitMeta:
    """Read a visit file once to capture identifier metadata."""
    payload = read_json(path)
    basic = payload.get("basic_information", {}) or {}
    enter = payload.get("enter_record", {}) or {}
    exit_record = payload.get("exit_record", {}) or {}

    patient_id = first_non_empty(
        basic.get("document_id"),
        basic.get("visit_id"),
        enter.get("visit_id"),
        exit_record.get("visit_id"),
        path.stem,
    )
    visit_id = first_non_empty(
        enter.get("visit_id"),
        exit_record.get("visit_id"),
        basic.get("visit_id"),
        f"{patient_id}_{path.stem}",
    )
    patient_name = first_non_empty(
        basic.get("name"),
        enter.get("name"),
        exit_record.get("name"),
    )
    enter_time_raw = first_non_empty(
        enter.get("enter_time"),
        basic.get("enter_time"),
        exit_record.get("enter_time"),
    )
    enter_time = parse_visit_time(enter_time_raw)

    return VisitMeta(
        patient_id=patient_id or path.stem,
        patient_name=patient_name,
        visit_id=visit_id or path.stem,
        enter_time_raw=enter_time_raw,
        enter_time=enter_time,
        file_path=path,
        record=payload,
    )


def group_by_patient(visits: List[VisitMeta]) -> Dict[str, List[VisitMeta]]:
    """Group visit metadata by patient and sort each bucket chronologically."""
    grouped: Dict[str, List[VisitMeta]] = {}
    for visit in visits:
        grouped.setdefault(visit.patient_id, []).append(visit)
    for bucket in grouped.values():
        bucket.sort(key=lambda item: (item.enter_time, item.enter_time_raw, item.file_path.name))
    return dict(sorted(grouped.items(), key=lambda kv: kv[0]))


def auto_alias(name: str, fallback: str) -> str:
    """Generate a CamelCase alias from any feature name."""
    tokens = ALIAS_TOKEN_RE.findall(name)
    if not tokens:
        tokens = ALIAS_TOKEN_RE.findall(fallback)
    if not tokens:
        return fallback
    return "".join(token.capitalize() for token in tokens)


def normalize_feature_specs(config: Dict[str, Any]) -> Dict[str, List[FeatureSpec]]:
    """Normalize both continuous and categorical feature lists."""

    def normalize_one(items: Sequence[Any], kind: str) -> List[FeatureSpec]:
        specs: List[FeatureSpec] = []
        for idx, item in enumerate(items):
            if isinstance(item, str):
                name = item
                alias = auto_alias(name, f"{kind.title()}Feature{idx}")
            elif isinstance(item, dict):
                if "name" not in item:
                    raise ValueError(f"Feature entry missing 'name': {item}")
                name = item["name"]
                alias_hint = item.get("alias") or item.get("key") or item.get("column") or ""
                alias = alias_hint.strip() or auto_alias(name, f"{kind.title()}Feature{idx}")
            else:
                raise TypeError(f"Unsupported feature entry: {item}")
            specs.append(FeatureSpec(name=name, alias=alias, kind=kind))
        return specs

    required_keys = {"continuous_features", "categorical_features"}
    missing = required_keys - set(config.keys())
    if missing:
        raise ValueError(f"Feature config missing keys: {missing}")

    features = {
        "continuous": normalize_one(config["continuous_features"], "continuous"),
        "categorical": normalize_one(config["categorical_features"], "categorical"),
    }

    aliases = {}
    for bucket in features.values():
        for spec in bucket:
            if spec.alias in aliases:
                raise ValueError(
                    f"Duplicate alias '{spec.alias}' for features '{spec.name}' and '{aliases[spec.alias]}'"
                )
            aliases[spec.alias] = spec.name
    return features


def build_csv_rows(
    patient_results: Dict[str, Any],
    features: Dict[str, List[FeatureSpec]],
    include_evidence: bool,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Convert nested patient results into CSV columns."""
    specs_all = features["continuous"] + features["categorical"]
    max_visits = max((data["total_visits"] for data in patient_results.values()), default=0)

    fieldnames: List[str] = ["patient_id", "patient_name", "total_visits"]
    for idx in range(1, max_visits + 1):
        fieldnames.extend([f"visit_id_T{idx}", f"visit_time_T{idx}"])
        for spec in specs_all:
            fieldnames.append(f"{spec.alias}_T{idx}")
            if include_evidence:
                fieldnames.append(f"{spec.alias}_T{idx}_evidence")

    rows: List[Dict[str, Any]] = []
    for patient_id, data in patient_results.items():
        row: Dict[str, Any] = {
            "patient_id": patient_id,
            "patient_name": data.get("patient_name", ""),
            "total_visits": data["total_visits"],
        }
        for visit in data["visits"]:
            idx = visit["visit_index"]
            row[f"visit_id_T{idx}"] = visit["visit_id"]
            row[f"visit_time_T{idx}"] = visit["enter_time"]
            features_payload = visit["features"]
            for spec in specs_all:
                section = f"{spec.kind}_features"
                section_payload = features_payload.get(section, {})
                cell = section_payload.get(spec.name, {})
                row[f"{spec.alias}_T{idx}"] = cell.get("value")
                if include_evidence:
                    row[f"{spec.alias}_T{idx}_evidence"] = cell.get("evidence")
        rows.append(row)

    return fieldnames, rows


def aggregate_patients(
    grouped_visits: Dict[str, List[VisitMeta]],
    features: Dict[str, List[FeatureSpec]],
    api_key: str,
    model: str,
    api_url: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_backoff: float,
) -> Dict[str, Any]:
    """Run extraction for every visit and aggregate outputs per patient."""
    results: Dict[str, Any] = {}
    for patient_id, visits in grouped_visits.items():
        patient_visits: List[Dict[str, Any]] = []
        for idx, visit in enumerate(visits, start=1):
            structured = extract_for_visit(
                visit=visit,
                features=features,
                api_key=api_key,
                model=model,
                api_url=api_url,
                temperature=temperature,
                max_tokens=max_tokens,
                retries=retries,
                retry_backoff=retry_backoff,
            )
            patient_visits.append(
                {
                    "visit_index": idx,
                    "visit_id": visit.visit_id,
                    "enter_time": visit.enter_time_raw,
                    "source_file": str(visit.file_path),
                    "features": structured,
                }
            )
            visit.record = None  # free memory

        patient_name = next((v.patient_name for v in visits if v.patient_name), "")
        results[patient_id] = {
            "patient_name": patient_name,
            "total_visits": len(visits),
            "visits": patient_visits,
        }
    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="LLM-powered feature extractor for follicular lymphoma cohorts."
    )
    parser.add_argument(
        "--patient-file",
        action="append",
        help="Absolute path to a patient JSON file. Provide multiple times for batching.",
    )
    parser.add_argument(
        "--patient-dir",
        action="append",
        help="Directory containing patient JSON files (searched recursively).",
    )
    parser.add_argument(
        "--config",
        default="features_config.json",
        help="Absolute path to the feature configuration JSON file.",
    )
    parser.add_argument(
        "--output-json",
        default="extracted_features.json",
        help="Where to store the aggregated JSON results.",
    )
    parser.add_argument(
        "--output-csv",
        default="extracted_features.csv",
        help="Where to store the flattened CSV table.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"DeepSeek model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Override the DeepSeek REST endpoint if needed.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature for the chat completion request.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="Max tokens for the LLM response.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retry attempts on API failures.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=2.5,
        help="Base seconds to wait between retries (scaled by attempt count).",
    )
    parser.add_argument(
        "--omit-evidence",
        action="store_true",
        help="If set, omit *_evidence columns from the CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing DEEPSEEK_API_KEY environment variable.")

    config_path = Path(args.config).expanduser()
    config = read_json(config_path)
    features = normalize_feature_specs(config)

    patient_paths = collect_input_paths(args.patient_file, args.patient_dir)
    if not patient_paths:
        raise ValueError("No patient JSON files detected. Use --patient-file or --patient-dir.")

    visit_metadata = [extract_visit_metadata(path) for path in patient_paths]
    grouped_visits = group_by_patient(visit_metadata)

    patient_results = aggregate_patients(
        grouped_visits=grouped_visits,
        features=features,
        api_key=api_key,
        model=args.model,
        api_url=args.api_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
    )

    json_output_path = Path(args.output_json).expanduser()
    write_json(patient_results, json_output_path)

    csv_fieldnames, csv_rows = build_csv_rows(
        patient_results=patient_results,
        features=features,
        include_evidence=not args.omit_evidence,
    )
    csv_output_path = Path(args.output_csv).expanduser()
    write_csv(csv_fieldnames, csv_rows, csv_output_path)


if __name__ == "__main__":
    main()

