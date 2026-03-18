#!/usr/bin/env python3
"""
Validate SWE-bench trajectory data pushed to the repository.

Checks:
  1. Directory structure: Week*/trajectories/SWE/<task_id>/<model>/run_<N>/
  2. Model folder names are in the allowed list
  3. File presence per run folder
  4. JSON / JSONL syntax validation
  5. Schema validation for config.json, metadata.json, output.report.json, pass_at_8_summary.json
  6. Git patch exists in every entry of output.jsonl
  7. output.report.json has 0 error instance IDs
  8. conversations/ folder contains at least one .tar.gz file
  9. pass_at_8_summary.json exists when all 8 runs are present
"""

import json
import glob
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ─── Configuration ────────────────────────────────────────────────────────────

ALLOWED_MODELS = [
    "claude",
    "gemini",
    # Add new models here as needed:
    # "gpt",
    # "deepseek",
    # "llama",
]

MAX_RUNS = 8

EXPECTED_RUN_FILES = [
    "config.json",
    "metadata.json",
    "output.jsonl",
    "output_converted.jsonl",
    "output.report.json",
]

# Directories expected inside each run folder
EXPECTED_RUN_DIRS = [
    "conversations",
    "eval_files",
]

# ─── Schema definitions (required top-level keys) ────────────────────────────

CONFIG_REQUIRED_KEYS = [
    "mode",
    "workdir",
    "patch_files",
    "dataset_files",
    "output_dir",
    "log_dir",
    "max_workers",
]

METADATA_REQUIRED_KEYS = [
    "llm",
    "dataset",
    "max_iterations",
    "eval_output_dir",
]

METADATA_LLM_REQUIRED_KEYS = [
    "model",
]

OUTPUT_REPORT_REQUIRED_KEYS = [
    "total_instances",
    "submitted_instances",
    "completed_instances",
    "incomplete_instances",
    "resolved_instances",
    "unresolved_instances",
    "empty_patch_instances",
    "error_instances",
    "error_ids",
]

PASS_AT_8_REQUIRED_KEYS = [
    "metric",
    "k",
    "model",
    "dataset",
    "total_instances",
    "pass_at_k",
    "per_run",
    "per_instance",
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

class ValidationError:
    def __init__(self, path: str, message: str, severity: str = "ERROR"):
        self.path = path
        self.message = message
        self.severity = severity  # ERROR or WARNING

    def __str__(self):
        return f"[{self.severity}] {self.path}: {self.message}"


def load_json(filepath: str) -> tuple[Optional[dict], Optional[str]]:
    """Load and parse a JSON file. Returns (data, error_message)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    except FileNotFoundError:
        return None, "File not found"
    except Exception as e:
        return None, f"Error reading file: {e}"


def load_jsonl(filepath: str) -> tuple[Optional[list[dict]], Optional[str]]:
    """Load and parse a JSONL file. Returns (list_of_dicts, error_message)."""
    entries = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    return None, f"Invalid JSON on line {i}: {e}"
        return entries, None
    except FileNotFoundError:
        return None, "File not found"
    except Exception as e:
        return None, f"Error reading file: {e}"


def check_required_keys(data: dict, required_keys: list[str], label: str) -> list[str]:
    """Check that all required keys exist in a dict. Returns list of missing keys."""
    missing = [k for k in required_keys if k not in data]
    return missing


# ─── Path parsing ─────────────────────────────────────────────────────────────

# Pattern: Week*/trajectories/SWE/<task_id>/<model>/run_<N>/...
RUN_PATH_RE = re.compile(
    r"^(Week\d+/trajectories/SWE/[^/]+/([^/]+)/run_(\d+))(/.*)?$"
)

# Pattern for model-level (where pass_at_8_summary.json lives)
MODEL_PATH_RE = re.compile(
    r"^(Week\d+/trajectories/SWE/[^/]+/([^/]+))(/[^/]+)?$"
)


def extract_run_dirs_from_changed_files(changed_files: list[str]) -> dict[str, dict]:
    """
    From a list of changed file paths, extract unique run directories
    and model directories that need validation.

    Returns:
        {
            "run_dirs": { "path/to/run_1": {"model": "claude", "run_num": 1}, ... },
            "model_dirs": { "path/to/claude": "claude", ... },
        }
    """
    run_dirs = {}
    model_dirs = {}

    for fpath in changed_files:
        fpath = fpath.strip()
        if not fpath:
            continue

        m = RUN_PATH_RE.match(fpath)
        if m:
            run_dir = m.group(1)
            model = m.group(2)
            run_num = int(m.group(3))
            run_dirs[run_dir] = {"model": model, "run_num": run_num}
            # Also track the model dir
            model_dir = os.path.dirname(run_dir)
            model_dirs[model_dir] = model
        else:
            # Could be a model-level file like pass_at_8_summary.json
            m2 = MODEL_PATH_RE.match(fpath)
            if m2:
                model_dir = m2.group(1)
                model = m2.group(2)
                model_dirs[model_dir] = model

    return {"run_dirs": run_dirs, "model_dirs": model_dirs}


# ─── Validators ───────────────────────────────────────────────────────────────

def validate_model_name(model: str, path: str) -> list[ValidationError]:
    """Check model folder name is in the allowed list."""
    errors = []
    if model not in ALLOWED_MODELS:
        errors.append(ValidationError(
            path,
            f"Model '{model}' is not in allowed list: {ALLOWED_MODELS}. "
            f"Update ALLOWED_MODELS in validate_trajectories.py if this is intentional."
        ))
    return errors


def validate_run_file_presence(run_dir: str) -> list[ValidationError]:
    """Check that all expected files and directories exist in a run folder."""
    errors = []

    for fname in EXPECTED_RUN_FILES:
        fpath = os.path.join(run_dir, fname)
        if not os.path.isfile(fpath):
            errors.append(ValidationError(fpath, "Required file is missing"))

    for dname in EXPECTED_RUN_DIRS:
        dpath = os.path.join(run_dir, dname)
        if not os.path.isdir(dpath):
            errors.append(ValidationError(dpath, "Required directory is missing"))

    return errors


def validate_conversations_tar(run_dir: str) -> list[ValidationError]:
    """Check that conversations/ folder contains at least one .tar.gz file."""
    errors = []
    conv_dir = os.path.join(run_dir, "conversations")
    if not os.path.isdir(conv_dir):
        return errors  # Already caught by file presence check

    tar_files = glob.glob(os.path.join(conv_dir, "*.tar.gz"))
    if not tar_files:
        errors.append(ValidationError(
            conv_dir,
            "conversations/ folder must contain at least one .tar.gz file"
        ))
    return errors


def validate_config_json(run_dir: str) -> list[ValidationError]:
    """Validate config.json schema."""
    errors = []
    fpath = os.path.join(run_dir, "config.json")
    data, err = load_json(fpath)

    if err:
        errors.append(ValidationError(fpath, err))
        return errors

    if not isinstance(data, dict):
        errors.append(ValidationError(fpath, "Expected a JSON object at top level"))
        return errors

    missing = check_required_keys(data, CONFIG_REQUIRED_KEYS, "config.json")
    if missing:
        errors.append(ValidationError(fpath, f"Missing required keys: {missing}"))

    return errors


def validate_metadata_json(run_dir: str) -> list[ValidationError]:
    """Validate metadata.json schema."""
    errors = []
    fpath = os.path.join(run_dir, "metadata.json")
    data, err = load_json(fpath)

    if err:
        errors.append(ValidationError(fpath, err))
        return errors

    if not isinstance(data, dict):
        errors.append(ValidationError(fpath, "Expected a JSON object at top level"))
        return errors

    missing = check_required_keys(data, METADATA_REQUIRED_KEYS, "metadata.json")
    if missing:
        errors.append(ValidationError(fpath, f"Missing required keys: {missing}"))

    # Validate nested llm object
    llm = data.get("llm")
    if llm is not None:
        if not isinstance(llm, dict):
            errors.append(ValidationError(fpath, "'llm' field must be a JSON object"))
        else:
            missing_llm = check_required_keys(llm, METADATA_LLM_REQUIRED_KEYS, "llm")
            if missing_llm:
                errors.append(ValidationError(fpath, f"Missing required keys in 'llm': {missing_llm}"))

    return errors


def validate_output_report_json(run_dir: str) -> list[ValidationError]:
    """Validate output.report.json schema and check 0 error instances."""
    errors = []
    fpath = os.path.join(run_dir, "output.report.json")
    data, err = load_json(fpath)

    if err:
        errors.append(ValidationError(fpath, err))
        return errors

    if not isinstance(data, dict):
        errors.append(ValidationError(fpath, "Expected a JSON object at top level"))
        return errors

    missing = check_required_keys(data, OUTPUT_REPORT_REQUIRED_KEYS, "output.report.json")
    if missing:
        errors.append(ValidationError(fpath, f"Missing required keys: {missing}"))

    # Check 0 error instance IDs
    error_ids = data.get("error_ids", [])
    if isinstance(error_ids, list) and len(error_ids) > 0:
        errors.append(ValidationError(
            fpath,
            f"error_ids must be empty but found {len(error_ids)} entries: {error_ids}"
        ))

    error_instances = data.get("error_instances", 0)
    if isinstance(error_instances, (int, float)) and error_instances != 0:
        errors.append(ValidationError(
            fpath,
            f"error_instances must be 0 but found {error_instances}"
        ))

    return errors


def validate_output_jsonl(run_dir: str) -> list[ValidationError]:
    """Validate output.jsonl: parseable JSONL + every entry has a non-empty git_patch."""
    errors = []
    fpath = os.path.join(run_dir, "output.jsonl")
    entries, err = load_jsonl(fpath)

    if err:
        errors.append(ValidationError(fpath, err))
        return errors

    if not entries:
        errors.append(ValidationError(fpath, "output.jsonl is empty (no entries)"))
        return errors

    for i, entry in enumerate(entries):
        instance_id = entry.get("instance_id", f"entry_{i}")

        # Check test_result.git_patch exists and is non-empty
        test_result = entry.get("test_result")
        if test_result is None:
            errors.append(ValidationError(
                fpath,
                f"Entry '{instance_id}' is missing 'test_result' field"
            ))
            continue

        if not isinstance(test_result, dict):
            errors.append(ValidationError(
                fpath,
                f"Entry '{instance_id}': 'test_result' must be a JSON object"
            ))
            continue

        git_patch = test_result.get("git_patch")
        if not git_patch or not isinstance(git_patch, str) or git_patch.strip() == "":
            errors.append(ValidationError(
                fpath,
                f"Entry '{instance_id}': 'test_result.git_patch' is missing or empty"
            ))

    return errors


def validate_output_converted_jsonl(run_dir: str) -> list[ValidationError]:
    """Validate output_converted.jsonl is parseable."""
    errors = []
    fpath = os.path.join(run_dir, "output_converted.jsonl")
    if not os.path.isfile(fpath):
        return errors  # Already caught by file presence check

    _, err = load_jsonl(fpath)
    if err:
        errors.append(ValidationError(fpath, err))

    return errors


def validate_critic_jsonl(run_dir: str) -> list[ValidationError]:
    """Validate output.critic_attempt_*.jsonl files if they exist (optional)."""
    errors = []
    pattern = os.path.join(run_dir, "output.critic_attempt_*.jsonl")
    for fpath in glob.glob(pattern):
        _, err = load_jsonl(fpath)
        if err:
            errors.append(ValidationError(fpath, err))
    return errors


def validate_pass_at_8_summary(model_dir: str, model: str) -> list[ValidationError]:
    """
    Validate pass_at_8_summary.json at the model level.
    Only required when all 8 run folders exist.
    """
    errors = []

    # Count existing run folders
    existing_runs = []
    for i in range(1, MAX_RUNS + 1):
        run_path = os.path.join(model_dir, f"run_{i}")
        if os.path.isdir(run_path):
            existing_runs.append(i)

    fpath = os.path.join(model_dir, "pass_at_8_summary.json")
    all_runs_present = len(existing_runs) == MAX_RUNS

    if all_runs_present and not os.path.isfile(fpath):
        errors.append(ValidationError(
            fpath,
            f"pass_at_8_summary.json is required when all {MAX_RUNS} runs exist, but file is missing"
        ))
        return errors

    if not os.path.isfile(fpath):
        return errors  # Not all runs present yet, file is optional

    # Validate content
    data, err = load_json(fpath)
    if err:
        errors.append(ValidationError(fpath, err))
        return errors

    if not isinstance(data, dict):
        errors.append(ValidationError(fpath, "Expected a JSON object at top level"))
        return errors

    missing = check_required_keys(data, PASS_AT_8_REQUIRED_KEYS, "pass_at_8_summary.json")
    if missing:
        errors.append(ValidationError(fpath, f"Missing required keys: {missing}"))

    # Validate per_run entries have no "error" status
    per_run = data.get("per_run", [])
    if isinstance(per_run, list):
        for run_entry in per_run:
            if isinstance(run_entry, dict):
                status = run_entry.get("status", "")
                run_num = run_entry.get("run", "?")
                if status == "error":
                    errors.append(ValidationError(
                        fpath,
                        f"Run {run_num} has 'error' status in pass_at_8_summary"
                    ))

    return errors


def validate_directory_structure(run_dir: str, model: str, run_num: int) -> list[ValidationError]:
    """Validate the directory path follows the expected naming convention."""
    errors = []

    # Validate run number
    if run_num < 1 or run_num > MAX_RUNS:
        errors.append(ValidationError(
            run_dir,
            f"Run number {run_num} is out of expected range 1-{MAX_RUNS}"
        ))

    return errors


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: validate_trajectories.py <changed_files_list>")
        sys.exit(1)

    changed_files_path = sys.argv[1]

    with open(changed_files_path, "r") as f:
        changed_files = [line.strip() for line in f if line.strip()]

    if not changed_files:
        print("✅ No changed files to validate.")
        sys.exit(0)

    print(f"📋 Found {len(changed_files)} changed file(s) to validate.\n")

    parsed = extract_run_dirs_from_changed_files(changed_files)
    run_dirs = parsed["run_dirs"]
    model_dirs = parsed["model_dirs"]

    if not run_dirs and not model_dirs:
        print("✅ No SWE trajectory run directories affected by this push.")
        sys.exit(0)

    all_errors: list[ValidationError] = []
    all_warnings: list[ValidationError] = []

    # ── Validate each affected run directory ──
    for run_dir, info in sorted(run_dirs.items()):
        model = info["model"]
        run_num = info["run_num"]

        print(f"🔍 Validating: {run_dir}")

        # 1. Model name
        all_errors.extend(validate_model_name(model, run_dir))

        # 2. Directory structure
        all_errors.extend(validate_directory_structure(run_dir, model, run_num))

        # 3. File presence
        all_errors.extend(validate_run_file_presence(run_dir))

        # 4. conversations/ has .tar.gz
        all_errors.extend(validate_conversations_tar(run_dir))

        # 5. config.json schema
        all_errors.extend(validate_config_json(run_dir))

        # 6. metadata.json schema
        all_errors.extend(validate_metadata_json(run_dir))

        # 7. output.report.json schema + 0 error IDs
        all_errors.extend(validate_output_report_json(run_dir))

        # 8. output.jsonl: valid JSONL + git_patch present
        all_errors.extend(validate_output_jsonl(run_dir))

        # 9. output_converted.jsonl: valid JSONL
        all_errors.extend(validate_output_converted_jsonl(run_dir))

        # 10. critic files (optional, just validate if present)
        all_errors.extend(validate_critic_jsonl(run_dir))

    # ── Validate model-level concerns (pass_at_8_summary.json) ──
    for model_dir, model in sorted(model_dirs.items()):
        print(f"🔍 Validating model dir: {model_dir}")
        all_errors.extend(validate_pass_at_8_summary(model_dir, model))

    # ── Report ──
    errors = [e for e in all_errors if e.severity == "ERROR"]
    warnings = [e for e in all_errors if e.severity == "WARNING"]

    print("\n" + "=" * 70)

    if warnings:
        print(f"\n⚠️  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  {w}")

    if errors:
        print(f"\n❌ {len(errors)} validation error(s):\n")
        for e in errors:
            print(f"  {e}")
        print(f"\n{'=' * 70}")
        print("VALIDATION FAILED")
        sys.exit(1)
    else:
        print(f"\n✅ All validations passed!")
        if run_dirs:
            print(f"   Validated {len(run_dirs)} run folder(s) across {len(model_dirs)} model dir(s).")
        print(f"{'=' * 70}")
        sys.exit(0)


if __name__ == "__main__":
    main()