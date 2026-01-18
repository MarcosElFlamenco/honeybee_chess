#!/usr/bin/env python3
"""Generate a SLURM script from a training config and template.

Usage: python generate_script.py <config_name> [--arg1 val --arg2 val ...]

The script looks for `training_configs/<config_name>.py`, loads variables
from it (if present), merges them with defaults extracted from
`src/train.py`, then writes `<config_name>.sh` based on
`scripts/main_train.sh` with the training args embedded.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import re
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN_PY = ROOT / "src" / "train.py"
TEMPLATE = ROOT / "scripts" / "template.sh"
CONFIG_DIR = ROOT / "training_configs"


def extract_train_defaults(train_path: Path):
    src = train_path.read_text()
    pattern = r"parser\.add_argument\(([^\)]*)\)"
    defaults = {}
    types = {}
    actions = {}
    for m in re.finditer(pattern, src, re.DOTALL):
        call = m.group(1)
        name_m = re.search(r"[\"']--([A-Za-z0-9_\-]+)[\"']", call)
        if not name_m:
            continue
        name = name_m.group(1)

        # default
        def_m = re.search(r"default\s*=\s*([^,\n]+)", call)
        if def_m:
            raw = def_m.group(1).strip()
            try:
                val = ast.literal_eval(raw)
            except Exception:
                if raw in ("None", "True", "False"):
                    val = eval(raw)
                else:
                    # fallback to string without quotes
                    val = raw
            defaults[name] = val
        # type
        type_m = re.search(r"type\s*=\s*([^,\n]+)", call)
        if type_m:
            t_raw = type_m.group(1).strip()
            if t_raw in ("int", "float", "str", "bool"):
                types[name] = eval(t_raw)
        # action
        action_m = re.search(r"action\s*=\s*[\"']([^\"']+)[\"']", call)
        if action_m:
            actions[name] = action_m.group(1)

        # if no default but action=store_true, default is False
        if name not in defaults and actions.get(name) == "store_true":
            defaults[name] = False
    return defaults, types, actions


def load_config_module(name: str, config_dir: Path):
    path = config_dir / f"{name}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"training_configs.{name}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_config_file(name: str, defaults: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.py"
    lines = ["# Auto-generated config matching src/train.py defaults\n"]
    for k in sorted(defaults.keys()):
        v = defaults[k]
        lines.append(f"{k} = {repr(v)}\n")
    path.write_text("".join(lines))
    return path


def generate_slurm(template_text: str, config_name: str, final_args: dict):
    # Adjust SBATCH metadata
    template_text = re.sub(r"^#SBATCH\s+--job-name=.*$", f"#SBATCH --job-name=train_{config_name}", template_text, flags=re.M)
    template_text = re.sub(r"^#SBATCH\s+--output=.*$", f"#SBATCH --output=logs/{config_name}.out", template_text, flags=re.M)
    template_text = re.sub(r"^#SBATCH\s+--error=.*$", f"#SBATCH --error=logs/{config_name}.out", template_text, flags=re.M)
    template_text = re.sub(r"--output_dir * \\$", f"--output_dir ./outputs/{config_name}", template_text, flags=re.M)
    
    # Build python arg block
    parts = ["python -m src.train \\"]
    for k, v in final_args.items():
        if k == "output_dir":
            parts.append(f"    --{k} ./outputs/{config_name} \\")
            continue
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                parts.append(f"    --{k} \\")
            continue
        # quote strings
        if isinstance(v, str):
            val = shlex.quote(v)
        else:
            val = str(v)
        parts.append(f"    --{k} {val} \\")


    # remove trailing backslash on last
    if parts:
        parts[-1] = parts[-1].rstrip(" \\") + "\n"
    cmd_block = "\n".join(parts)

    # Replace existing python -m block
    new_text = re.sub(r"python -m src\.train[\s\S]*$", cmd_block, template_text, flags=re.M)
    return new_text


def main():
    # First-stage parse: get config name and leftover args
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("config_name", help="Name of training config (file in training_configs without .py)")
    known, remaining = first.parse_known_args()
    config_name = known.config_name

    defaults, types, actions = extract_train_defaults(TRAIN_PY)

    # load config module if present and merge
    mod = load_config_module(config_name, CONFIG_DIR)
    if mod:
        for k in list(defaults.keys()):
            if hasattr(mod, k):
                defaults[k] = getattr(mod, k)

    # Build final parser that accepts overrides
    parser = argparse.ArgumentParser(description="Generate SLURM script for training")
    parser.add_argument("config_name")
    for k, dv in defaults.items():
        argname = f"--{k}"
        if actions.get(k) == "store_true":
            parser.add_argument(argname, action="store_true", default=dv)
        else:
            typ = types.get(k, type(dv) if dv is not None else str)
            parser.add_argument(argname, type=typ, default=dv)

    parsed = parser.parse_args([config_name] + remaining)

    # gather final args for trainer invocation
    final_args = {}
    for k in defaults.keys():
        attr = k.replace("-", "_")
        val = getattr(parsed, k.replace("-", "_"))
        final_args[k] = val

    # read template
    template_text = TEMPLATE.read_text()

    slurm_text = generate_slurm(template_text, config_name, final_args)
    out_path = ROOT / f"scripts/{config_name}.sh"
    out_path.write_text(slurm_text)
    print(f"Wrote SLURM script: {out_path}")

    # ensure a config file exists for the user to edit
    cfg_path = CONFIG_DIR / f"{config_name}.py"
    if not cfg_path.exists():
        write_config_file(config_name, defaults, CONFIG_DIR)
        print(f"Wrote example config: {cfg_path}")


if __name__ == "__main__":
    main()
