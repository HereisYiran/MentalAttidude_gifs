from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gridworld.helper.render import render_scenario
else:
    from .helper.render import render_scenario


def find_scenario_files(scenarios_root: Path, category: str | None, scenario: str | None) -> list[Path]:
    if category and scenario:
        return [scenarios_root / category / f"{scenario}.json"]

    if scenario and not category:
        return sorted(scenarios_root.glob(f"**/{scenario}.json"))

    if category and not scenario:
        return sorted((scenarios_root / category).glob("*.json"))

    return sorted(scenarios_root.glob("**/*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JSON-defined Gridworld scenarios and export GIFs.")
    parser.add_argument("--category", help="belief|hope|fear|desire")
    parser.add_argument("--scenario", help="Scenario file stem, e.g. belief_scenario1")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Path to the gridworld folder",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    scenarios_root = root / "scenarios"
    output_root = root / "output"

    files = find_scenario_files(scenarios_root, args.category, args.scenario)
    files = [p for p in files if p.exists()]

    if not files:
        raise SystemExit("No matching scenario JSON files found.")

    for path in files:
        gif_path = render_scenario(path, output_root)
        print(f"Rendered: {gif_path}")


if __name__ == "__main__":
    main()
