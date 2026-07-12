#!/usr/bin/env python3
"""Inventory compatibility-package exposure in wheels and source archives.

This utility reads archives only.  Baseline mode documents the current
top-level ToolBox/pySNIFS exposure without failing.  Minimized mode treats
that exposure as an error and checks for the intended private compatibility
namespace and artifact-visible licensing/provenance material.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import zipfile


LEGACY_TOP_LEVEL = ("ToolBox", "pySNIFS")


def _archive_paths(path: Path) -> list[str]:
    """Return normalized member paths from a wheel, zip, or tar archive."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return sorted(
                PurePosixPath(name).as_posix().lstrip("./")
                for name in archive.namelist()
                if name and not name.endswith("/")
            )
    if tarfile.is_tarfile(path):
        with tarfile.open(path, mode="r:*") as archive:
            return sorted(
                PurePosixPath(member.name).as_posix().lstrip("./")
                for member in archive.getmembers()
                if member.isfile()
            )
    raise ValueError(f"unsupported or invalid archive: {path}")


def _strip_sdist_root(paths: list[str]) -> list[str]:
    roots = {PurePosixPath(path).parts[0] for path in paths if "/" in path}
    for root in sorted(roots):
        prefix = root + "/"
        nested = [path[len(prefix):] for path in paths if path.startswith(prefix)]
        if "pyproject.toml" in nested or "setup.py" in nested:
            return sorted(nested)
    return paths


def inspect_archive(path: Path, mode: str) -> dict[str, object]:
    members = _strip_sdist_root(_archive_paths(path))
    legacy = {
        package: sorted(
            member for member in members
            if member == package or member.startswith(package + "/")
        )
        for package in LEGACY_TOP_LEVEL
    }
    legacy = {name: paths for name, paths in legacy.items() if paths}
    private_compat = sorted(
        member for member in members
        if member.startswith("scene_model/_compat/")
    )
    notices = sorted(
        member for member in members
        if "LICENSE" in PurePosixPath(member).name.upper()
        or "NOTICE" in PurePosixPath(member).name.upper()
        or "PROVENANCE" in PurePosixPath(member).name.upper()
    )

    errors: list[str] = []
    notes: list[str] = []
    if mode == "baseline":
        if legacy:
            notes.append(
                "expected baseline exposure found: " + ", ".join(legacy)
            )
        else:
            notes.append("baseline archive has no legacy top-level exposure")
    else:
        if legacy:
            errors.append(
                "unexpected minimized-artifact exposure: "
                + ", ".join(legacy)
            )
        if not private_compat:
            errors.append("scene_model/_compat payload is absent")
        if "scene_model/_compat/NOTICE.md" not in members:
            errors.append("scene_model/_compat/NOTICE.md is absent")
        extract_star_licenses = [
            member for member in members
            if PurePosixPath(member).name == "extract-star-MIT.txt"
        ]
        if not extract_star_licenses:
            errors.append("extract-star MIT license payload is absent")

    return {
        "artifact": str(path),
        "mode": mode,
        "member_count": len(members),
        "legacy_top_level": legacy,
        "private_compat": private_compat,
        "notice_and_license_paths": notices,
        "notes": notes,
        "errors": errors,
        "ok": not errors,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts", nargs="+", type=Path,
        help="wheel, zip, .tar.gz, or other tar archive to inspect",
    )
    parser.add_argument(
        "--mode", choices=("baseline", "minimized"), default="baseline",
        help="baseline inventories legacy exposure; minimized rejects it",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reports = []
    for artifact in args.artifacts:
        try:
            reports.append(inspect_archive(artifact, args.mode))
        except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
            reports.append({
                "artifact": str(artifact), "mode": args.mode,
                "errors": [str(exc)], "ok": False,
            })

    if args.json:
        print(json.dumps(reports, indent=2, sort_keys=True))
    else:
        for report in reports:
            status = "PASS" if report["ok"] else "FAIL"
            print(f"[{status}] {report['artifact']} ({report['mode']})")
            legacy = report.get("legacy_top_level", {})
            for package, paths in legacy.items():
                print(f"  legacy {package}: {len(paths)} file(s)")
            private = report.get("private_compat", [])
            if private:
                print(f"  private scene_model._compat: {len(private)} file(s)")
            for note in report.get("notes", []):
                print(f"  note: {note}")
            for error in report.get("errors", []):
                print(f"  error: {error}")

    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
