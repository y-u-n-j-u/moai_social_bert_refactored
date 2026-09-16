#!/usr/bin/env python3
"""Hash a deployment snapshot and verify the files actually imported by Python.

This tool never constructs ROS nodes or starts the model. Only the package,
launch/config/RViz files and packaging metadata are included, not caches/tests.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


PACKAGE = "moai_jackal_spubert"
DEFAULT_SOURCE = Path("/root/moai_stability_ws/src") / PACKAGE
DEFAULT_MANIFEST = Path("/root/jackal_deployment/source_manifest.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files(package):
    paths = []
    for name in ("package.xml", "setup.py", "setup.cfg"):
        paths.append(package / name)
    paths.extend((package / "resource").glob("*"))
    for folder, pattern in ((PACKAGE, "*.py"), ("launch", "*.py"),
                            ("config", "*.yaml"), ("rviz", "*.rviz")):
        paths.extend((package / folder).rglob(pattern))
    return sorted(p for p in paths if p.is_file() and "__pycache__" not in p.parts)


def create(package, revision):
    files = {str(p.relative_to(package).as_posix()): digest(p)
             for p in source_files(package)}
    for required in ("package.xml", "setup.py", "setup.cfg", f"{PACKAGE}/__init__.py",
                     "config/real_jackal_social005.yaml", "config/nav2_route_planner.yaml",
                     "launch/real_jackal_spubert.launch.py", "launch/nav2_route_planner.launch.py"):
        if required not in files:
            raise RuntimeError(f"Required package input missing: {required}")
    content_hash = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"format": 1, "package": PACKAGE, "git_revision": revision,
            "source_sha256": content_hash, "files": files}


def verify(manifest, package, source_only=False):
    expected = manifest["files"]
    actual = create(package, manifest.get("git_revision", "unknown"))
    if actual["files"] != expected:
        differences = sorted(k for k in set(expected) | set(actual["files"])
                             if expected.get(k) != actual["files"].get(k))
        raise RuntimeError("Source snapshot differs: " + ", ".join(differences))
    if actual["source_sha256"] != manifest["source_sha256"]:
        raise RuntimeError("Manifest aggregate checksum does not match its files")
    checks = {"source": len(expected), "imports": 0, "installed_share": 0}
    if not source_only:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory(PACKAGE))
        for relative, checksum in expected.items():
            parts = Path(relative).parts
            if parts[0] == PACKAGE and relative.endswith(".py"):
                module_parts = list(Path(relative).with_suffix("").parts)
                if module_parts[-1] == "__init__":
                    module_parts.pop()
                module = importlib.import_module(".".join(module_parts))
                imported = Path(module.__file__).resolve()
                wanted = (package / relative).resolve()
                if imported != wanted or digest(imported) != checksum:
                    raise RuntimeError(f"Wrong Python import for {relative}: {imported}")
                checks["imports"] += 1
            elif parts[0] in ("launch", "config", "rviz") or relative == "package.xml":
                installed = share / relative
                if not installed.is_file() or digest(installed) != checksum:
                    raise RuntimeError(f"Installed share differs: {installed}")
                checks["installed_share"] += 1
    return {"result": "PASS", "source_sha256": manifest["source_sha256"],
            "git_revision": manifest.get("git_revision"), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("create")
    make.add_argument("--package", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--revision", default="unrecorded")
    check = sub.add_parser("verify")
    check.add_argument("--package", type=Path, default=DEFAULT_SOURCE)
    check.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    check.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    if args.command == "create":
        manifest = create(args.package.resolve(), args.revision)
        args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(manifest["source_sha256"])
    else:
        manifest = json.loads(args.manifest.read_text())
        print(json.dumps(verify(manifest, args.package.resolve(), args.source_only), indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"DEPLOYMENT VERIFY FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
