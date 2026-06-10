import configparser
from email import policy
from email.parser import Parser
import os
import subprocess
import sys
import tarfile
import unittest
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
DIST_DIR = REPO_ROOT / "dist"
EXPECTED_EXTRAS = {"cpu", "dev", "docs", "geo", "plots", "timm"}
EXPECTED_EXTRA_REQUIREMENTS = {
    "cpu": (
        "torch",
        "torchvision",
        "pytorch-lightning",
        "numpy",
        "scipy",
        "pycocotools",
        "terminaltables",
    ),
    "dev": ("build", "coverage", "tomli", "twine", "ruff"),
    "docs": (),
    "geo": ("pandas", "rasterio", "geopandas", "shapely"),
    "plots": ("SciencePlots", "matplotlib"),
    "timm": ("timm",),
}
REQUIRED_WHEEL_MODULES = (
    "simpledet/__init__.py",
    "simpledet/__main__.py",
    "simpledet/api.py",
    "simpledet/cli.py",
    "simpledet/discovery.py",
    "simpledet/detectors/cli.py",
    "simpledet/extensions/registry.py",
    "simpledet/native/cnn_blocks.py",
    "simpledet/native/modeling.py",
    "simpledet/suite/__init__.py",
)
REQUIRED_SDIST_SUFFIXES = (
    "/PKG-INFO",
    "/pyproject.toml",
    "/README.md",
    "/LICENSE",
    "/MANIFEST.in",
    "/docs/installation.html",
    "/docs/package-surface-audit.html",
    "/docs/quickstart.html",
    "/examples/quick_detector.py",
    "/examples/timm_retinanet.py",
    "/simpledet/simpledet/__init__.py",
    "/simpledet/simpledet/__main__.py",
    "/simpledet/simpledet/cli.py",
    "/simpledet/simpledet/native/cnn_blocks.py",
    "/simpledet/simpledet/native/modeling.py",
    "/simpledet/simpledet/suite/__init__.py",
    "/tests/test_packaging.py",
)


def _assert_required_exact_members(
    names: set[str], required: tuple[str, ...], artifact: str
) -> None:
    missing = [name for name in required if name not in names]
    if missing:
        raise AssertionError(
            f"{artifact} missing required members: {', '.join(missing)}"
        )


def _assert_required_suffix_members(
    names: set[str], required: tuple[str, ...], artifact: str
) -> None:
    missing = [
        suffix
        for suffix in required
        if not any(name.endswith(suffix) for name in names)
    ]
    if missing:
        raise AssertionError(
            f"{artifact} missing required members: {', '.join(missing)}"
        )


def _assert_no_generated_cache_members(names: set[str], artifact: str) -> None:
    offenders = [
        name
        for name in sorted(names)
        if "__pycache__" in name.split("/")
        or ".pytest_cache" in name.split("/")
        or ".ruff_cache" in name.split("/")
        or name.endswith((".pyc", ".pyo"))
    ]
    if offenders:
        raise AssertionError(
            f"{artifact} contains generated cache files: {', '.join(offenders)}"
        )


def _project_metadata() -> dict[str, object]:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))["project"]


def _assert_wheel_metadata_contract(metadata_text: str) -> None:
    metadata = Parser(policy=policy.default).parsestr(metadata_text)
    project = _project_metadata()

    expected_fields = {
        "Name": project["name"],
        "Version": project["version"],
        "Summary": project["description"],
        "License-Expression": project["license"],
    }
    mismatched = [
        f"{field}={metadata.get(field)!r}"
        for field, expected in expected_fields.items()
        if metadata.get(field) != expected
    ]
    if _specifier_set(metadata.get("Requires-Python")) != _specifier_set(
        str(project["requires-python"])
    ):
        mismatched.append(f"Requires-Python={metadata.get('Requires-Python')!r}")
    if mismatched:
        raise AssertionError(
            "wheel metadata fields do not match pyproject contract: "
            + ", ".join(mismatched)
        )

    provided_extras = set(metadata.get_all("Provides-Extra", ()))
    if provided_extras != EXPECTED_EXTRAS:
        raise AssertionError(
            "wheel extras metadata mismatch: "
            f"expected {sorted(EXPECTED_EXTRAS)}, got {sorted(provided_extras)}"
        )

    requirements = tuple(metadata.get_all("Requires-Dist", ()))
    for extra, package_names in EXPECTED_EXTRA_REQUIREMENTS.items():
        for package_name in package_names:
            if not _requirement_for_extra_exists(requirements, package_name, extra):
                raise AssertionError(
                    "wheel extras metadata missing requirement "
                    f"{package_name!r} for extra {extra!r}"
                )

    if any('extra == "docs"' in requirement for requirement in requirements):
        raise AssertionError("docs extra must stay dependency-free")

    legacy_requirements = [
        requirement
        for requirement in requirements
        if any(token in requirement.lower() for token in ("mmcv", "mmdet", "mmengine"))
    ]
    if legacy_requirements:
        raise AssertionError(
            "wheel metadata must not require legacy MMDetection packages: "
            + ", ".join(legacy_requirements)
        )


def _requirement_for_extra_exists(
    requirements: tuple[str, ...], package_name: str, extra: str
) -> bool:
    normalized = package_name.lower()
    return any(
        requirement.lower().startswith(normalized)
        and f'extra == "{extra}"' in requirement
        for requirement in requirements
    )


def _specifier_set(value: str | None) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def _assert_console_script_contract(entry_points_text: str) -> None:
    parser = configparser.ConfigParser()
    parser.read_string(entry_points_text)
    if not parser.has_section("console_scripts"):
        raise AssertionError("wheel missing console_scripts entry point group")
    target = parser.get("console_scripts", "simpledet", fallback=None)
    if target != "simpledet.cli:main":
        raise AssertionError(
            "wheel console script mismatch: "
            f"expected simpledet = simpledet.cli:main, got {target!r}"
        )


def _assert_wheel_distribution_contract(
    names: set[str], metadata_text: str, entry_points_text: str
) -> None:
    _assert_required_exact_members(names, REQUIRED_WHEEL_MODULES, "wheel")
    _assert_no_generated_cache_members(names, "wheel")
    _assert_wheel_metadata_contract(metadata_text)
    _assert_console_script_contract(entry_points_text)


def _assert_sdist_distribution_contract(names: set[str]) -> None:
    _assert_required_suffix_members(names, REQUIRED_SDIST_SUFFIXES, "sdist")
    _assert_no_generated_cache_members(names, "sdist")


class TestPackagingMetadata(unittest.TestCase):
    def test_project_metadata_matches_supported_publish_matrix(self):
        project = _project_metadata()
        extras = project["optional-dependencies"]

        self.assertEqual(project["requires-python"], ">=3.10,<3.13")
        self.assertEqual(project["license"], "MIT")
        self.assertEqual(project["dependencies"], ["tomli>=2.0.1; python_version < '3.11'"])
        self.assertEqual(set(extras), EXPECTED_EXTRAS)
        self.assertIn("Programming Language :: Python :: 3.12", project["classifiers"])

    def test_optional_extras_keep_runtime_groups_explicit(self):
        extras = _project_metadata()["optional-dependencies"]

        self.assertIn("torch>=2.4,<2.11", extras["cpu"])
        self.assertIn("torchvision>=0.19,<0.26", extras["cpu"])
        self.assertIn("pytorch-lightning>=2.4,<3", extras["cpu"])
        self.assertNotIn("timm>=1.0,<2", extras["cpu"])
        self.assertNotIn("matplotlib>=3.8,<4", extras["cpu"])
        self.assertNotIn("shapely>=2,<3", extras["cpu"])
        self.assertEqual(extras["docs"], [])
        self.assertEqual(extras["timm"], ["timm>=1.0,<2"])
        self.assertIn("rasterio>=1.4,<2", extras["geo"])
        self.assertIn("shapely>=2,<3", extras["geo"])
        self.assertIn("matplotlib>=3.8,<4", extras["plots"])
        self.assertFalse(
            any(
                "mmdet" in dependency.lower() or "mmcv" in dependency.lower()
                for values in extras.values()
                for dependency in values
            )
        )

    def test_public_package_imports_do_not_require_optional_runtime_extras(self):
        script = r"""
import importlib.abc
import importlib
import sys

blocked = {
    "matplotlib",
    "numpy",
    "pytorch_lightning",
    "rasterio",
    "timm",
    "torch",
    "torchvision",
}
safe_modules = (
    "simpledet",
    "simpledet.cli",
    "simpledet.discovery",
    "simpledet.extensions",
    "simpledet.suite",
)

class BlockOptionalExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            message = f"blocked optional dependency: {fullname}"
            raise ModuleNotFoundError(message, name=fullname)
        return None

sys.meta_path.insert(0, BlockOptionalExtras())
for module_name in safe_modules:
    importlib.import_module(module_name)
from simpledet import __version__, list_detectors
assert __version__
assert "retinanet" in list_detectors()
print("imports-ok")
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "simpledet")
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_license_file_exists(self):
        self.assertTrue((REPO_ROOT / "LICENSE").is_file())


class TestBuiltDistributions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wheel_path = next(DIST_DIR.glob("*.whl"), None)
        cls.sdist_path = next(DIST_DIR.glob("*.tar.gz"), None)
        if cls.wheel_path is None or cls.sdist_path is None:
            raise unittest.SkipTest(
                "Build artifacts not present in dist/. Run python3 -m build first."
            )
        source_inputs = [
            REPO_ROOT / "pyproject.toml",
            REPO_ROOT / "README.md",
            REPO_ROOT / "LICENSE",
            REPO_ROOT / "MANIFEST.in",
            REPO_ROOT / "assets" / "simpledet-logo.svg",
            REPO_ROOT / "notebooks" / "Tools" / "simpledet_showcase.ipynb",
            REPO_ROOT / "notebooks" / "Tools" / "create_train_object_detector.ipynb",
            REPO_ROOT / "examples" / "timm_retinanet.py",
            REPO_ROOT / "examples" / "custom_components_training.py",
        ]
        latest_source_mtime = max(
            path.stat().st_mtime for path in source_inputs if path.exists()
        )
        earliest_dist_mtime = min(
            cls.wheel_path.stat().st_mtime, cls.sdist_path.stat().st_mtime
        )
        if earliest_dist_mtime < latest_source_mtime:
            raise unittest.SkipTest(
                "Build artifacts are stale for the current source tree."
            )

    def test_wheel_has_package_metadata_modules_extras_and_console_script(self):
        with zipfile.ZipFile(self.wheel_path) as wheel:
            names = set(wheel.namelist())
            metadata_name = next(
                name for name in names if name.endswith(".dist-info/METADATA")
            )
            entry_points_name = next(
                name for name in names if name.endswith(".dist-info/entry_points.txt")
            )
            metadata = wheel.read(metadata_name).decode("utf-8")
            entry_points = wheel.read(entry_points_name).decode("utf-8")

        _assert_wheel_distribution_contract(names, metadata, entry_points)

        self.assertIn("Requires-Python: <3.13,>=3.10", metadata)
        self.assertIn("Provides-Extra: cpu", metadata)
        self.assertIn("Provides-Extra: docs", metadata)
        self.assertIn("Provides-Extra: timm", metadata)
        requirements = [
            line for line in metadata.splitlines() if line.startswith("Requires-Dist: ")
        ]
        self.assertTrue(
            any(
                line.startswith("Requires-Dist: timm") and 'extra == "timm"' in line
                for line in requirements
            )
        )
        self.assertFalse(
            any(
                line.startswith("Requires-Dist: timm") and 'extra == "cpu"' in line
                for line in requirements
            )
        )
        self.assertFalse(
            any(
                line.startswith("Requires-Dist: matplotlib") and 'extra == "cpu"' in line
                for line in requirements
            )
        )
        self.assertNotIn("Provides-Extra: openmmlab", metadata)
        self.assertNotIn("mmcv-lite", metadata)

    def test_wheel_excludes_repo_only_pyscript_helpers(self):
        with zipfile.ZipFile(self.wheel_path) as wheel:
            names = set(wheel.namelist())

        self.assertFalse(any(name.startswith("simpledet/src/") for name in names))
        self.assertNotIn("simpledet/pyscripts/Basetrainer.py", names)

    def test_sdist_has_project_metadata_docs_modules_and_no_cache_files(self):
        with tarfile.open(self.sdist_path, "r:gz") as archive:
            names = set(archive.getnames())

        _assert_sdist_distribution_contract(names)
        self.assertTrue(any(name.endswith("/LICENSE") for name in names))
        self.assertTrue(any(name.endswith("/assets/simpledet-logo.svg") for name in names))
        self.assertTrue(
            any(name.endswith("/notebooks/Tools/simpledet_showcase.ipynb") for name in names)
        )
        self.assertTrue(
            any(name.endswith("/notebooks/Tools/create_train_object_detector.ipynb") for name in names)
        )

    def test_sdist_includes_example_gallery(self):
        with tarfile.open(self.sdist_path, "r:gz") as archive:
            names = archive.getnames()

        self.assertTrue(any(name.endswith("/examples/timm_retinanet.py") for name in names))
        self.assertTrue(any(name.endswith("/examples/quick_detector.py") for name in names))
        self.assertTrue(any(name.endswith("/examples/custom_components.py") for name in names))
        self.assertTrue(any(name.endswith("/examples/custom_components_training.py") for name in names))


class TestDistributionVerifierNegativeCases(unittest.TestCase):
    def test_sdist_verifier_fails_when_docs_are_missing(self):
        names = _minimal_valid_sdist_names()
        names.remove("simpledet-0.1.0/docs/quickstart.html")

        with self.assertRaisesRegex(AssertionError, "docs/quickstart.html"):
            _assert_sdist_distribution_contract(names)

    def test_sdist_verifier_fails_when_package_modules_are_missing(self):
        names = _minimal_valid_sdist_names()
        names.remove("simpledet-0.1.0/simpledet/simpledet/__init__.py")

        with self.assertRaisesRegex(AssertionError, "simpledet/simpledet/__init__.py"):
            _assert_sdist_distribution_contract(names)

    def test_wheel_verifier_fails_when_package_modules_are_missing(self):
        names = _minimal_valid_wheel_names()
        names.remove("simpledet/cli.py")

        with self.assertRaisesRegex(AssertionError, "simpledet/cli.py"):
            _assert_wheel_distribution_contract(
                names,
                _minimal_valid_wheel_metadata(),
                _minimal_valid_entry_points(),
            )

    def test_sdist_verifier_fails_when_pyproject_metadata_is_missing(self):
        names = _minimal_valid_sdist_names()
        names.remove("simpledet-0.1.0/pyproject.toml")

        with self.assertRaisesRegex(AssertionError, "pyproject.toml"):
            _assert_sdist_distribution_contract(names)

    def test_distribution_verifiers_fail_on_generated_cache_files(self):
        sdist_names = _minimal_valid_sdist_names()
        sdist_names.add("simpledet-0.1.0/simpledet/simpledet/__pycache__/cli.cpython-312.pyc")
        wheel_names = _minimal_valid_wheel_names()
        wheel_names.add("simpledet/__pycache__/cli.cpython-312.pyc")

        with self.assertRaisesRegex(AssertionError, "generated cache"):
            _assert_sdist_distribution_contract(sdist_names)
        with self.assertRaisesRegex(AssertionError, "generated cache"):
            _assert_wheel_distribution_contract(
                wheel_names,
                _minimal_valid_wheel_metadata(),
                _minimal_valid_entry_points(),
            )


def _minimal_valid_sdist_names() -> set[str]:
    return {f"simpledet-0.1.0{suffix}" for suffix in REQUIRED_SDIST_SUFFIXES}


def _minimal_valid_wheel_names() -> set[str]:
    return {
        *REQUIRED_WHEEL_MODULES,
        "simpledet-0.1.0.dist-info/METADATA",
        "simpledet-0.1.0.dist-info/entry_points.txt",
    }


def _minimal_valid_wheel_metadata() -> str:
    lines = [
        "Metadata-Version: 2.4",
        "Name: simpledet",
        "Version: 0.1.0",
        "Summary: Native PyTorch Lightning toolkit for object detection workflows",
        "License-Expression: MIT",
        "Requires-Python: <3.13,>=3.10",
    ]
    for extra, package_names in EXPECTED_EXTRA_REQUIREMENTS.items():
        lines.append(f"Provides-Extra: {extra}")
        for package_name in package_names:
            lines.append(f'Requires-Dist: {package_name}; extra == "{extra}"')
    return "\n".join(lines)


def _minimal_valid_entry_points() -> str:
    return "[console_scripts]\nsimpledet = simpledet.cli:main\n"
