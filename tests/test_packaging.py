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


class TestPackagingMetadata(unittest.TestCase):
    def test_project_metadata_matches_supported_publish_matrix(self):
        payload = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
        project = payload["project"]
        extras = project["optional-dependencies"]

        self.assertEqual(project["requires-python"], ">=3.10,<3.13")
        self.assertEqual(project["license"], "MIT")
        self.assertEqual(project["dependencies"], [])
        self.assertEqual(set(extras), {"cpu", "dev", "docs", "geo", "plots", "timm"})
        self.assertIn("Programming Language :: Python :: 3.12", project["classifiers"])

    def test_optional_extras_keep_runtime_groups_explicit(self):
        payload = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
        extras = payload["project"]["optional-dependencies"]

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

    def test_base_package_import_does_not_require_optional_runtime_extras(self):
        script = r"""
import importlib.abc
import sys

blocked = {"matplotlib", "rasterio", "timm", "torch"}

class BlockOptionalExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            message = f"blocked optional dependency: {fullname}"
            raise ModuleNotFoundError(message, name=fullname)
        return None

sys.meta_path.insert(0, BlockOptionalExtras())
import simpledet
print(simpledet.__version__)
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
            REPO_ROOT / "examples" / "timm_retinanet.py",
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

    def test_wheel_metadata_advertises_cpu_runtime_extra(self):
        with zipfile.ZipFile(self.wheel_path) as wheel:
            metadata_name = next(
                name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
            )
            metadata = wheel.read(metadata_name).decode("utf-8")

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

    def test_sdist_includes_license(self):
        with tarfile.open(self.sdist_path, "r:gz") as archive:
            names = archive.getnames()

        self.assertTrue(any(name.endswith("/LICENSE") for name in names))

    def test_sdist_includes_showcase_assets(self):
        with tarfile.open(self.sdist_path, "r:gz") as archive:
            names = archive.getnames()

        self.assertTrue(any(name.endswith("/assets/simpledet-logo.svg") for name in names))
        self.assertTrue(
            any(name.endswith("/notebooks/Tools/simpledet_showcase.ipynb") for name in names)
        )

    def test_sdist_includes_example_gallery(self):
        with tarfile.open(self.sdist_path, "r:gz") as archive:
            names = archive.getnames()

        self.assertTrue(any(name.endswith("/examples/timm_retinanet.py") for name in names))
        self.assertTrue(any(name.endswith("/examples/quick_detector.py") for name in names))
