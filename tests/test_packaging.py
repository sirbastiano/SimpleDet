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
        self.assertIn("cpu", extras)
        self.assertIn("openmmlab", extras)
        self.assertIn("geo", extras)
        self.assertIn("plots", extras)
        self.assertIn("Programming Language :: Python :: 3.12", project["classifiers"])

    def test_license_file_exists(self):
        self.assertTrue((REPO_ROOT / "LICENSE").is_file())


class TestBuiltDistributions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wheel_path = next(DIST_DIR.glob("*.whl"), None)
        cls.sdist_path = next(DIST_DIR.glob("*.tar.gz"), None)
        if cls.wheel_path is None or cls.sdist_path is None:
            raise unittest.SkipTest("Build artifacts not present in dist/. Run python3 -m build first.")
        source_inputs = [
            REPO_ROOT / "pyproject.toml",
            REPO_ROOT / "README.md",
            REPO_ROOT / "LICENSE",
        ]
        latest_source_mtime = max(path.stat().st_mtime for path in source_inputs if path.exists())
        earliest_dist_mtime = min(cls.wheel_path.stat().st_mtime, cls.sdist_path.stat().st_mtime)
        if earliest_dist_mtime < latest_source_mtime:
            raise unittest.SkipTest("Build artifacts are stale for the current source tree.")

    def test_wheel_metadata_advertises_cpu_runtime_extra(self):
        with zipfile.ZipFile(self.wheel_path) as wheel:
            metadata_name = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
            metadata = wheel.read(metadata_name).decode("utf-8")

        self.assertIn("Requires-Python: <3.13,>=3.10", metadata)
        self.assertIn("Provides-Extra: cpu", metadata)
        self.assertIn("Provides-Extra: openmmlab", metadata)
        self.assertIn("Requires-Dist: mmcv-lite<2.2,>=2.1.0; extra == \"cpu\"", metadata)

    def test_wheel_excludes_repo_only_pyscript_helpers(self):
        with zipfile.ZipFile(self.wheel_path) as wheel:
            names = set(wheel.namelist())

        self.assertIn("simpledet/src/base_config.py", names)
        self.assertNotIn("simpledet/pyscripts/Basetrainer.py", names)

    def test_sdist_includes_license(self):
        with tarfile.open(self.sdist_path, "r:gz") as archive:
            names = archive.getnames()

        self.assertTrue(any(name.endswith("/LICENSE") for name in names))
