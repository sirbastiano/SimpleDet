import ast
import importlib
import inspect
import pathlib
import py_compile
import re
import unittest
from collections import Counter


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "simpledet" / "simpledet"
REPO_ROOT = PACKAGE_ROOT.parents[1]
SAFE_IMPORT_MODULES = [
    "simpledet",
    "simpledet._model_resolution",
    "simpledet.api",
    "simpledet.cli",
    "simpledet.__main__",
    "simpledet.bench",
    "simpledet.detectors",
    "simpledet.detectors._deps",
    "simpledet.detectors.cli",
    "simpledet.detectors.data",
    "simpledet.detectors.evaluate",
    "simpledet.detectors.infer",
    "simpledet.detectors.train",
    "simpledet.suite",
    "simpledet.suite.catalog",
    "simpledet.suite.specs",
    "simpledet.native",
    "simpledet.native.assemblers",
    "simpledet.native.backbones",
    "simpledet.native.dense_ops",
    "simpledet.native.engine",
    "simpledet.native.heads",
    "simpledet.native.modeling",
    "simpledet.native.necks",
    "simpledet.native.runtime",
]
PUBLIC_ATTRIBUTE_EXPORT_MODULES = [
    "simpledet.suite",
    "simpledet.extensions",
]
PUBLIC_NAMESPACE_EXPORT_MODULES = {
    "simpledet.detectors": ("train", "evaluate", "infer", "data", "cli"),
}
OPTIONAL_DEPS = {
    "numpy",
    "torch",
    "torchvision",
    "timm",
    "cv2",
    "PIL",
    "onnxruntime",
    "pycocotools",
}
FORBIDDEN_LEGACY_MODULES = {"mmdet", "mmcv", "mmengine"}
FORBIDDEN_LEGACY_TEXT = re.compile(r"\b(mmdet|mmdetection|mmcv|mmengine)\b", re.IGNORECASE)
EXPLICIT_LEGACY_TEXT_ALLOWLIST = {
    "_legacy.py": (
        "Legacy MMDetection .py config import/conversion is unsupported.",
    ),
    "suite/compiler.py": (
        "compile_detector_spec is a retired MMDet-era compiler.",
    ),
    "suite/native_plan.py": (
        "Native backend component plan independent of MMDet config shape.",
    ),
}


def _root_module(module_name):
    return str(module_name).split(".", 1)[0]


def _constant_string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _legacy_stack_import_references(tree):
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _root_module(alias.name) in FORBIDDEN_LEGACY_MODULES:
                    offenders.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module and _root_module(node.module) in FORBIDDEN_LEGACY_MODULES:
                offenders.append((node.lineno, node.module))
        elif isinstance(node, ast.Call):
            function_name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if function_name not in {"import_module", "__import__"} or not node.args:
                continue
            module_name = _constant_string(node.args[0])
            if module_name and _root_module(module_name) in FORBIDDEN_LEGACY_MODULES:
                offenders.append((node.lineno, module_name))
    return offenders


def _legacy_text_references(path, text):
    relative_path = path.relative_to(PACKAGE_ROOT).as_posix()
    allowed_snippets = EXPLICIT_LEGACY_TEXT_ALLOWLIST.get(relative_path, ())
    references = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not FORBIDDEN_LEGACY_TEXT.search(line):
            continue
        if any(snippet in line for snippet in allowed_snippets):
            continue
        references.append((line_no, line.strip()))
    return references


class TestRepoAudit(unittest.TestCase):
    @classmethod
    def _python_files(cls):
        return [
            path
            for path in PACKAGE_ROOT.rglob("*.py")
            if "__pycache__" not in path.parts and "configs" not in path.parts
        ]

    def test_all_python_files_compile(self):
        failures = []

        for path in sorted(self._python_files()):
            try:
                py_compile.compile(str(path), doraise=True)
            except Exception as exc:
                failures.append(f"{path.relative_to(PACKAGE_ROOT)}: {exc}")

        self.assertEqual(
            len(failures),
            0,
            "Syntax/compile errors found:\n" + "\n".join(failures[:20]),
        )

    def test_all_top_level_functions_and_classes_are_discoverable(self):
        summary = Counter()
        skipped = []

        for path in sorted(self._python_files()):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                skipped.append(f"{path.name}: non-utf8")
                continue

            tree = ast.parse(text, filename=str(path))
            nodes = [node for node in tree.body]
            function_nodes = [
                node for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            class_nodes = [node for node in nodes if isinstance(node, ast.ClassDef)]
            function_count = len(function_nodes)
            class_count = len(class_nodes)

            if "configs" in path.parts:
                continue

            summary.update({"functions": function_count, "classes": class_count})
            names = [node.name for node in function_nodes + class_nodes]
            duplicates = [name for name in set(names) if names.count(name) > 1]
            self.assertEqual(
                duplicates,
                [],
                f"{path.relative_to(PACKAGE_ROOT)} has duplicate top-level symbols: {duplicates}",
            )

        self.assertNotEqual(summary["functions"], 0)
        self.assertNotEqual(summary["classes"], 0)
        self.assertEqual(skipped, [])

    def test_public_all_exports_are_discoverable(self):
        simpledet = importlib.import_module("simpledet")
        self.assertEqual(len(simpledet.__all__), len(set(simpledet.__all__)))
        for symbol in simpledet.__all__:
            self.assertIn(symbol, dir(simpledet))

        for module_name in PUBLIC_ATTRIBUTE_EXPORT_MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                exports = tuple(getattr(module, "__all__", ()))
                self.assertNotEqual(exports, ())
                self.assertEqual(len(exports), len(set(exports)))
                missing = [symbol for symbol in exports if not hasattr(module, symbol)]
                self.assertEqual(missing, [])

        for module_name, expected_submodules in PUBLIC_NAMESPACE_EXPORT_MODULES.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                exports = tuple(getattr(module, "__all__", ()))
                self.assertEqual(exports, expected_submodules)
                for symbol in exports:
                    importlib.import_module(f"{module_name}.{symbol}")

    def test_safe_public_modules_import_and_expose_api(self):
        for module_name in SAFE_IMPORT_MODULES:
            with self.subTest(module=module_name):
                try:
                    module = importlib.import_module(module_name)
                except ModuleNotFoundError as exc:
                    missing = exc.name
                    if missing in OPTIONAL_DEPS:
                        self.skipTest(f"Optional dependency not installed: {missing}")
                    raise
                except ImportError as exc:
                    message = str(exc)
                    if any(f"'{dependency}'" in message for dependency in OPTIONAL_DEPS):
                        self.skipTest(message)
                    raise

                self.assertIsNotNone(module)

                module_file = pathlib.Path(module.__file__ or "")
                if module_file.exists():
                    ast_tree = ast.parse(module_file.read_text(encoding="utf-8"))
                    ast_defs = [
                        node.name
                        for node in ast_tree.body
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        and not node.name.startswith("_")
                    ]
                    for symbol in ast_defs:
                        attr = getattr(module, symbol, None)
                        self.assertTrue(
                            inspect.isfunction(attr) or inspect.isclass(attr),
                            f"{module_name}.{symbol} not loaded as function/class",
                        )

    def test_maintained_package_paths_do_not_reference_legacy_mmdet_stack(self):
        offenders = []
        text_offenders = []
        for path in sorted(self._python_files()):
            text = path.read_text(encoding="utf-8")
            for line_no, line in _legacy_text_references(path, text):
                text_offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{line_no}: {line}")
            tree = ast.parse(text, filename=str(path))
            references = _legacy_stack_import_references(tree)
            for line_no, module_name in references:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{line_no}: {module_name}")

        self.assertEqual(
            text_offenders,
            [],
            "Found legacy MMDet/MMCV/MMEngine references:\n" + "\n".join(text_offenders),
        )
        self.assertEqual(offenders, [], "Found legacy MMDet/MMCV/MMEngine imports:\n" + "\n".join(offenders))

    def test_archived_configs_and_explicit_migration_references_are_not_audited_as_runtime_deps(self):
        audited_paths = set(self._python_files())
        archived_config = REPO_ROOT / "MyConfigs" / "coco_pretraining.py"
        self.assertTrue(archived_config.exists())
        self.assertNotIn(archived_config, audited_paths)
        self.assertRegex(archived_config.read_text(encoding="utf-8"), FORBIDDEN_LEGACY_TEXT)

        for relative_path, allowed_snippets in EXPLICIT_LEGACY_TEXT_ALLOWLIST.items():
            with self.subTest(path=relative_path):
                path = PACKAGE_ROOT / relative_path
                text = path.read_text(encoding="utf-8")
                for snippet in allowed_snippets:
                    self.assertIn(snippet, text)
                self.assertEqual(_legacy_text_references(path, text), [])

    def test_legacy_stack_import_audit_helper_detects_static_and_dynamic_imports(self):
        tree = ast.parse(
            "\n".join(
                [
                    "import mmdet.models",
                    "from mmengine.config import Config",
                    "import_module('mmcv.ops')",
                    "__import__('mmdet.apis')",
                    "importlib.import_module('mmengine.runner')",
                    "text = 'mmdet is mentioned but not imported'",
                ]
            )
        )

        offenders = _legacy_stack_import_references(tree)

        self.assertEqual(
            offenders,
            [
                (1, "mmdet.models"),
                (2, "mmengine.config"),
                (3, "mmcv.ops"),
                (4, "mmdet.apis"),
                (5, "mmengine.runner"),
            ],
        )
