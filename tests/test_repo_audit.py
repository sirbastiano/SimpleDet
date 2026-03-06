import ast
import pathlib
import py_compile
import importlib
import inspect
import unittest
from collections import Counter


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "simpledet" / "simpledet"
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
    "simpledet.suite.compiler",
    "simpledet.suite.specs",
]
OPTIONAL_DEPS = {
    "numpy",
    "torch",
    "torchvision",
    "mmcv",
    "mmengine",
    "mmdet",
    "timm",
    "cv2",
    "PIL",
    "onnxruntime",
    "pycocotools",
}


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
