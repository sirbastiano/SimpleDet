"""Native SimpleDet component registry primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable


@dataclass(slots=True)
class ExtensionRegistry:
    """Small explicit registry for native SimpleDet component factories."""

    kind: str
    _items: dict[str, Any] = field(default_factory=dict)

    def register(self, name: str | None = None) -> Callable[[Any], Any]:
        def decorator(component: Any) -> Any:
            resolved = str(name or getattr(component, "__name__", "")).strip()
            if not resolved:
                raise ValueError(f"{self.kind} registrations require a non-empty name.")
            existing = self._items.get(resolved)
            if existing is not None and existing is not component:
                if (
                    getattr(existing, "__module__", None) == getattr(component, "__module__", None)
                    and getattr(existing, "__name__", None) == getattr(component, "__name__", None)
                ):
                    return existing
                raise ValueError(
                    f"{self.kind} component '{resolved}' is already registered."
                )
            self._items[resolved] = component
            return component

        return decorator

    def get(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._items))
            raise KeyError(
                f"Unknown {self.kind} component '{name}'. Registered: {known}."
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._items)

    def import_modules(self, *module_names: str) -> list[str]:
        imported: list[str] = []
        for module_name in module_names:
            text = str(module_name).strip()
            if not text:
                continue
            import_module(text)
            imported.append(text)
        return imported


ENCODERS = ExtensionRegistry("encoder")
NECKS = ExtensionRegistry("neck")
HEADS = ExtensionRegistry("head")
DECODERS = ExtensionRegistry("decoder")
DETECTORS = ExtensionRegistry("detector")
