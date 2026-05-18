"""Native SimpleDet component registry primitives."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass, field
from difflib import get_close_matches
from importlib import import_module
from typing import Any, Callable


@dataclass(slots=True, frozen=True)
class DependencyRequirement:
    """Optional runtime dependency required by a registered component."""

    module: str
    extra: str | None = None
    package: str | None = None

    def __post_init__(self) -> None:
        if not str(self.module).strip():
            raise ValueError("Dependency requirements need a non-empty module name.")

    @classmethod
    def coerce(cls, value: Any) -> "DependencyRequirement":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, dict):
            return cls(
                str(value["module"]),
                extra=value.get("extra"),
                package=value.get("package"),
            )
        if isinstance(value, (tuple, list)) and value:
            module = str(value[0])
            extra = None if len(value) < 2 else value[1]
            package = None if len(value) < 3 else value[2]
            return cls(module, extra=extra, package=package)
        raise TypeError("Dependency requirements must be strings, mappings, or tuples.")


@dataclass(slots=True, frozen=True)
class ComponentMetadata:
    """Stable registry metadata for a native SimpleDet component."""

    name: str
    aliases: tuple[str, ...]
    kind: str
    factory: Any
    required_dependencies: tuple[DependencyRequirement, ...] = ()
    tensor_contracts: tuple[str, ...] = ()
    validation_status: str = "unvalidated"
    family: str | None = None
    summary: str | None = None

    @property
    def module(self) -> str:
        return str(getattr(self.factory, "__module__", ""))

    @property
    def callable(self) -> str:
        return str(
            getattr(self.factory, "__qualname__", None)
            or getattr(self.factory, "__name__", None)
            or self.factory.__class__.__name__
        )

    @property
    def factory_path(self) -> str:
        if self.module:
            return f"{self.module}.{self.callable}"
        return self.callable

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "aliases": list(self.aliases),
            "factory": self.factory_path,
            "module": self.module,
            "callable": self.callable,
            "required_dependencies": [
                asdict(dependency) for dependency in self.required_dependencies
            ],
            "tensor_contracts": list(self.tensor_contracts),
            "validation_status": self.validation_status,
            "family": self.family,
            "summary": self.summary,
        }


class RegistryLookupError(KeyError):
    """Raised when a registry lookup cannot resolve a component name or alias."""


class MissingComponentDependencyError(ImportError):
    """Raised when a registered component is missing optional dependencies."""


@dataclass(slots=True)
class ExtensionRegistry:
    """Small explicit registry for native SimpleDet component factories."""

    kind: str
    _items: dict[str, Any] = field(default_factory=dict)
    _metadata: dict[str, ComponentMetadata] = field(default_factory=dict)

    def register(
        self,
        name: str | None = None,
        *,
        aliases: tuple[str, ...] | list[str] = (),
        required_dependencies: tuple[Any, ...] | list[Any] = (),
        tensor_contracts: tuple[str, ...] | list[str] = (),
        validation_status: str = "unvalidated",
        family: str | None = None,
        summary: str | None = None,
    ) -> Callable[[Any], Any]:
        def decorator(component: Any) -> Any:
            resolved = str(name or getattr(component, "__name__", "")).strip()
            if not resolved:
                raise ValueError(f"{self.kind} registrations require a non-empty name.")
            aliases_tuple = _normalize_text_tuple(aliases)
            self._validate_aliases(resolved, aliases_tuple, component)
            existing = self._items.get(resolved)
            if existing is not None and existing is not component:
                if _same_component(existing, component):
                    self._metadata.setdefault(
                        resolved,
                        self._build_metadata(
                            resolved,
                            existing,
                            aliases_tuple,
                            required_dependencies,
                            tensor_contracts,
                            validation_status,
                            family,
                            summary,
                        ),
                    )
                    return existing
                raise ValueError(
                    f"{self.kind} component '{resolved}' is already registered."
                )
            self._items[resolved] = component
            self._metadata[resolved] = self._build_metadata(
                resolved,
                component,
                aliases_tuple,
                required_dependencies,
                tensor_contracts,
                validation_status,
                family,
                summary,
            )
            self._propagate_component_contract(resolved)
            return component

        return decorator

    def get(self, name: str) -> Any:
        return self.lookup(name).factory

    def lookup(self, name: str) -> ComponentMetadata:
        resolved = self.resolve_name(name)
        return self._metadata_for(resolved)

    def resolve_name(self, name: str) -> str:
        requested = str(name).strip()
        if not requested:
            raise ValueError(f"{self.kind} component name must be a non-empty string.")
        if requested in self._items:
            return requested

        requested_lower = requested.lower()
        for item in self.names():
            metadata = self._metadata_for(item)
            for alias in metadata.aliases:
                if alias.lower() == requested_lower:
                    return item

        for item in self.names():
            if item.lower() == requested_lower:
                return item

        requested_key = _registry_key(requested)
        for item in self.names():
            metadata = self._metadata_for(item)
            for alias in metadata.aliases:
                if _registry_key(alias) == requested_key:
                    return item

        for item in self.names():
            if _registry_key(item) == requested_key:
                return item

        raise RegistryLookupError(self._unknown_message(requested))

    def metadata(self, name: str) -> ComponentMetadata:
        return self.lookup(name)

    def aliases(self) -> dict[str, str]:
        pairs: dict[str, str] = {}
        for name in self.names():
            metadata = self._metadata_for(name)
            for alias in metadata.aliases:
                pairs[alias] = name
        return dict(sorted(pairs.items()))

    def entries(self) -> list[ComponentMetadata]:
        return [self._metadata_for(name) for name in self.names()]

    def require_dependencies(self, name: str) -> None:
        metadata = self.lookup(name)
        missing: list[DependencyRequirement] = []
        for dependency in metadata.required_dependencies:
            try:
                import_module(dependency.module)
            except ModuleNotFoundError as exc:
                missing_name = exc.name or dependency.module
                if missing_name == dependency.module or dependency.module.startswith(
                    f"{missing_name}."
                ):
                    missing.append(dependency)
                else:
                    raise
        if missing:
            details = "; ".join(_dependency_message(dependency) for dependency in missing)
            raise MissingComponentDependencyError(
                f"{self.kind} component '{metadata.name}' requires optional "
                f"dependencies: {details}."
            )

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

    def _build_metadata(
        self,
        name: str,
        component: Any,
        aliases: tuple[str, ...],
        required_dependencies: tuple[Any, ...] | list[Any],
        tensor_contracts: tuple[str, ...] | list[str],
        validation_status: str,
        family: str | None,
        summary: str | None,
    ) -> ComponentMetadata:
        dependencies = tuple(
            DependencyRequirement.coerce(dependency)
            for dependency in required_dependencies
        )
        return ComponentMetadata(
            name=name,
            aliases=aliases,
            kind=self.kind,
            factory=component,
            required_dependencies=dependencies,
            tensor_contracts=_normalize_text_tuple(tensor_contracts),
            validation_status=str(validation_status).strip() or "unvalidated",
            family=None if family is None else str(family).strip() or None,
            summary=None if summary is None else str(summary).strip() or None,
        )

    def _metadata_for(self, name: str) -> ComponentMetadata:
        metadata = self._metadata.get(name)
        component = self._items[name]
        if metadata is not None and metadata.factory is component:
            return metadata
        return ComponentMetadata(
            name=name,
            aliases=(),
            kind=self.kind,
            factory=component,
        )

    def _validate_aliases(
        self,
        name: str,
        aliases: tuple[str, ...],
        component: Any,
    ) -> None:
        name_key = _registry_key(name)
        alias_keys: set[str] = set()
        for alias in aliases:
            alias_key = _registry_key(alias)
            if alias_key in alias_keys:
                raise ValueError(
                    f"{self.kind} component '{name}' repeats alias '{alias}'."
                )
            alias_keys.add(alias_key)

        for existing_name in self.names():
            existing_component = self._items[existing_name]
            existing_name_key = _registry_key(existing_name)
            existing_alias_keys = {
                _registry_key(alias)
                for alias in self._metadata_for(existing_name).aliases
            }
            collision = (alias_keys & ({existing_name_key} | existing_alias_keys)) | (
                {name_key} & existing_alias_keys
            ) | (
                {name_key} & {existing_name_key}
            )
            if collision and not _same_component(existing_component, component):
                token = sorted(collision)[0]
                raise ValueError(
                    f"{self.kind} component '{name}' conflicts with existing "
                    f"component '{existing_name}' for alias '{token}'."
                )

    def _propagate_component_contract(self, source_name: str) -> None:
        source = self._metadata[source_name]
        if _metadata_contract_is_empty(source):
            return
        for name, component in list(self._items.items()):
            if name == source_name or not _same_component(component, source.factory):
                continue
            current = self._metadata_for(name)
            self._metadata[name] = ComponentMetadata(
                name=name,
                aliases=current.aliases,
                kind=current.kind,
                factory=current.factory,
                required_dependencies=current.required_dependencies
                or source.required_dependencies,
                tensor_contracts=current.tensor_contracts or source.tensor_contracts,
                validation_status=current.validation_status
                if current.validation_status != "unvalidated"
                else source.validation_status,
                family=current.family or source.family,
                summary=current.summary or source.summary,
            )

    def _unknown_message(self, requested: str) -> str:
        names = self.names()
        alias_pairs = [
            f"{alias} -> {name}"
            for name in names
            for alias in self._metadata_for(name).aliases
        ]
        candidates = names + [pair.split(" -> ", 1)[0] for pair in alias_pairs]
        nearby = get_close_matches(requested, candidates, n=5, cutoff=0.25)
        if not nearby and alias_pairs:
            nearby = [pair.split(" -> ", 1)[0] for pair in alias_pairs[:5]]
        available = ", ".join(names) or "<none>"
        nearby_text = ", ".join(nearby) if nearby else "<none>"
        alias_text = ", ".join(alias_pairs[:10]) if alias_pairs else "<none>"
        return (
            f"Unknown {self.kind} component '{requested}'. Registered {self.kind} "
            f"names: {available}. Nearby {self.kind} aliases/names: {nearby_text}. "
            f"Aliases: {alias_text}."
        )


def _normalize_text_tuple(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _registry_key(value: str) -> str:
    return "".join(character for character in str(value).strip().lower() if character.isalnum())


def _same_component(left: Any, right: Any) -> bool:
    return left is right or (
        getattr(left, "__module__", None) == getattr(right, "__module__", None)
        and getattr(left, "__qualname__", None) == getattr(right, "__qualname__", None)
    )


def _metadata_contract_is_empty(metadata: ComponentMetadata) -> bool:
    return (
        not metadata.required_dependencies
        and not metadata.tensor_contracts
        and metadata.validation_status == "unvalidated"
        and metadata.family is None
        and metadata.summary is None
    )


def _dependency_message(dependency: DependencyRequirement) -> str:
    package = dependency.package or dependency.module
    if dependency.extra:
        return (
            f"'{dependency.module}' (install '{package}' via "
            f"`python -m pip install 'simpledet[{dependency.extra}]'`)"
        )
    return f"'{dependency.module}' (install '{package}')"


ENCODERS = ExtensionRegistry("encoder")
NECKS = ExtensionRegistry("neck")
HEADS = ExtensionRegistry("head")
DECODERS = ExtensionRegistry("decoder")
DETECTORS = ExtensionRegistry("detector")
LOSSES = ExtensionRegistry("loss")
ASSIGNERS = ExtensionRegistry("assigner")
POSTPROCESSORS = ExtensionRegistry("postprocessor")
