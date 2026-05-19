"""Optional-dependency-safe discovery metadata for the package CLI."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import util as importlib_util
from typing import Iterable

from .detectors.data import list_formats
from .suite.backbone_aliases import BACKBONE_ALIASES
from .suite.catalog import ARCHITECTURE_FAMILIES


@dataclass(frozen=True, slots=True)
class Requirement:
    module: str
    extra: str | None = None


@dataclass(frozen=True, slots=True)
class DiscoveryComponent:
    name: str
    kind: str
    validation_status: str
    aliases: tuple[str, ...] = ()
    required_dependencies: tuple[Requirement, ...] = ()

    @property
    def required_extras(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    str(requirement.extra)
                    for requirement in self.required_dependencies
                    if requirement.extra
                }
            )
        )


@dataclass(frozen=True, slots=True)
class DiscoveryRow:
    name: str
    kind: str
    validation_status: str
    required_extras: tuple[str, ...] = ()
    required_dependencies: tuple[str, ...] = ()

    @property
    def required_extra_text(self) -> str:
        return ",".join(self.required_extras) if self.required_extras else "-"

    @property
    def required_dependency_text(self) -> str:
        return ",".join(self.required_dependencies) if self.required_dependencies else "-"


_TORCH_CPU = (Requirement("torch", "cpu"),)
_TORCHVISION_CPU = (Requirement("torch", "cpu"), Requirement("torchvision", "cpu"))
_TIMM_BACKBONE = (Requirement("torch", "cpu"), Requirement("timm", "timm"))


_STATIC_HEADS: tuple[DiscoveryComponent, ...] = (
    DiscoveryComponent("RetinaHead", "dense", "runtime_validated", ("retina", "retina_head"), _TORCHVISION_CPU),
    DiscoveryComponent("RetinaNetHead", "dense", "unvalidated", required_dependencies=_TORCHVISION_CPU),
    DiscoveryComponent("FreeAnchorRetinaHead", "dense", "runtime_validated", ("free_anchor", "free_anchor_head", "free_anchor_retina_head"), _TORCHVISION_CPU),
    DiscoveryComponent("FCOSHead", "dense", "runtime_validated", ("fcos", "fcos_head"), _TORCH_CPU),
    DiscoveryComponent("FCOSV2Head", "dense", "unvalidated", required_dependencies=_TORCH_CPU),
    DiscoveryComponent("FCOSHeadV2", "dense", "unvalidated", required_dependencies=_TORCH_CPU),
    DiscoveryComponent("FSAFHead", "dense", "runtime_validated", ("fsaf", "fsaf_head"), _TORCH_CPU),
    DiscoveryComponent("ATSSHead", "dense", "runtime_validated", ("atss", "atss_head"), _TORCH_CPU),
    DiscoveryComponent("ATSSV2Head", "dense", "unvalidated", required_dependencies=_TORCH_CPU),
    DiscoveryComponent("RPNHead", "dense", "runtime_validated", ("rpn", "rpn_head"), _TORCH_CPU),
    DiscoveryComponent("VFNetHead", "dense", "runtime_validated", ("VFNet", "vfnet_head"), _TORCH_CPU),
    DiscoveryComponent("RepPointsHead", "dense", "runtime_validated", ("RepPoints", "reppoints_head"), _TORCH_CPU),
    DiscoveryComponent("FoveaHead", "dense", "runtime_validated", ("FOVEA", "fovea_head"), _TORCH_CPU),
    DiscoveryComponent("YOLOFHead", "dense", "runtime_validated", ("YOLOF", "yolof_head"), _TORCH_CPU),
    DiscoveryComponent("YOLOXHead", "dense", "runtime_validated", ("yolox", "yolox_head"), _TORCH_CPU),
    DiscoveryComponent("YOLOHead", "dense", "unvalidated", required_dependencies=_TORCH_CPU),
    DiscoveryComponent("RTMDetHead", "dense", "runtime_validated", ("rtmdet", "rtmdet_head"), _TORCH_CPU),
    DiscoveryComponent("SSDHead", "dense", "runtime_validated", ("ssd", "ssd_head", "ssd300", "ssd512", "ssdlite"), _TORCH_CPU),
    DiscoveryComponent("EfficientDetHead", "dense", "runtime_validated", ("efficientdet", "efficientdet_head"), _TORCH_CPU),
    DiscoveryComponent("TOODHead", "dense", "runtime_validated", ("TOOD", "tood_head"), _TORCH_CPU),
    DiscoveryComponent("SOLOV2Head", "dense", "unvalidated", required_dependencies=_TORCH_CPU),
    DiscoveryComponent("CenterNetHead", "dense", "runtime_validated", ("CenterNet", "centernet_head"), _TORCH_CPU),
    DiscoveryComponent("CornerNetHead", "dense", "runtime_validated", ("CornerNet", "cornernet_head"), _TORCH_CPU),
    DiscoveryComponent("GFLHead", "dense", "runtime_validated", ("gfl", "gfl_head", "gfocal", "gfocal_head"), _TORCH_CPU),
    DiscoveryComponent("GFLV2Head", "dense", "runtime_validated", ("GFLV2", "gflv2_head", "gfocalv2", "gfocalv2_head"), _TORCH_CPU),
    DiscoveryComponent("PAAHead", "dense", "runtime_validated", ("PAA", "paa_head"), _TORCH_CPU),
    DiscoveryComponent("DDODHead", "dense", "runtime_validated", ("DDOD", "ddod_head"), _TORCH_CPU),
    DiscoveryComponent("AutoAssignHead", "dense", "runtime_validated", ("AutoAssign", "auto_assign_head"), _TORCH_CPU),
    DiscoveryComponent("NASFCOSHead", "dense", "runtime_validated", ("NAS-FCOS", "nas_fcos_head"), _TORCH_CPU),
    DiscoveryComponent("Shared2FCBBoxHead", "roi", "runtime_validated", ("shared_2fc_bbox_head", "shared2fc"), _TORCH_CPU),
    DiscoveryComponent("ConvFCBBoxHead", "roi", "runtime_validated", ("convfc_bbox_head", "convfc"), _TORCH_CPU),
    DiscoveryComponent("DoubleConvFCBBoxHead", "roi", "runtime_validated", ("double_convfc_bbox_head",), _TORCH_CPU),
    DiscoveryComponent("DynamicBBoxHead", "roi", "runtime_validated", ("dynamic_bbox_head", "dynamic_bbox"), _TORCH_CPU),
    DiscoveryComponent("CascadeBBoxHead", "roi", "runtime_validated", ("cascade_bbox_head", "cascade_bbox"), _TORCH_CPU),
    DiscoveryComponent("SABLHead", "roi", "runtime_validated", ("sabl_head", "sabl_bbox_head", "side_aware_bbox_head"), _TORCH_CPU),
    DiscoveryComponent("SparseRoIHead", "roi", "runtime_validated", ("sparse_roi_head", "sparse_bbox_head", "sparse_rcnn_roi_head"), _TORCH_CPU),
    DiscoveryComponent("FCNMaskHead", "roi", "runtime_validated", ("fcn_mask_head", "fcn_mask", "mask_head"), _TORCH_CPU),
    DiscoveryComponent("CascadeMaskHead", "roi", "runtime_validated", ("cascade_mask_head", "cascade_mask"), _TORCH_CPU),
    DiscoveryComponent("GridHead", "roi", "runtime_validated", ("grid_head", "grid_roi_head"), _TORCH_CPU),
    DiscoveryComponent("DETRHead", "transformer", "runtime_validated", ("DETR", "detr_head"), _TORCH_CPU),
    DiscoveryComponent("ConditionalDETRHead", "transformer", "runtime_validated", ("ConditionalDETR", "conditional_detr_head"), _TORCH_CPU),
    DiscoveryComponent("DABDETRHead", "transformer", "runtime_validated", ("DAB-DETR", "dab_detr_head"), _TORCH_CPU),
    DiscoveryComponent("DeformableDETRHead", "transformer", "runtime_validated", ("DeformableDETR", "deformable_detr_head"), _TORCH_CPU),
    DiscoveryComponent("DINOHead", "transformer", "runtime_validated", ("DINO", "dino_head"), _TORCH_CPU),
)


_STATIC_NECKS: tuple[DiscoveryComponent, ...] = (
    DiscoveryComponent("FPN", "neck", "runtime_validated", ("fpn",), _TORCHVISION_CPU),
    DiscoveryComponent("ChannelMapper", "neck", "runtime_validated", ("channel_mapper",), _TORCH_CPU),
    DiscoveryComponent("PAN", "neck", "runtime_validated", required_dependencies=_TORCHVISION_CPU),
    DiscoveryComponent("PANET", "neck", "runtime_validated", ("panet_neck",), _TORCHVISION_CPU),
    DiscoveryComponent("BiFPN", "neck", "runtime_validated", ("bi_fpn",), _TORCHVISION_CPU),
    DiscoveryComponent("BiFPNV2", "neck", "runtime_validated", ("bi_fpn_v2",), _TORCHVISION_CPU),
    DiscoveryComponent("PAFPN", "neck", "runtime_validated", ("pa_fpn",), _TORCHVISION_CPU),
    DiscoveryComponent("NASFPN", "neck", "runtime_validated", ("nas_fpn",), _TORCHVISION_CPU),
    DiscoveryComponent("FPNLite", "neck", "runtime_validated", required_dependencies=_TORCHVISION_CPU),
    DiscoveryComponent("FPNLiteNeck", "neck", "runtime_validated", required_dependencies=_TORCHVISION_CPU),
    DiscoveryComponent("DilatedEncoder", "neck", "runtime_validated", ("dilated_encoder",), _TORCH_CPU),
    DiscoveryComponent("HRFPN", "neck", "runtime_validated", ("hr_fpn",), _TORCH_CPU),
    DiscoveryComponent("SSDNeck", "neck", "runtime_validated", ("ssd_neck",), _TORCH_CPU),
    DiscoveryComponent("YOLOXPAFPN", "neck", "runtime_validated", ("yolox_pafpn",), _TORCHVISION_CPU),
)


_EXTRA_REQUIREMENTS: tuple[DiscoveryComponent, ...] = (
    DiscoveryComponent("cpu", "extra", "optional", required_dependencies=(
        Requirement("torch", "cpu"),
        Requirement("torchvision", "cpu"),
        Requirement("pytorch_lightning", "cpu"),
        Requirement("numpy", "cpu"),
        Requirement("scipy", "cpu"),
        Requirement("pycocotools", "cpu"),
        Requirement("terminaltables", "cpu"),
    )),
    DiscoveryComponent("timm", "extra", "optional", required_dependencies=(Requirement("timm", "timm"),)),
    DiscoveryComponent("geo", "extra", "optional", required_dependencies=(
        Requirement("pandas", "geo"),
        Requirement("rasterio", "geo"),
        Requirement("geopandas", "geo"),
        Requirement("shapely", "geo"),
    )),
    DiscoveryComponent("plots", "extra", "optional", required_dependencies=(
        Requirement("scienceplots", "plots"),
        Requirement("matplotlib", "plots"),
    )),
    DiscoveryComponent("docs", "extra", "optional"),
)


def detector_rows(family: str | None = None, pattern: str | None = None) -> list[DiscoveryRow]:
    family_filter = _normalize_filter(family)
    pattern_filter = _normalize_filter(pattern)
    components = _native_registry_components("detector")
    if components is None:
        components = tuple(
            DiscoveryComponent(
                name=name,
                kind=detector_family,
                validation_status="metadata_validated",
                required_dependencies=_TORCH_CPU,
            )
            for name, detector_family in ARCHITECTURE_FAMILIES.items()
        )
    return _component_rows(components, kind=family_filter, pattern=pattern_filter)


def head_rows(kind: str | None = None, pattern: str | None = None) -> list[DiscoveryRow]:
    kind_filter = _normalize_filter(kind)
    pattern_filter = _normalize_filter(pattern)
    components = _native_registry_components("head") or _STATIC_HEADS
    return _component_rows(components, kind=kind_filter, pattern=pattern_filter)


def backbone_rows(pattern: str | None = None) -> list[DiscoveryRow]:
    pattern_filter = _normalize_filter(pattern)
    components = tuple(
        DiscoveryComponent(
            name=backbone.name,
            kind=backbone.family,
            validation_status="metadata_validated",
            aliases=backbone.aliases,
            required_dependencies=_TIMM_BACKBONE,
        )
        for backbone in BACKBONE_ALIASES
    )
    return _component_rows(components, pattern=pattern_filter)


def neck_rows(pattern: str | None = None) -> list[DiscoveryRow]:
    pattern_filter = _normalize_filter(pattern)
    components = _native_registry_components("neck") or _STATIC_NECKS
    return _component_rows(components, pattern=pattern_filter)


def dataset_rows(pattern: str | None = None) -> list[DiscoveryRow]:
    pattern_filter = _normalize_filter(pattern)
    rows = [
        DiscoveryRow(
            name=name,
            kind="dataset",
            validation_status="runtime_validated",
        )
        for name in list_formats()
    ]
    return _filter_rows(rows, pattern=pattern_filter)


def extra_rows() -> list[DiscoveryRow]:
    rows: list[DiscoveryRow] = []
    for extra in _EXTRA_REQUIREMENTS:
        missing = [
            requirement.module
            for requirement in extra.required_dependencies
            if importlib_util.find_spec(requirement.module) is None
        ]
        status = "installed" if not missing else "missing"
        rows.append(
            DiscoveryRow(
                name=extra.name,
                kind=extra.kind,
                validation_status=status,
                required_extras=extra.required_extras,
                required_dependencies=tuple(missing),
            )
        )
    return sorted(rows, key=lambda row: row.name)


def _native_registry_components(kind: str) -> tuple[DiscoveryComponent, ...] | None:
    try:
        from .suite.catalog import _native_registries_available, _native_registry

        if not _native_registries_available():
            return None
        registry = _native_registry(kind)
    except ImportError:
        return None

    components: list[DiscoveryComponent] = []
    for metadata in registry.entries():
        dependencies = tuple(
            Requirement(dependency.module, dependency.extra)
            for dependency in metadata.required_dependencies
        )
        components.append(
            DiscoveryComponent(
                name=metadata.name,
                aliases=metadata.aliases,
                kind=metadata.family or metadata.kind,
                validation_status=metadata.validation_status,
                required_dependencies=dependencies,
            )
        )
    return tuple(components)


def _component_rows(
    components: Iterable[DiscoveryComponent],
    *,
    kind: str | None = None,
    pattern: str | None = None,
) -> list[DiscoveryRow]:
    rows: list[DiscoveryRow] = []
    seen: set[str] = set()
    for component in components:
        if kind is not None and component.kind.lower() != kind:
            continue
        for name in (component.name, *component.aliases):
            if pattern is not None and pattern not in name.lower():
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                DiscoveryRow(
                    name=name,
                    kind=component.kind,
                    validation_status=component.validation_status,
                    required_extras=component.required_extras,
                    required_dependencies=tuple(
                        requirement.module for requirement in component.required_dependencies
                    ),
                )
            )
    return sorted(rows, key=lambda row: row.name.lower())


def _filter_rows(rows: Iterable[DiscoveryRow], *, pattern: str | None = None) -> list[DiscoveryRow]:
    if pattern is None:
        return sorted(rows, key=lambda row: row.name.lower())
    return sorted(
        (row for row in rows if pattern in row.name.lower()),
        key=lambda row: row.name.lower(),
    )


def _normalize_filter(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    return token or None
