"""Prepare COCO detection annotations for native SimpleDet training."""

from __future__ import annotations

import argparse
import filecmp
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _materialize_image(source: Path, destination: Path, *, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.resolve() == destination.resolve():
            return
        try:
            if source.stat().st_ino == destination.stat().st_ino:
                return
        except OSError:
            pass
        if filecmp.cmp(source, destination, shallow=False):
            return
        raise FileExistsError(f"Refusing to replace existing image: {destination}")
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def _category_remap(categories: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, int]]:
    ordered = sorted(categories, key=lambda item: int(item["id"]))
    remap = {int(item["id"]): index for index, item in enumerate(ordered, start=1)}
    remapped_categories = [
        {**item, "id": remap[int(item["id"])]}
        for item in ordered
    ]
    return remapped_categories, remap


def _select_images(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    *,
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(images):
        return list(images)
    annotated_ids = {
        int(annotation["image_id"])
        for annotation in annotations
        if not int(annotation.get("iscrowd", 0) or 0)
    }
    candidates = [image for image in images if int(image["id"]) in annotated_ids]
    if limit > len(candidates):
        raise ValueError(
            f"Requested {limit} annotated images, but only {len(candidates)} are available."
        )
    rng = random.Random(seed)
    selected = rng.sample(candidates, limit)
    return sorted(selected, key=lambda item: int(item["id"]))


def _valid_bbox(annotation: dict[str, Any]) -> bool:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(value) for value in values) and values[2] > 0 and values[3] > 0


def _rewrite_split(
    source_annotation: Path,
    output_annotation: Path,
    *,
    source_images_dir: Path,
    output_images_root: Path,
    image_prefix: str,
    categories: list[dict[str, Any]],
    category_id_map: dict[int, int],
    image_limit: int | None,
    image_mode: str,
    seed: int,
) -> dict[str, int]:
    source = _load_json(source_annotation)
    raw_images = list(source.get("images", []))
    raw_annotations = list(source.get("annotations", []))
    selected_images = _select_images(raw_images, raw_annotations, limit=image_limit, seed=seed)
    selected_ids = {int(image["id"]) for image in selected_images}

    images = []
    for image in selected_images:
        file_name = str(image["file_name"])
        _materialize_image(
            source_images_dir / file_name,
            output_images_root / image_prefix / file_name,
            mode=image_mode,
        )
        images.append({**image, "file_name": f"{image_prefix}/{file_name}"})
    annotations = []
    skipped_annotations = 0
    for annotation in raw_annotations:
        image_id = int(annotation["image_id"])
        if image_id not in selected_ids:
            continue
        if not _valid_bbox(annotation):
            skipped_annotations += 1
            continue
        source_category_id = int(annotation["category_id"])
        if source_category_id not in category_id_map:
            skipped_annotations += 1
            continue
        annotations.append({**annotation, "category_id": category_id_map[source_category_id]})

    _write_json(
        output_annotation,
        {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        },
    )
    return {
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(categories),
        "skipped_annotations": skipped_annotations,
    }


def prepare_coco(
    source_root: Path,
    output_root: Path,
    *,
    train_limit: int | None,
    val_limit: int | None,
    test_limit: int | None,
    image_mode: str,
    seed: int,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    train_json = source_root / "annotations" / "instances_train2017.json"
    val_json = source_root / "annotations" / "instances_val2017.json"
    train_images = source_root / "train2017"
    val_images = source_root / "val2017"
    for path in (train_json, val_json, train_images, val_images):
        if not path.exists():
            raise FileNotFoundError(path)

    train_payload = _load_json(train_json)
    categories, category_id_map = _category_remap(list(train_payload.get("categories", [])))
    classes = [str(category["name"]) for category in categories]

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "image_mode": image_mode,
        "class_count": len(classes),
        "classes": classes,
        "splits": {
            "train": _rewrite_split(
                train_json,
                output_root / "annotations" / "train.json",
                source_images_dir=train_images,
                output_images_root=output_root / "images",
                image_prefix="train2017",
                categories=categories,
                category_id_map=category_id_map,
                image_limit=train_limit,
                image_mode=image_mode,
                seed=seed,
            ),
            "val": _rewrite_split(
                val_json,
                output_root / "annotations" / "val.json",
                source_images_dir=val_images,
                output_images_root=output_root / "images",
                image_prefix="val2017",
                categories=categories,
                category_id_map=category_id_map,
                image_limit=val_limit,
                image_mode=image_mode,
                seed=seed,
            ),
            "test": _rewrite_split(
                val_json,
                output_root / "annotations" / "test.json",
                source_images_dir=val_images,
                output_images_root=output_root / "images",
                image_prefix="val2017",
                categories=categories,
                category_id_map=category_id_map,
                image_limit=test_limit,
                image_mode=image_mode,
                seed=seed + 1,
            ),
        },
        "category_id_map": category_id_map,
    }
    _write_json(output_root / "simpledet-coco-summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--image-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--seed", type=int, default=71)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = prepare_coco(
        args.source_root,
        args.output_root,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        image_mode=args.image_mode,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
