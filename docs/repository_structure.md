# Repository Structure and Module Ownership

This document defines where to implement features in this repository.
Use it as the default placement rule for new code.

## Top-level layout

- `src/excvish/`: main package source code.
- `docs/`: repository documentation and development rules.
- `pyproject.toml`: dependencies, lint/type settings, and build configuration.
- `.pre-commit-config.yaml`: pre-commit hooks for file checks plus commit-msg lint (`make lint`, `commitlint`).
- `commitlint.config.cjs`: commit message/PR title lint rules (Conventional Commits).
- `.github/workflows/`: CI workflows, including PR title lint for squash-merge safety.

## Package-level modules (`src/excvish`)

### `src/excvish/types.py`

- Shared domain data structures used across modules.
- Current role: immutable `BBox` and geometry helpers attached to `BBox`.
- Implement here when adding reusable primitive types used by multiple subpackages.

### `src/excvish/__init__.py`

- Public package entrypoint.
- Exposes stable, package-wide utilities and types (`BBox`, `clip`, etc.).
- Add exports here only for APIs that should be imported as `import excvish`.

### `src/excvish/pil.py`

- PIL/Pillow-based image utilities.
- Responsibilities:
  - PIL image composition and canvas operations.
  - PIL resize/pad and text rendering.
  - PIL <-> NumPy bridge helpers when PIL is the primary input.
- Do not place OpenCV-specific logic here.

### `src/excvish/spectrum.py`

- Hyper-spectral or spectral-signal algorithms.
- Current role: Spectral Angle Mapper (SAM).
- Add spectral-domain methods here rather than in `cv/`.

## `src/excvish/cv/`

- Consolidated NumPy/OpenCV-based computer vision utilities.
- Includes color processing, geometry, edges, masking, labeling, drawing,
  matching, and metric computation.
- Add framework-agnostic CV algorithms here.
- Keep this layer independent from framework adapters such as Ultralytics-specific code.

## Integration layers

### `src/excvish/albumentations/`

- Adapter layer for Albumentations integration.
- `transforms.py`: custom transforms that follow Albumentations interfaces.
- `helpers.py`: transform inspection and composition utilities.
- Put augmentation framework-specific glue here; keep framework-agnostic math in `cv/`.

### `src/excvish/ultralytics/`

- Adapter layer for Ultralytics training pipelines.
- Responsibilities:
  - map data between Ultralytics `Instances` and augmentation code,
  - keep bbox/keypoint consistency after transforms,
  - apply Ultralytics-specific resize/padding behavior.
- Put Ultralytics-only logic here, not in generic `cv/` or `albumentations/` modules.

## Placement rules for new implementation

When adding new functionality, choose the location in this order:

1. Is it a shared primitive data model? -> `types.py`
2. Is it PIL-first image handling? -> `pil.py`
3. Is it spectral-domain processing? -> `spectrum.py`
4. Is it NumPy/OpenCV generic CV logic? -> `cv/`
5. Is it Albumentations-specific API glue? -> `albumentations/`
6. Is it Ultralytics-specific API glue? -> `ultralytics/`

## Dependency direction (recommended)

Keep dependencies one-way to avoid cyclic coupling:

- `types` and package utilities (`__init__.py`) are base layers.
- `cv/*` may depend on `types` and base utilities.
- `albumentations/*` may depend on `cv/*`.
- `ultralytics/*` may depend on `albumentations/*` and `cv/*`.
- Avoid reverse dependencies (for example, `cv/*` importing `ultralytics/*`).

## What to avoid

- Mixing framework glue (Albumentations/Ultralytics) into generic `cv/` modules.
- Duplicating identical logic across `pil.py` and `cv/*` when a shared helper can be extracted.
- Exporting unstable internal helpers from `src/excvish/__init__.py`.
