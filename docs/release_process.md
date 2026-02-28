# Release Process

This repository enforces PR-only merges. Release operations follow that rule.

## Workflows

- `release-prepare.yml`:
  - Triggered manually from Actions (`workflow_dispatch`).
  - Validates the requested version.
  - Updates `pyproject.toml` (`project.version`).
  - Creates or updates a release PR targeting `main`.
- `release-publish.yml`:
  - Triggered on push to `main`.
  - Runs only when `pyproject.toml` changed in that push.
  - Reads `project.version`, creates `v<version>` tag, and publishes a GitHub Release.

## Standard flow

1. Run `release-prepare` with a version (example: `0.5.2`).
2. Review and merge the generated PR.
3. Confirm the `release-publish` workflow succeeded.
4. Verify release page contains `v0.5.2`.
