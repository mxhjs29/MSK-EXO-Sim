# Contributing

Thanks for contributing to `musclemimic`.

The README's [shared files and extension boundaries](README.md#文件职责与-contributor-扩展位置)
define the contribution policy. Do not edit the listed shared implementations to
add a task or experiment. Add modules and independent configs at the documented
extension points. Shared framework fixes and interface changes belong in separate
maintainer-reviewed PRs with appropriate regression checks. Registration additions
may add new entries without changing existing entries or discovery behavior.

See [repository structure and migration boundaries](docs/repository_structure.md)
before adding a task, reward, simulator component or controller. Put shared rewards
in `musclemimic/rewards/common/` and task additions in `rewards/tasks/`; maintain the
base clipping/state-update order unless the PR explicitly changes behavior.

The public repository excludes local research, datasets, outputs and learned
checkpoints. `local/assets/smpl/` and its compatibility symlink are the explicit
exception: binary model files use Git LFS and retain their bundled licenses. Run `python scripts/check_repository_scope.py` before submitting a PR.
Use short-lived topic branches, one change per PR, and review public-interface
changes with the affected module owners. CODEOWNERS currently contains examples;
real teams and hosted branch protection must be configured when contributors join.

## Development Setup

This repository targets Python 3.11 and uses `uv` for environment management.

```bash
make install-dev
make precommit-install
```

Optional extras:

- `uv sync --extra dev --extra cuda` for Linux x86_64 CUDA setups
- `uv sync --extra dev --extra smpl --extra gmr` for full SMPL/GMR retargeting workflows

## Local Checks

Use the `Makefile` targets for the default developer workflow:

```bash
make format
make lint
make test
make ci
```

Developer tooling is expected inside `.venv/bin`. `make lint`, `make test`, and `make smoke` use those local executables directly instead of spawning `uv run` each time.

By default, `make test` runs:

```bash
.venv/bin/pytest -m "not integration"
```

If you need a different subset, override `PYTEST_ARGS`:

```bash
make test PYTEST_ARGS='tests/unit/test_ppo_config.py -q'
```

## Pre-commit Scope

`pre-commit` is intentionally scoped to a curated subset of files in `.pre-commit-config.yaml`.

`make lint` and `make format` follow that same scoped set of Python paths. They are intentionally not whole-repository checks today.

This is a migration guardrail, not an accident. Please do not remove or broadly expand the `files:` allowlist in unrelated pull requests. Large formatting-only churn has previously broken working flows and made review difficult.

If you want to expand `pre-commit` coverage:

1. Do it in a dedicated cleanup PR.
2. Confirm the newly covered paths pass `ruff` and `pytest` as appropriate.
3. Keep the diff reviewable and separate from functional changes.

## Pull Requests

Before opening a PR:

1. Run `make ci`.
2. Update tests for behavior changes.
3. Update `README.md` or other docs when user-facing commands or workflows change.
4. Keep refactors separate from behavior changes when practical.

PRs should include:

- A short summary of the change
- The motivation or linked issue
- The validation you ran locally
- Screenshots or logs when they help reviewers

## Issues

Use the GitHub issue templates for bug reports and feature requests.

For bug reports, include:

- The command you ran
- Your platform and Python version
- Whether you used CPU, CUDA, MuJoCo, SMPL, or GMR extras
- The full traceback or failure log
