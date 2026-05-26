# Contributing to BordAX

Thank you for your interest in contributing to BordAX!

## Reporting Bugs

Please [open a GitHub Issue](https://github.com/SynthesisLab/bordax/issues/new/choose) using the **Bug Report** template. Include:

- A minimal reproducible example
- Your environment (OS, Python version, JAX version)
- The full error traceback

## Asking Questions

Open a [GitHub Issue](https://github.com/SynthesisLab/bordax/issues/new/choose) using the **Question** label.

## Proposing Changes

For non-trivial changes, open an Issue to discuss the idea before writing code. For small fixes (typos, documentation), you can go straight to a pull request.

## Development Setup

**Requirements**: Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/SynthesisLab/bordax.git
cd bordax
uv sync
```

Verify the setup:

```bash
uv run pytest tests/ -m "not slow" -v
```

## Code Style

- Format with [black](https://github.com/psf/black): `uv run black .`
- Use type hints throughout

## Running Tests

```bash
# Fast tests (run before every commit)
uv run pytest tests/ -m "not slow" -v

# Full suite including slow learning tests
uv run pytest tests/ -v
```

New code should come with tests.

## Submitting a Pull Request

1. Fork the repository and create a branch from `main`
2. Make your changes with tests
3. Ensure all fast tests pass: `uv run pytest tests/ -m "not slow"`
4. Open a pull request against `main` — fill out the PR template

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold it.
