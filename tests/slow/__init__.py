"""Slow integration tests for BordAX.

These tests verify that algorithms actually learn over multiple training epochs.
They use production hyperparameters and can take 10-20 seconds to run.

Run with: pytest tests/slow/ -v
Skip with: pytest tests/ -m "not slow" -v
"""
