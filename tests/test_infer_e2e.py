"""Test end-to-end inference."""
import sys
from pathlib import Path
import pytest
import subprocess
import os

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.timeout(120)
def test_infer_returns_nonempty_text():
    """Test that infer.py returns non-empty text."""
    repo_root = Path(__file__).parent.parent
    ae_ckpt = repo_root / "checkpoints" / "ae.pt"
    student_ckpt = repo_root / "checkpoints" / "student.pt"

    if not ae_ckpt.exists() or not student_ckpt.exists():
        pytest.skip("Checkpoints not found, skipping e2e inference test")

    # Run inference
    result = subprocess.run(
        [
            "python",
            str(repo_root / "infer.py"),
            "--ae", str(ae_ckpt),
            "--student", str(student_ckpt),
            "--prompt", "Test prompt",
            "--max_new_tokens", "32",
            "--seed", "42",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(repo_root),
    )

    assert result.returncode == 0, f"infer.py failed: {result.stderr}"
    assert len(result.stdout) > 0, "infer.py should produce output"


@pytest.mark.timeout(120)
def test_infer_deterministic_with_seed():
    """Test that infer.py is deterministic with same seed."""
    repo_root = Path(__file__).parent.parent
    ae_ckpt = repo_root / "checkpoints" / "ae.pt"
    student_ckpt = repo_root / "checkpoints" / "student.pt"

    if not ae_ckpt.exists() or not student_ckpt.exists():
        pytest.skip("Checkpoints not found, skipping determinism test")

    # Run twice with same seed
    outputs = []
    for _ in range(2):
        result = subprocess.run(
            [
                "python",
                str(repo_root / "infer.py"),
                "--ae", str(ae_ckpt),
                "--student", str(student_ckpt),
                "--prompt", "Determinism test",
                "--max_new_tokens", "16",
                "--seed", "123",
                "--temperature", "0.0",  # Greedy
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(repo_root),
        )

        assert result.returncode == 0
        outputs.append(result.stdout)

    # Check outputs are identical
    assert outputs[0] == outputs[1], "Outputs should be deterministic with same seed"
