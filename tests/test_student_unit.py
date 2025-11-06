"""Test student model unit functions."""
import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from student.student_model import StudentTransformer


@pytest.fixture
def toy_student():
    """Create a small student for testing."""
    return StudentTransformer(
        latent_dim=256,
        hidden_dim=512,
        num_layers=2,
        num_heads=4,
        max_seq_len=128,
    )


def test_student_forward_no_nan(toy_student):
    """Test that student forward produces no NaN."""
    batch_size = 4
    seq_len = 16
    latent_dim = toy_student.latent_dim

    # Random latents
    latents = torch.randn(batch_size, seq_len, latent_dim)

    # Forward
    output = toy_student(latents)

    # Check no NaN or inf
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains inf"


def test_student_optimizer_step_updates_parameters():
    """Test that one optimizer step produces finite losses and parameter updates."""
    toy_student = StudentTransformer(
        latent_dim=128,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
    )

    optimizer = torch.optim.Adam(toy_student.parameters(), lr=0.001)

    # Sample input and target
    batch_size = 2
    seq_len = 8
    latent_dim = toy_student.latent_dim

    input_latents = torch.randn(batch_size, seq_len, latent_dim)
    target_latents = torch.randn(batch_size, seq_len, latent_dim)

    # Get initial parameters
    initial_params = [p.clone() for p in toy_student.parameters()]

    # Forward
    predicted = toy_student(input_latents)

    # Compute loss (MSE)
    loss = nn.functional.mse_loss(predicted, target_latents)

    # Check finite loss
    assert torch.isfinite(loss), "Loss should be finite"

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check that parameters were updated
    params_updated = False
    for initial_param, current_param in zip(initial_params, toy_student.parameters()):
        if not torch.equal(initial_param, current_param):
            params_updated = True
            break

    assert params_updated, "At least one parameter should have been updated"
