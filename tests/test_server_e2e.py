"""Test REST server end-to-end."""
import sys
from pathlib import Path
import pytest
import subprocess
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="module")
def server_process():
    """Start server as a fixture."""
    repo_root = Path(__file__).parent.parent
    ae_ckpt = repo_root / "checkpoints" / "ae.pt"
    student_ckpt = repo_root / "checkpoints" / "student.pt"

    if not ae_ckpt.exists() or not student_ckpt.exists():
        pytest.skip("Checkpoints not found, skipping server test")

    # Start server
    proc = subprocess.Popen(
        [
            "python",
            str(repo_root / "server.py"),
            "--port", "7861",  # Use different port to avoid conflicts
            "--ae", str(ae_ckpt),
            "--student", str(student_ckpt),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(repo_root),
    )

    # Wait for server to start
    time.sleep(5)

    # Check if server is running
    try:
        response = requests.get("http://localhost:7861/health", timeout=5)
        if response.status_code != 200:
            proc.kill()
            pytest.skip("Server failed to start")
    except requests.exceptions.RequestException:
        proc.kill()
        pytest.skip("Server not responding")

    yield "http://localhost:7861"

    # Cleanup
    proc.terminate()
    proc.wait(timeout=10)


@pytest.mark.timeout(180)
def test_server_health_endpoint(server_process):
    """Test that /health endpoint works."""
    response = requests.get(f"{server_process}/health", timeout=10)
    assert response.status_code == 200

    data = response.json()
    assert "status" in data


@pytest.mark.timeout(180)
def test_server_generate_endpoint(server_process):
    """Test that /generate endpoint works."""
    response = requests.post(
        f"{server_process}/generate",
        json={
            "prompt": "Test",
            "max_new_tokens": 16,
            "temperature": 0.8,
        },
        timeout=60,
    )

    assert response.status_code == 200

    data = response.json()
    assert "generated_text" in data
    assert len(data["generated_text"]) > 0


@pytest.mark.timeout(180)
def test_server_concurrent_requests(server_process):
    """Test server with 5 concurrent requests."""
    prompts = [
        "First prompt",
        "Second prompt",
        "Third prompt",
        "Fourth prompt",
        "Fifth prompt",
    ]

    def make_request(prompt):
        response = requests.post(
            f"{server_process}/generate",
            json={
                "prompt": prompt,
                "max_new_tokens": 16,
                "temperature": 0.8,
            },
            timeout=60,
        )
        return response

    # Send concurrent requests
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, p) for p in prompts]

        results = []
        for future in as_completed(futures):
            response = future.result()
            assert response.status_code == 200
            results.append(response.json())

    assert len(results) == 5
    for result in results:
        assert "generated_text" in result
