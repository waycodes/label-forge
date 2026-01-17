"""
Environment capture for reproducible runs.

Captures all relevant environment information to enable run reconstruction.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from labelforge.core.json_canonical import canonical_json_dumps


@dataclass
class GPUInfo:
    """Information about a GPU device."""

    index: int
    name: str
    memory_total_mb: int
    driver_version: str
    cuda_version: str | None = None


@dataclass
class EnvironmentSnapshot:
    """
    Complete environment snapshot for a run.

    Contains all information needed to reproduce the execution environment.
    """

    # Timing
    captured_at: datetime = field(default_factory=datetime.utcnow)

    # Platform
    platform_system: str = ""
    platform_release: str = ""
    platform_machine: str = ""
    python_version: str = ""
    python_executable: str = ""

    # Package versions
    pip_freeze: list[str] = field(default_factory=list)
    ray_version: str | None = None
    vllm_version: str | None = None

    # GPU/CUDA
    cuda_version: str | None = None
    cudnn_version: str | None = None
    gpu_devices: list[GPUInfo] = field(default_factory=list)
    gpu_count: int = 0

    # Git info
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool = False

    # Environment variables (filtered)
    env_vars: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "captured_at": self.captured_at.isoformat() + "Z",
            "platform": {
                "system": self.platform_system,
                "release": self.platform_release,
                "machine": self.platform_machine,
            },
            "python": {
                "version": self.python_version,
                "executable": self.python_executable,
            },
            "packages": {
                "pip_freeze": self.pip_freeze,
                "ray_version": self.ray_version,
                "vllm_version": self.vllm_version,
            },
            "gpu": {
                "cuda_version": self.cuda_version,
                "cudnn_version": self.cudnn_version,
                "gpu_count": self.gpu_count,
                "devices": [
                    {
                        "index": gpu.index,
                        "name": gpu.name,
                        "memory_total_mb": gpu.memory_total_mb,
                        "driver_version": gpu.driver_version,
                        "cuda_version": gpu.cuda_version,
                    }
                    for gpu in self.gpu_devices
                ],
            },
            "git": {
                "commit": self.git_commit,
                "branch": self.git_branch,
                "dirty": self.git_dirty,
            },
            "env_vars": self.env_vars,
        }

    def to_json(self, indent: bool = True) -> str:
        """Serialize to canonical JSON."""
        return canonical_json_dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        """Save snapshot to file."""
        path.write_text(self.to_json(indent=True))


def capture_environment(
    include_env_vars: list[str] | None = None,
    working_dir: Path | None = None,
) -> EnvironmentSnapshot:
    """
    Capture a complete environment snapshot.

    Args:
        include_env_vars: List of environment variable prefixes to capture.
            Defaults to common ML-related variables.
        working_dir: Directory to check for git info. Defaults to cwd.

    Returns:
        Complete environment snapshot.
    """
    if include_env_vars is None:
        include_env_vars = [
            "CUDA_",
            "VLLM_",
            "RAY_",
            "HF_",
            "HUGGINGFACE_",
            "TORCH_",
            "NCCL_",
        ]

    if working_dir is None:
        working_dir = Path.cwd()

    snapshot = EnvironmentSnapshot(
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        python_version=platform.python_version(),
        python_executable=sys.executable,
    )

    # Capture pip freeze
    snapshot.pip_freeze = _capture_pip_freeze()

    # Extract key versions
    snapshot.ray_version = _get_package_version("ray")
    snapshot.vllm_version = _get_package_version("vllm")

    # Capture GPU info
    gpu_info = _capture_gpu_info()
    snapshot.cuda_version = gpu_info.get("cuda_version")
    snapshot.cudnn_version = gpu_info.get("cudnn_version")
    snapshot.gpu_devices = gpu_info.get("devices", [])
    snapshot.gpu_count = len(snapshot.gpu_devices)

    # Capture git info
    git_info = _capture_git_info(working_dir)
    snapshot.git_commit = git_info.get("commit")
    snapshot.git_branch = git_info.get("branch")
    snapshot.git_dirty = git_info.get("dirty", False)

    # Capture filtered environment variables
    snapshot.env_vars = _capture_env_vars(include_env_vars)

    return snapshot


def _capture_pip_freeze() -> list[str]:
    """Capture pip freeze output."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return sorted(result.stdout.strip().split("\n"))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return []


def _get_package_version(package_name: str) -> str | None:
    """Get version of an installed package."""
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return None


def _capture_gpu_info() -> dict[str, Any]:
    """Capture GPU and CUDA information."""
    result: dict[str, Any] = {"devices": []}

    # Try to get CUDA version from environment
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        nvcc_path = Path(cuda_home) / "bin" / "nvcc"
        if nvcc_path.exists():
            try:
                output = subprocess.run(
                    [str(nvcc_path), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if output.returncode == 0:
                    # Parse CUDA version from nvcc output
                    for line in output.stdout.split("\n"):
                        if "release" in line.lower():
                            parts = line.split("release")
                            if len(parts) > 1:
                                version_str = parts[1].strip().split(",")[0]
                                result["cuda_version"] = version_str
                                break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

    # Try nvidia-smi for GPU info
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if output.returncode == 0:
            for line in output.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    result["devices"].append(
                        GPUInfo(
                            index=int(parts[0]),
                            name=parts[1],
                            memory_total_mb=int(parts[2]),
                            driver_version=parts[3],
                        )
                    )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return result


def _capture_git_info(working_dir: Path) -> dict[str, Any]:
    """Capture git repository information."""
    result: dict[str, Any] = {}

    try:
        # Get current commit
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=working_dir,
            timeout=10,
        )
        if output.returncode == 0:
            result["commit"] = output.stdout.strip()

        # Get current branch
        output = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=working_dir,
            timeout=10,
        )
        if output.returncode == 0:
            result["branch"] = output.stdout.strip()

        # Check if dirty
        output = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=working_dir,
            timeout=10,
        )
        if output.returncode == 0:
            result["dirty"] = bool(output.stdout.strip())

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return result


def _capture_env_vars(prefixes: list[str]) -> dict[str, str]:
    """Capture environment variables matching given prefixes."""
    result = {}
    for key, value in os.environ.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                result[key] = value
                break
    return dict(sorted(result.items()))
