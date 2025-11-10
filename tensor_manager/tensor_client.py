#!/usr/bin/env python3

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch

try:
    import cupy as cp  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - cupy may be unavailable on CPU-only systems
    cp = None

import numpy as np

from .loader import (
    DEFAULT_SAMPLE_RATE,
    LoadedTensor,
    load_tensor_from_vcon,
    normalize_vcon,
)

logger = logging.getLogger(__name__)


def _parse_tensor_ref(ref: Optional[str]) -> Optional[str]:
    if not ref:
        return None
    return ref.split("tensor://", 1)[1] if ref.startswith("tensor://") else ref


def _device_to_index(device: str) -> int:
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except Exception:
            return 0
    return 0


def _normalise_device(device: Optional[Any]) -> str:
    if device is None:
        return "cpu"
    if isinstance(device, torch.device):
        if device.type == "cuda":
            index = 0 if device.index is None else device.index
            return f"cuda:{index}"
        return device.type
    if isinstance(device, int):
        return f"cuda:{device}"
    device = str(device)
    if device == "cuda":
        return "cuda:0"
    return device


class TensorClient:
    """Lightweight tensor manager client with optional remote acceleration."""

    _SERVER_CHECK_INTERVAL = 30.0

    def __init__(self, host: str = "127.0.0.1", port: int = 8003):
        self.host = host
        self.port = port
        self.server_url = f"http://{self.host}:{self.port}"
        self._server_state = {"available": None, "checked_at": 0.0}

    # ------------------------------------------------------------------ #
    # HTTP helpers
    # ------------------------------------------------------------------ #

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.server_url}{path}"
        timeout = kwargs.pop("timeout", 30)
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
        except requests.exceptions.RequestException as exc:
            logger.debug("Tensor manager request %s %s failed: %s", method, url, exc)
            self._mark_server_unavailable()
            raise
        return response

    def _ensure_server_available(self) -> bool:
        now = time.time()
        state = self._server_state
        if state["available"] is False and (now - state["checked_at"]) < self._SERVER_CHECK_INTERVAL:
            return False
        if state["available"] is True and (now - state["checked_at"]) < self._SERVER_CHECK_INTERVAL:
            return True

        try:
            response = self._request("GET", "/", timeout=2)
            ok = response.status_code == 200
        except Exception:
            ok = False

        state["available"] = ok
        state["checked_at"] = now
        return ok

    def _mark_server_unavailable(self) -> None:
        self._server_state["available"] = False
        self._server_state["checked_at"] = time.time()

    def is_server_available(self) -> bool:
        """Public helper so callers can check availability without loading tensors."""
        return self._ensure_server_available()

    # ------------------------------------------------------------------ #
    # Core HTTP API wrappers
    # ------------------------------------------------------------------ #

    def get_tensor_handle(self, tensor_id: str) -> dict:
        response = self._request("GET", f"/tensors/{tensor_id}/handle")
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get tensor handle: {response.text}")
        return response.json()

    def access_shared_tensor(self, tensor_id: str) -> torch.Tensor:
        handle_data = self.get_tensor_handle(tensor_id)

        shape = tuple(handle_data["shape"])
        dtype_str = handle_data.get("dtype", "float32")
        device_str = handle_data.get("device", "cuda:0")

        ipc_handle = handle_data.get("ipc_handle")
        has_cuda = torch.cuda.is_available() and cp is not None

        torch_dtype_map = {
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
            "torch.int32": torch.int32,
            "torch.int64": torch.int64,
            "torch.float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "float16": torch.float16,
        }

        if ipc_handle and has_cuda:
            data_ptr = int(handle_data["data_ptr"])
            nbytes = int(handle_data["nbytes"])

            dev_index = _device_to_index(device_str)

            torch.cuda.set_device(dev_index)
            cp.cuda.Device(dev_index).use()

            cupy_dtype_map = {
                "torch.float32": cp.float32,
                "torch.float64": cp.float64,
                "torch.int32": cp.int32,
                "torch.int64": cp.int64,
                "torch.float16": cp.float16,
                "float32": cp.float32,
                "float64": cp.float64,
                "int32": cp.int32,
                "int64": cp.int64,
                "float16": cp.float16,
            }
            t_dtype = torch_dtype_map.get(dtype_str, torch.float32)
            cp_dtype = cupy_dtype_map.get(dtype_str, cp.float32)

            unowned = cp.cuda.UnownedMemory(data_ptr, nbytes, None)
            memptr = cp.cuda.MemoryPointer(unowned, 0)
            cupy_arr = cp.ndarray(shape, dtype=cp_dtype, memptr=memptr)

            return torch.utils.dlpack.from_dlpack(cupy_arr).to(t_dtype)

        data_b64 = handle_data.get("data_b64")
        if data_b64 is None:
            if ipc_handle and cp is None:
                raise RuntimeError("CuPy is required to consume CUDA IPC handles, but it is not available.")
            raise RuntimeError("Tensor handle did not include a transferable payload.")

        numpy_dtype_map = {
            "torch.float32": np.float32,
            "torch.float64": np.float64,
            "torch.int32": np.int32,
            "torch.int64": np.int64,
            "torch.float16": np.float16,
            "float32": np.float32,
            "float64": np.float64,
            "int32": np.int32,
            "int64": np.int64,
            "float16": np.float16,
        }
        t_dtype = torch_dtype_map.get(dtype_str, torch.float32)
        np_dtype = numpy_dtype_map.get(dtype_str, np.float32)

        raw = base64.b64decode(data_b64)
        np_array = np.frombuffer(raw, dtype=np_dtype).copy()
        if shape:
            np_array = np_array.reshape(shape)

        tensor = torch.from_numpy(np_array).to(t_dtype)
        if isinstance(device_str, str) and device_str.startswith("cuda") and torch.cuda.is_available():
            tensor = tensor.to(device_str)
        return tensor

    def upload_wav_file(self, wav_file_path: str, cuda_device: int) -> str:
        with open(wav_file_path, "rb") as f:
            files = {"wav_file": (wav_file_path, f, "audio/wav")}
            data = {"cuda_device": cuda_device}
            response = self._request("POST", "/tensors", files=files, data=data)

        if response.status_code != 200:
            raise RuntimeError(f"Failed to upload WAV file: {response.text}")

        return response.json()["tensor_id"]

    def upload_cpu_tensor(self, tensor: torch.Tensor, sample_rate: int, cuda_device: int = 0) -> str:
        if tensor.is_cuda:
            tensor = tensor.cpu()
        arr = tensor.contiguous().numpy()
        raw = arr.tobytes()
        b64 = base64.b64encode(raw).decode("utf-8")
        payload = {
            "cuda_device": cuda_device,
            "sample_rate": sample_rate,
            "data_b64": b64,
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
        }
        response = self._request("POST", "/tensors/from-array", json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to upload tensor: {response.text}")
        return response.json()["tensor_id"]

    def delete_tensor(self, tensor_id: str) -> bool:
        response = self._request("DELETE", f"/tensors/{tensor_id}")
        if response.status_code == 200:
            return True
        if response.status_code == 404:
            return False
        raise RuntimeError(f"Failed to delete tensor {tensor_id}: {response.text}")

    def list_tensors(self) -> list:
        response = self._request("GET", "/tensors")
        if response.status_code != 200:
            raise RuntimeError(f"Failed to list tensors: {response.text}")
        return response.json()["tensors"]

    # ------------------------------------------------------------------ #
    # High-level operations
    # ------------------------------------------------------------------ #

    def get_or_lookup_tensor_by_vcon(self, vcon: Any, device: Optional[Any] = None) -> torch.Tensor:
        device_str = _normalise_device(device)
        vcon_dict, assign_back = normalize_vcon(vcon)
        vcon_uuid = vcon_dict.get("uuid")
        if not vcon_uuid:
            raise ValueError("vCon is missing a uuid")

        if device_str.startswith("cuda"):
            if self._ensure_server_available():
                tensor = self._get_tensor_remote(vcon_dict, assign_back, device_str)
                if assign_back:
                    assign_back(vcon_dict)
                return tensor
            logger.debug("Tensor manager unavailable; falling back to local tensor for %s", vcon_uuid)

        tensor = self._get_tensor_local(vcon_dict, assign_back, device_str)
        if assign_back:
            assign_back(vcon_dict)
        return tensor

    def vcon_to_tensor(self, vcon: Any, device: Optional[Any] = None) -> torch.Tensor:
        return self.get_or_lookup_tensor_by_vcon(vcon, device=device)

    def ensure_remote_tensor(self, vcon: Any, device: Optional[Any] = None) -> Optional[torch.Tensor]:
        """Materialise the vCon's audio tensor on the tensor-manager server if available."""
        if not self._ensure_server_available():
            return None
        device_str = _normalise_device(device or "cuda:0")
        vcon_dict, assign_back = normalize_vcon(vcon)
        tensor = self._get_tensor_remote(vcon_dict, assign_back, device_str)
        if assign_back:
            assign_back(vcon_dict)
        return tensor

    def unload_remote_tensor(self, vcon: Any) -> int:
        """Remove any GPU tensors associated with the vCon from the tensor manager."""
        if not self._ensure_server_available():
            return 0
        return self.delete_tensor_by_vcon(vcon)

    # ------------------------------------------------------------------ #
    # Remote helpers
    # ------------------------------------------------------------------ #

    def _get_tensor_remote(
        self,
        vcon_dict: Dict[str, Any],
        assign_back: Optional[callable],
        device_str: str,
    ) -> torch.Tensor:
        dialogs: List[Dict[str, Any]] = vcon_dict.get("dialog", []) or []
        tensor_entry, entry_index = self._find_existing_tensor_entry(dialogs, device_str)
        tensor_id: Optional[str] = None

        if tensor_entry:
            tensor_id = _parse_tensor_ref(tensor_entry.get("tensor_ref"))
            if tensor_id:
                try:
                    tensor = self.access_shared_tensor(tensor_id)
                    if str(tensor.device) != device_str:
                        tensor = tensor.to(device_str)
                    return tensor
                except Exception as exc:
                    logger.debug("Existing tensor %s unavailable, removing entry: %s", tensor_id, exc)
                    dialogs.pop(entry_index)
                    if assign_back:
                        assign_back(vcon_dict)

        loaded = load_tensor_from_vcon(vcon_dict, device="cpu")
        tensor_cpu = loaded.tensor
        sample_rate = loaded.sample_rate or DEFAULT_SAMPLE_RATE

        cuda_index = _device_to_index(device_str)
        try:
            tensor_id = self.upload_cpu_tensor(tensor_cpu, sample_rate, cuda_device=cuda_index)
            remote_tensor = self.access_shared_tensor(tensor_id)
        except Exception as exc:
            logger.debug("Remote tensor upload/access failed, falling back to local: %s", exc)
            self._mark_server_unavailable()
            if device_str.startswith("cuda") and torch.cuda.is_available():
                return tensor_cpu.to(device_str)
            return tensor_cpu

        if str(remote_tensor.device) != device_str:
            remote_tensor = remote_tensor.to(device_str)

        entry = {
            "type": "tensor",
            "tensor_ref": f"tensor://{tensor_id}",
            "device": str(remote_tensor.device),
            "shape": list(remote_tensor.shape),
            "sample_rate": sample_rate,
        }
        dialogs.append(entry)
        vcon_dict["dialog"] = dialogs
        return remote_tensor

    def _find_existing_tensor_entry(
        self, dialogs: List[Dict[str, Any]], device_str: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        fallback_entry = None
        fallback_idx = None
        for idx in range(len(dialogs) - 1, -1, -1):
            entry = dialogs[idx]
            if entry.get("type") != "tensor":
                continue
            tensor_id = _parse_tensor_ref(entry.get("tensor_ref"))
            if not tensor_id:
                continue
            entry_device = entry.get("device")
            if entry_device == device_str:
                return entry, idx
            if fallback_entry is None and entry_device is None:
                fallback_entry, fallback_idx = entry, idx
        return fallback_entry, fallback_idx

    # ------------------------------------------------------------------ #
    # Local helpers
    # ------------------------------------------------------------------ #

    def _get_tensor_local(
        self,
        vcon_dict: Dict[str, Any],
        assign_back: Optional[callable],
        device_str: str,
    ) -> torch.Tensor:
        loaded: LoadedTensor = load_tensor_from_vcon(
            vcon_dict, device=device_str if device_str != "cpu" else None
        )
        tensor = loaded.tensor
        if device_str.startswith("cuda") and torch.cuda.is_available():
            tensor = tensor.to(device_str)
        return tensor

    # ------------------------------------------------------------------ #
    # Removal helpers
    # ------------------------------------------------------------------ #

    def delete_tensor_by_vcon(self, vcon: Any) -> int:
        vcon_dict, assign_back = normalize_vcon(vcon)
        dialogs: List[Dict[str, Any]] = vcon_dict.get("dialog", []) or []
        deleted = 0
        kept: List[Dict[str, Any]] = []

        for entry in dialogs:
            if entry.get("type") == "tensor":
                tensor_id = _parse_tensor_ref(entry.get("tensor_ref"))
                if tensor_id:
                    try:
                        if self.delete_tensor(tensor_id):
                            deleted += 1
                    except Exception:
                        pass
                continue
            kept.append(entry)

        if deleted:
            vcon_dict["dialog"] = kept
            if assign_back:
                assign_back(vcon_dict)
        return deleted

    def remove_loaded_tensor(self, vcon: Any) -> int:
        """Remove any tensor references from a vCon without contacting the server."""
        vcon_dict, assign_back = normalize_vcon(vcon)
        dialogs: List[Dict[str, Any]] = vcon_dict.get("dialog", []) or []
        kept: List[Dict[str, Any]] = []
        removed = 0
        for entry in dialogs:
            if entry.get("type") == "tensor":
                removed += 1
                continue
            kept.append(entry)
        if removed:
            vcon_dict["dialog"] = kept
            if assign_back:
                assign_back(vcon_dict)
        return removed


def example_usage():
    print("=== Tensor Client Example ===\n")

    client = TensorClient()

    sample_rate = 16000
    duration = 1.0
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    try:
        import soundfile as sf  # type: ignore[import-not-found]
    except ImportError:
        print("soundfile is not available; skipping example usage.")
        return
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        sf.write(tmp_file.name, audio_data, sample_rate)
        wav_path = tmp_file.name

    try:
        print("1. Uploading WAV file to server...")
        tensor_id = client.upload_wav_file(wav_path, cuda_device=0)
        print(f"   Tensor ID: {tensor_id}")

        print("\n2. Accessing shared tensor from this process...")
        shared_tensor = client.access_shared_tensor(tensor_id)
        print(f"   Shared tensor shape: {shared_tensor.shape}")
        print(f"   Shared tensor device: {shared_tensor.device}")
        print(f"   Shared tensor dtype: {shared_tensor.dtype}")
        print(f"   Data range: [{shared_tensor.min().item():.3f}, {shared_tensor.max().item():.3f}]")

        print("\n3. Demonstrating zero-copy access...")
        original_mean = shared_tensor.mean().item()
        print(f"   Original mean: {original_mean:.6f}")

        shared_tensor.mul_(2.0)
        new_mean = shared_tensor.mean().item()
        print(f"   After in-place multiplication by 2: {new_mean:.6f}")
        print(f"   Ratio (should be ~2.0): {new_mean/original_mean:.6f}")

        print("\n   SUCCESS: Tensor was modified in-place on GPU!")
        print("   Any other process accessing this tensor will see the modified data.")

        print("\n4. Cleanup...")
        client.delete_tensor(tensor_id)
        print("   Tensor deleted from server")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass

    print("\n=== Example completed ===")


if __name__ == "__main__":
    example_usage()
