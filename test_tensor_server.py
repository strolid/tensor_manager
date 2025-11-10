import base64
import pytest
import torch
import numpy as np
import tempfile
import io
from fastapi.testclient import TestClient
from tensor_manager.tensor_server import app
import importlib

from tensor_manager.tensor_client import TensorClient

load_tensor_module = importlib.import_module("tensor_manager.load_tensor")
unload_tensor_module = importlib.import_module("tensor_manager.unload_tensor")

try:
    import cupy as cp
except Exception:  # pragma: no cover - cupy may be unavailable
    cp = None

HAS_CUDA_IPC = torch.cuda.is_available() and cp is not None

client = TestClient(app)

@pytest.fixture
def sample_wav_file():
    sample_rate = 16000
    duration = 1.0
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        import soundfile as sf
        sf.write(tmp_file.name, audio_data, sample_rate)
        return tmp_file.name

@pytest.fixture
def sample_wav_bytes():
    sample_rate = 16000
    duration = 0.5
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    buffer = io.BytesIO()
    import soundfile as sf
    sf.write(buffer, audio_data, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer.getvalue()

class TestTensorServer:
    
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "GPU Tensor Server is running"}
    
    def test_cuda_info_endpoint(self):
        response = client.get("/cuda/info")
        assert response.status_code == 200
        data = response.json()
        assert "cuda_available" in data
        
        if torch.cuda.is_available():
            assert data["cuda_available"] is True
            assert "device_count" in data
            assert "devices" in data
            assert data["device_count"] > 0
        else:
            assert data["cuda_available"] is False
    
    def test_list_tensors_empty(self):
        response = client.get("/tensors")
        assert response.status_code == 200
        data = response.json()
        assert "tensors" in data
        assert "count" in data
        assert isinstance(data["tensors"], list)
    
    def test_create_tensor_success(self, sample_wav_bytes):
        files = {"wav_file": ("test.wav", sample_wav_bytes, "audio/wav")}
        data = {"cuda_device": 0}
        
        response = client.post("/tensors", files=files, data=data)
        assert response.status_code == 200
        
        result = response.json()
        assert "tensor_id" in result
        assert "shape" in result
        assert "dtype" in result
        assert "device" in result
        assert "sample_rate" in result
        if HAS_CUDA_IPC:
            assert result["device"].startswith("cuda:")
        else:
            assert result["device"] == "cpu"
        
        tensor_id = result["tensor_id"]
        
        response = client.delete(f"/tensors/{tensor_id}")
        assert response.status_code == 200
    
    def test_create_tensor_invalid_device(self, sample_wav_bytes):
        files = {"wav_file": ("test.wav", sample_wav_bytes, "audio/wav")}
        data = {"cuda_device": 999}
        
        response = client.post("/tensors", files=files, data=data)
        if HAS_CUDA_IPC:
            assert response.status_code == 400
            assert "Invalid CUDA device" in response.json()["detail"]
        else:
            # In CPU mode the request succeeds regardless of CUDA device id.
            assert response.status_code == 200
    
    def test_create_tensor_invalid_file_type(self):
        fake_file = b"not a wav file"
        files = {"wav_file": ("test.txt", fake_file, "text/plain")}
        data = {"cuda_device": 0}
        
        response = client.post("/tensors", files=files, data=data)
        assert response.status_code == 400
        assert "Only WAV files are supported" in response.json()["detail"]
    
    def test_tensor_lifecycle(self, sample_wav_bytes):
        files = {"wav_file": ("test.wav", sample_wav_bytes, "audio/wav")}
        data = {"cuda_device": 0}
        
        create_response = client.post("/tensors", files=files, data=data)
        assert create_response.status_code == 200
        tensor_id = create_response.json()["tensor_id"]
        
        info_response = client.get(f"/tensors/{tensor_id}")
        assert info_response.status_code == 200
        info_data = info_response.json()
        assert "shape" in info_data
        assert "dtype" in info_data
        assert "device" in info_data
        assert "sample_rate" in info_data
        assert "original_filename" in info_data
        assert "tensor" not in info_data
        
        handle_response = client.get(f"/tensors/{tensor_id}/handle")
        assert handle_response.status_code == 200
        handle_result = handle_response.json()
        if HAS_CUDA_IPC:
            assert "ipc_handle" in handle_result
        else:
            assert "data_b64" in handle_result
        assert handle_result["tensor_id"] == tensor_id
        assert "element_size" in handle_result
        assert "numel" in handle_result
        if HAS_CUDA_IPC:
            assert "data_ptr" in handle_result
        else:
            decoded = base64.b64decode(handle_result["data_b64"])
            assert len(decoded) == handle_result["nbytes"]
        
        list_response = client.get("/tensors")
        assert list_response.status_code == 200
        tensors = list_response.json()["tensors"]
        tensor_ids = [t["tensor_id"] for t in tensors]
        assert tensor_id in tensor_ids
        
        delete_response = client.delete(f"/tensors/{tensor_id}")
        assert delete_response.status_code == 200
        assert "deleted successfully" in delete_response.json()["message"]
        
        get_deleted_response = client.get(f"/tensors/{tensor_id}")
        assert get_deleted_response.status_code == 404
        assert "Tensor not found" in get_deleted_response.json()["detail"]
    
    def test_get_nonexistent_tensor(self):
        fake_id = "nonexistent-tensor-id"
        response = client.get(f"/tensors/{fake_id}")
        assert response.status_code == 404
        assert "Tensor not found" in response.json()["detail"]
    
    def test_delete_nonexistent_tensor(self):
        fake_id = "nonexistent-tensor-id"
        response = client.delete(f"/tensors/{fake_id}")
        assert response.status_code == 404
        assert "Tensor not found" in response.json()["detail"]
    
    def test_get_handle_nonexistent_tensor(self):
        fake_id = "nonexistent-tensor-id"
        response = client.get(f"/tensors/{fake_id}/handle")
        assert response.status_code == 404
        assert "Tensor not found" in response.json()["detail"]
    
    @pytest.mark.skipif(not HAS_CUDA_IPC, reason="CUDA not available")
    def test_multiple_tensors_different_devices(self, sample_wav_bytes):
        if torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 CUDA devices for this test")
        
        tensor_ids = []
        
        for device_id in [0, 1]:
            files = {"wav_file": (f"test_{device_id}.wav", sample_wav_bytes, "audio/wav")}
            data = {"cuda_device": device_id}
            
            response = client.post("/tensors", files=files, data=data)
            assert response.status_code == 200
            result = response.json()
            assert f"cuda:{device_id}" in result["device"]
            tensor_ids.append(result["tensor_id"])
        
        list_response = client.get("/tensors")
        assert list_response.status_code == 200
        assert list_response.json()["count"] >= 2
        
        for tensor_id in tensor_ids:
            delete_response = client.delete(f"/tensors/{tensor_id}")
            assert delete_response.status_code == 200

class TestTensorServerIntegration:
    
    @pytest.mark.skipif(not HAS_CUDA_IPC, reason="CUDA not available")
    def test_wav_file_loading_and_tensor_properties(self, sample_wav_bytes):
        files = {"wav_file": ("sine_wave.wav", sample_wav_bytes, "audio/wav")}
        data = {"cuda_device": 0}
        
        response = client.post("/tensors", files=files, data=data)
        assert response.status_code == 200
        result = response.json()
        
        tensor_id = result["tensor_id"]
        assert len(result["shape"]) == 1
        assert result["sample_rate"] > 0
        assert result["dtype"] == "torch.float32"
        
        info_response = client.get(f"/tensors/{tensor_id}")
        assert info_response.status_code == 200
        info = info_response.json()
        assert info["original_filename"] == "sine_wave.wav"
        
        handle_response = client.get(f"/tensors/{tensor_id}/handle")
        assert handle_response.status_code == 200
        handle_result = handle_response.json()
        
        assert handle_result["ipc_handle"] is not None
        assert handle_result["data_ptr"] > 0
        assert handle_result["element_size"] > 0
        assert handle_result["numel"] > 0
        assert len(handle_result["shape"]) == 1
        
        client.delete(f"/tensors/{tensor_id}")


class DummyVcon:
    def __init__(self, uuid: str):
        self.uuid = uuid
        self.dialog = []

    def to_dict(self):
        return {"uuid": self.uuid, "dialog": self.dialog}


class DummyRedis:
    def __init__(self, vcon: DummyVcon):
        self._vcon = vcon
        self.stored = False

    def get_vcon(self, uuid: str):
        return self._vcon if self._vcon.uuid == uuid else None

    def store_vcon(self, vcon):
        self.stored = True


def test_load_tensor_no_server(monkeypatch):
    monkeypatch.setattr(TensorClient, "is_server_available", lambda self: False)
    result = load_tensor_module.run("vcon-123", "load_tensor")
    assert result == "vcon-123"


def test_unload_tensor_no_server(monkeypatch):
    monkeypatch.setattr(TensorClient, "is_server_available", lambda self: False)
    result = unload_tensor_module.run("vcon-123", "unload_tensor")
    assert result == "vcon-123"


def test_load_tensor_invokes_client(monkeypatch):
    vcon = DummyVcon("vcon-123")
    dummy_redis = DummyRedis(vcon)

    monkeypatch.setattr(TensorClient, "is_server_available", lambda self: True)
    called = {}

    def fake_ensure(self, vcon_obj, device="cuda:0"):
        called["device"] = device
        called["uuid"] = vcon_obj.uuid
        return torch.zeros(1)

    monkeypatch.setattr(TensorClient, "ensure_remote_tensor", fake_ensure, raising=False)
    monkeypatch.setattr(TensorClient, "__init__", lambda self: None)
    monkeypatch.setattr(load_tensor_module, "VconRedis", lambda: dummy_redis)
    result = load_tensor_module.run("vcon-123", "load_tensor", {"device": "cuda:1"})
    assert result == "vcon-123"
    assert called == {"device": "cuda:1", "uuid": "vcon-123"}
    assert dummy_redis.stored is True


def test_unload_tensor_invokes_client(monkeypatch):
    vcon = DummyVcon("vcon-456")
    dummy_redis = DummyRedis(vcon)

    monkeypatch.setattr(TensorClient, "is_server_available", lambda self: True)
    called = {}

    def fake_unload(self, vcon_obj):
        called["uuid"] = vcon_obj.uuid
        return 1

    monkeypatch.setattr(TensorClient, "unload_remote_tensor", fake_unload, raising=False)
    monkeypatch.setattr(TensorClient, "__init__", lambda self: None)
    monkeypatch.setattr(unload_tensor_module, "VconRedis", lambda: dummy_redis)
    result = unload_tensor_module.run("vcon-456", "unload_tensor")
    assert result == "vcon-456"
    assert called == {"uuid": "vcon-456"}
    assert dummy_redis.stored is True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
