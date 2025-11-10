import argparse
import base64
import logging
import os
import sys
import tempfile
import traceback
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchaudio
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

try:
    import cupy as cp
except Exception:  # pragma: no cover - cupy may be unavailable on CPU-only systems
    cp = None

logger = logging.getLogger("tensor_manager")
app = FastAPI(title="GPU Tensor Server", version="1.0.0")

tensor_storage: Dict[str, Dict[str, Any]] = {}

@app.get("/")
async def root():
    return {"message": "GPU Tensor Server is running"}

@app.post("/tensors")
async def create_tensor(
    cuda_device: int = Form(...),
    wav_file: UploadFile = File(...)
):
    if not wav_file.filename.endswith((".wav", ".WAV")):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    using_cuda = torch.cuda.is_available() and cp is not None
    if using_cuda and (cuda_device >= torch.cuda.device_count() or cuda_device < 0):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CUDA device {cuda_device}. Available devices: 0-{torch.cuda.device_count()-1}"
        )

    tensor_id = str(uuid.uuid4())
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await wav_file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        try:
            waveform, sample_rate = torchaudio.load(tmp_file_path)
            audio = waveform.squeeze(0).contiguous().numpy()
        except (ImportError, RuntimeError) as err:
            if "torchcodec" not in str(err).lower():
                raise
            audio, sample_rate = sf.read(tmp_file_path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = np.asarray(audio, dtype=np.float32)

        if using_cuda:
            cp.cuda.Device(cuda_device).use()
            cp_arr = cp.asarray(audio, dtype=cp.float32)
            ptr = int(cp_arr.data.ptr)
            handle_obj = cp.cuda.runtime.ipcGetMemHandle(ptr)
            if isinstance(handle_obj, (bytes, bytearray)):
                handle_bytes = bytes(handle_obj)
            elif hasattr(handle_obj, "tobytes"):
                handle_bytes = handle_obj.tobytes()
            else:
                try:
                    handle_bytes = np.asarray(handle_obj, dtype=np.uint8).tobytes()
                except Exception:
                    raise HTTPException(status_code=500, detail="Unsupported IPC handle type")
            if len(handle_bytes) < 64:
                raise HTTPException(status_code=500, detail=f"Invalid IPC handle size: {len(handle_bytes)}")
            ipc_handle_b64 = base64.b64encode(handle_bytes[:64]).decode("utf-8")
            torch_tensor = torch.utils.dlpack.from_dlpack(cp_arr)
            element_size = torch_tensor.element_size()
            numel = int(torch_tensor.numel())
            tensor_storage[tensor_id] = {
                "mode": "cuda",
                "torch_tensor": torch_tensor,
                "cupy": cp_arr,
                "shape": list(cp_arr.shape),
                "dtype": str(torch_tensor.dtype),
                "device": f"cuda:{cuda_device}",
                "sample_rate": sample_rate,
                "original_filename": wav_file.filename,
                "ipc_handle": ipc_handle_b64,
                "data_ptr": ptr,
                "element_size": element_size,
                "numel": numel,
                "nbytes": element_size * numel,
            }
            device_label = f"cuda:{cuda_device}"
        else:
            torch_tensor = torch.from_numpy(audio).to(torch.float32)
            element_size = torch_tensor.element_size()
            numel = int(torch_tensor.numel())
            tensor_storage[tensor_id] = {
                "mode": "cpu",
                "torch_tensor": torch_tensor,
                "shape": list(torch_tensor.shape),
                "dtype": str(torch_tensor.dtype),
                "device": "cpu",
                "sample_rate": sample_rate,
                "original_filename": wav_file.filename,
                "element_size": element_size,
                "numel": numel,
                "nbytes": element_size * numel,
            }
            device_label = "cpu"
        
        os.unlink(tmp_file_path)
        
        return {
            "tensor_id": tensor_id,
            "shape": list(tensor_storage[tensor_id]["shape"]),
            "dtype": tensor_storage[tensor_id]["dtype"],
            "device": device_label,
            "sample_rate": sample_rate,
            "message": f"Tensor loaded successfully on {device_label}"
        }
        
    except Exception as e:
        import traceback
        print(f"Error creating tensor: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to load tensor: {str(e)}")


class TensorArrayPayload(BaseModel):
    cuda_device: Optional[int] = 0
    sample_rate: int
    data_b64: str
    dtype: str = "float32"
    shape: Optional[List[int]] = None


@app.post("/tensors/from-array")
async def create_tensor_from_array(payload: TensorArrayPayload):
    using_cuda = torch.cuda.is_available() and cp is not None
    cuda_device = payload.cuda_device or 0
    if using_cuda and (cuda_device >= torch.cuda.device_count() or cuda_device < 0):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CUDA device {cuda_device}. Available devices: 0-{torch.cuda.device_count()-1}"
        )

    try:
        raw = base64.b64decode(payload.data_b64)
        np_dtype = np.float32 if payload.dtype in ("float32", "torch.float32") else np.float32
        np_array = np.frombuffer(raw, dtype=np_dtype)
        if payload.shape:
            np_array = np_array.reshape(tuple(payload.shape))

        tensor_id = str(uuid.uuid4())
        if using_cuda:
            cp.cuda.Device(cuda_device).use()
            cp_arr = cp.asarray(np_array, dtype=cp.float32)
            torch_tensor = torch.utils.dlpack.from_dlpack(cp_arr)
            data_ptr = int(torch_tensor.data_ptr())
            handle_obj = cp.cuda.runtime.ipcGetMemHandle(data_ptr)
            if isinstance(handle_obj, (bytes, bytearray)):
                handle_bytes = bytes(handle_obj)
            elif hasattr(handle_obj, "tobytes"):
                handle_bytes = handle_obj.tobytes()
            else:
                handle_bytes = np.asarray(handle_obj, dtype=np.uint8).tobytes()
            if len(handle_bytes) < 64:
                raise HTTPException(status_code=500, detail=f"Invalid IPC handle size: {len(handle_bytes)}")
            ipc_handle_b64 = base64.b64encode(handle_bytes[:64]).decode("utf-8")
            element_size = torch_tensor.element_size()
            numel = int(torch_tensor.numel())
            tensor_storage[tensor_id] = {
                "mode": "cuda",
                "torch_tensor": torch_tensor,
                "cupy": cp_arr,
                "shape": list(torch_tensor.shape),
                "dtype": str(torch_tensor.dtype),
                "device": f"cuda:{cuda_device}",
                "sample_rate": payload.sample_rate,
                "original_filename": None,
                "ipc_handle": ipc_handle_b64,
                "data_ptr": data_ptr,
                "element_size": element_size,
                "numel": numel,
                "nbytes": element_size * numel,
            }
            device_label = f"cuda:{cuda_device}"
        else:
            torch_tensor = torch.from_numpy(np_array.copy()).to(torch.float32)
            element_size = torch_tensor.element_size()
            numel = int(torch_tensor.numel())
            tensor_storage[tensor_id] = {
                "mode": "cpu",
                "torch_tensor": torch_tensor,
                "shape": list(torch_tensor.shape),
                "dtype": str(torch_tensor.dtype),
                "device": "cpu",
                "sample_rate": payload.sample_rate,
                "original_filename": None,
                "element_size": element_size,
                "numel": numel,
                "nbytes": element_size * numel,
            }
            device_label = "cpu"

        return {
            "tensor_id": tensor_id,
            "shape": tensor_storage[tensor_id]["shape"],
            "dtype": tensor_storage[tensor_id]["dtype"],
            "device": device_label,
            "sample_rate": payload.sample_rate,
            "message": f"Tensor loaded successfully on {device_label}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create tensor from array: {str(e)}")

@app.get("/tensors/{tensor_id}")
async def get_tensor_info(tensor_id: str):
    if tensor_id not in tensor_storage:
        raise HTTPException(status_code=404, detail="Tensor not found")
    
    info = tensor_storage[tensor_id].copy()
    info.pop("cupy", None)
    info.pop("torch_tensor", None)
    info.pop("ipc_handle", None)
    return info

@app.get("/tensors/{tensor_id}/handle")
async def get_tensor_handle(tensor_id: str):
    if tensor_id not in tensor_storage:
        raise HTTPException(status_code=404, detail="Tensor not found")
    
    tensor_info = tensor_storage[tensor_id]
    if tensor_info.get("mode") == "cuda":
        return {
            "tensor_id": tensor_id,
            "shared_memory": True,
            "shape": tensor_info["shape"],
            "dtype": tensor_info["dtype"],
            "device": tensor_info["device"],
            "ipc_handle": tensor_info["ipc_handle"],
            "data_ptr": tensor_info["data_ptr"],
            "element_size": tensor_info["element_size"],
            "numel": tensor_info["numel"],
            "nbytes": tensor_info["nbytes"],
            "sample_rate": tensor_info["sample_rate"],
        }

    cpu_tensor = tensor_info["torch_tensor"]
    data_bytes = cpu_tensor.numpy().tobytes()
    data_b64 = base64.b64encode(data_bytes).decode("utf-8")
    return {
        "tensor_id": tensor_id,
        "shared_memory": False,
        "shape": tensor_info["shape"],
        "dtype": tensor_info["dtype"],
        "device": tensor_info["device"],
        "element_size": cpu_tensor.element_size(),
        "numel": cpu_tensor.numel(),
        "nbytes": cpu_tensor.element_size() * cpu_tensor.numel(),
        "sample_rate": tensor_info["sample_rate"],
        "data_b64": data_b64,
    }

@app.get("/tensors/{tensor_id}/direct")
async def get_direct_tensor(tensor_id: str):
    """Get direct access to GPU tensor for zero-copy within same process"""
    if tensor_id not in tensor_storage:
        raise HTTPException(status_code=404, detail="Tensor not found")
    
    tensor_info = tensor_storage[tensor_id]
    torch_tensor = tensor_info["torch_tensor"]
    
    print(f"DEBUG: Providing DIRECT access to tensor {tensor_id}")
    print(f"DEBUG: Tensor shape: {torch_tensor.shape}, dtype: {torch_tensor.dtype}, device: {torch_tensor.device}")
    
    return {
        "tensor_id": tensor_id,
        "tensor": torch_tensor,
        "shape": list(torch_tensor.shape),
        "dtype": str(torch_tensor.dtype),
        "device": str(torch_tensor.device),
        "data_ptr": int(torch_tensor.data_ptr()),
        "sample_rate": tensor_info["sample_rate"]
    }

@app.delete("/tensors/{tensor_id}")
async def delete_tensor(tensor_id: str):
    if tensor_id not in tensor_storage:
        raise HTTPException(status_code=404, detail="Tensor not found")
    
    tensor_info = tensor_storage.pop(tensor_id)
    
    try:
        cp_arr = tensor_info.get("cupy")
        if cp_arr is not None:
            try:
                cp_arr.data.mem.free()
            except Exception:
                pass
            tensor_info.pop("cupy", None)
    except Exception as e:
        pass
    
    # IPC handle is automatically cleaned up when CuPy array is freed
    
    return {"message": f"Tensor {tensor_id} deleted successfully"}

@app.get("/tensors")
async def list_tensors():
    tensor_list = []
    for tensor_id, info in tensor_storage.items():
        tensor_info = info.copy()
        tensor_info.pop("cupy", None)
        tensor_info.pop("torch_tensor", None)
        tensor_info.pop("ipc_handle", None)
        tensor_info["tensor_id"] = tensor_id
        tensor_list.append(tensor_info)
    
    return {
        "tensors": tensor_list,
        "count": len(tensor_list)
    }

@app.get("/cuda/info")
async def cuda_info():
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    devices = []
    for i in range(torch.cuda.device_count()):
        device_props = torch.cuda.get_device_properties(i)
        devices.append({
            "device_id": i,
            "name": device_props.name,
            "total_memory": device_props.total_memory,
            "major": device_props.major,
            "minor": device_props.minor
        })
    
    return {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "devices": devices
    }

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down server, cleaning up tensors...")
    global tensor_storage
    for tensor_id in list(tensor_storage.keys()):
        try:
            tensor_info = tensor_storage.pop(tensor_id)
            cp_arr = tensor_info.get("cupy")
            if cp_arr is not None:
                try:
                    cp_arr.data.mem.free()
                except Exception:
                    pass
        except Exception as e:
            print(f"Error cleaning up tensor {tensor_id}: {e}")
    print("Cleanup completed.")



def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tensor manager FastAPI server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8003, help="Port to listen on (default: 8003)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main(sys.argv[1:])
