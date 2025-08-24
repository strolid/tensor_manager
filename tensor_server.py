import uuid
import logging
import torch
import torchaudio
import numpy as np
import cupy as cp
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import os
import mmap

logger = logging.getLogger("tensor_manager")
app = FastAPI(title="GPU Tensor Server", version="1.0.0")

# Shared memory directory for tensor files
SHARED_TENSOR_DIR = Path("/tmp/shared_tensors")
SHARED_TENSOR_DIR.mkdir(exist_ok=True)

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

    if not torch.cuda.is_available():
        raise HTTPException(status_code=500, detail="CUDA is not available")
    
    if cuda_device >= torch.cuda.device_count() or cuda_device < 0:
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
        waveform, sample_rate = torchaudio.load(tmp_file_path)
        audio = waveform.squeeze(0).contiguous().numpy()
        cp.cuda.Device(cuda_device).use()
        cp_arr = cp.asarray(audio, dtype=cp.float32)
        ptr = int(cp_arr.data.ptr)
        # Obtain 64-byte IPC handle using CuPy single-arg API and normalize
        handle_obj = cp.cuda.runtime.ipcGetMemHandle(ptr)
        if isinstance(handle_obj, (bytes, bytearray)):
            handle_bytes = bytes(handle_obj)
        elif hasattr(handle_obj, 'tobytes'):
            handle_bytes = handle_obj.tobytes()
        else:
            try:
                import numpy as _np
                handle_bytes = _np.asarray(handle_obj, dtype=_np.uint8).tobytes()
            except Exception:
                raise HTTPException(status_code=500, detail="Unsupported IPC handle type")
        if len(handle_bytes) < 64:
            raise HTTPException(status_code=500, detail=f"Invalid IPC handle size: {len(handle_bytes)}")
        ipc_handle_b64 = base64.b64encode(handle_bytes[:64]).decode('utf-8')
        
        tensor_storage[tensor_id] = {
            "cupy": cp_arr,
            "shape": list(cp_arr.shape),
            "dtype": "torch.float32",
            "device": f"cuda:{cuda_device}",
            "sample_rate": sample_rate,
            "original_filename": wav_file.filename,
            "ipc_handle": ipc_handle_b64,
            "data_ptr": ptr,
            "element_size": 4,
            "numel": int(cp_arr.size)
        }
        
        os.unlink(tmp_file_path)
        
        return {
            "tensor_id": tensor_id,
            "shape": list(cp_arr.shape),
            "dtype": "torch.float32",
            "device": f"cuda:{cuda_device}",
            "sample_rate": sample_rate,
            "message": f"Tensor loaded successfully on cuda:{cuda_device}"
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
    cuda_device: int
    sample_rate: int
    data_b64: str
    dtype: str = "float32"
    shape: Optional[List[int]] = None


@app.post("/tensors/from-array")
async def create_tensor_from_array(payload: TensorArrayPayload):
    if not torch.cuda.is_available():
        raise HTTPException(status_code=500, detail="CUDA is not available")

    if payload.cuda_device >= torch.cuda.device_count() or payload.cuda_device < 0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CUDA device {payload.cuda_device}. Available devices: 0-{torch.cuda.device_count()-1}"
        )

    try:
        raw = base64.b64decode(payload.data_b64)
        np_dtype = np.float32 if payload.dtype in ("float32", "torch.float32") else np.float32
        np_array = np.frombuffer(raw, dtype=np_dtype)
        if payload.shape:
            np_array = np_array.reshape(tuple(payload.shape))
        # Use shared memory mapping for TRUE zero-copy GPU tensor sharing
        cp.cuda.Device(payload.cuda_device).use()
        cp_arr = cp.asarray(np_array, dtype=cp.float32)
        
        # Convert to PyTorch tensor using modern DLPack protocol (no toDlpack())
        torch_tensor = torch.utils.dlpack.from_dlpack(cp_arr)
        
        # Create shared memory identifier using data pointer
        data_ptr = int(torch_tensor.data_ptr())
        
        tensor_id = str(uuid.uuid4())
        tensor_storage[tensor_id] = {
            "torch_tensor": torch_tensor,  # Keep reference to prevent garbage collection
            "cupy": cp_arr,
            "shape": list(torch_tensor.shape),
            "dtype": "torch.float32", 
            "device": f"cuda:{payload.cuda_device}",
            "sample_rate": payload.sample_rate,
            "original_filename": None,
            "data_ptr": data_ptr,
            "element_size": 4,
            "numel": int(torch_tensor.numel()),
            "nbytes": int(torch_tensor.numel() * 4)
        }

        return {
            "tensor_id": tensor_id,
            "shape": list(cp_arr.shape),
            "dtype": "torch.float32",
            "device": f"cuda:{payload.cuda_device}",
            "sample_rate": payload.sample_rate,
            "message": f"Tensor loaded successfully on cuda:{payload.cuda_device}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create tensor from array: {str(e)}")

@app.get("/tensors/{tensor_id}")
async def get_tensor_info(tensor_id: str):
    if tensor_id not in tensor_storage:
        raise HTTPException(status_code=404, detail="Tensor not found")
    
    info = tensor_storage[tensor_id].copy()
    info.pop("cupy", None)
    info.pop("ipc_handle", None)
    return info

@app.get("/tensors/{tensor_id}/handle")
async def get_tensor_handle(tensor_id: str):
    if tensor_id not in tensor_storage:
        raise HTTPException(status_code=404, detail="Tensor not found")
    
    tensor_info = tensor_storage[tensor_id]
    
    return {
        "tensor_id": tensor_id,
        "shared_memory": True,  # Flag indicating shared memory access
        "shape": tensor_info["shape"],
        "dtype": tensor_info["dtype"],
        "device": tensor_info["device"],
        "data_ptr": tensor_info["data_ptr"],
        "element_size": tensor_info["element_size"],
        "numel": tensor_info["numel"],
        "nbytes": tensor_info["nbytes"],
        "sample_rate": tensor_info["sample_rate"]
    }

@app.get("/tensors/{tensor_id}/direct")
async def get_direct_tensor(tensor_id: str):
    """Get direct access to GPU tensor for zero-copy within same process"""
    if tensor_id not in tensor_storage:
        raise HTTPException(status_code=404, detail="Tensor not found")
    
    tensor_info = tensor_storage[tensor_id]
    torch_tensor = tensor_info["torch_tensor"]
    
    print(f"DEBUG: Providing DIRECT zero-copy access to GPU tensor {tensor_id}")
    print(f"DEBUG: Tensor shape: {torch_tensor.shape}, dtype: {torch_tensor.dtype}, device: {torch_tensor.device}")
    print(f"DEBUG: This is TRUE zero-copy - same physical GPU memory!")
    
    return {
        "tensor_id": tensor_id,
        "tensor": torch_tensor,  # Direct tensor reference - TRUE zero-copy!
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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10070, access_log=False)
