import uuid
import torch
import torchaudio
import numpy as np
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tempfile
import os

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
    if not wav_file.filename.endswith(('.wav', '.WAV')):
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
        
        tensor = waveform.squeeze(0).float()
        
        device = torch.device(f"cuda:{cuda_device}")
        tensor = tensor.to(device)
        
        tensor = tensor.contiguous()
        tensor.share_memory_()
        
        ipc_handle_result = tensor.untyped_storage()._share_cuda_()
        if isinstance(ipc_handle_result, tuple):
            ipc_handle = ipc_handle_result[1]
        else:
            ipc_handle = ipc_handle_result
        ipc_handle_b64 = base64.b64encode(ipc_handle).decode('utf-8')
        
        tensor_storage[tensor_id] = {
            "tensor": tensor,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "sample_rate": sample_rate,
            "original_filename": wav_file.filename,
            "ipc_handle": ipc_handle_b64,
            "data_ptr": tensor.data_ptr(),
            "element_size": tensor.element_size(),
            "numel": tensor.numel()
        }
        
        os.unlink(tmp_file_path)
        
        return {
            "tensor_id": tensor_id,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "sample_rate": sample_rate,
            "message": f"Tensor loaded successfully on {device}"
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

@app.get("/tensors/{tensor_id}")
async def get_tensor_info(tensor_id: str):
    if tensor_id not in tensor_storage:
        raise HTTPException(status_code=404, detail="Tensor not found")
    
    info = tensor_storage[tensor_id].copy()
    info.pop("tensor")
    info.pop("ipc_handle")
    return info

@app.get("/tensors/{tensor_id}/handle")
async def get_tensor_handle(tensor_id: str):
    if tensor_id not in tensor_storage:
        raise HTTPException(status_code=404, detail="Tensor not found")
    
    tensor_info = tensor_storage[tensor_id]
    
    return {
        "tensor_id": tensor_id,
        "ipc_handle": tensor_info["ipc_handle"],
        "shape": tensor_info["shape"],
        "dtype": tensor_info["dtype"],
        "device": tensor_info["device"],
        "data_ptr": tensor_info["data_ptr"],
        "element_size": tensor_info["element_size"],
        "numel": tensor_info["numel"],
        "sample_rate": tensor_info["sample_rate"]
    }

@app.delete("/tensors/{tensor_id}")
async def delete_tensor(tensor_id: str):
    if tensor_id not in tensor_storage:
        raise HTTPException(status_code=404, detail="Tensor not found")
    
    tensor_info = tensor_storage.pop(tensor_id)
    
    try:
        tensor = tensor_info["tensor"]
        device = tensor.device
        del tensor_info["tensor"]
        del tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
    except Exception as e:
        pass
    
    return {"message": f"Tensor {tensor_id} deleted successfully"}

@app.get("/tensors")
async def list_tensors():
    tensor_list = []
    for tensor_id, info in tensor_storage.items():
        tensor_info = info.copy()
        tensor_info.pop("tensor")
        tensor_info.pop("ipc_handle")
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
            tensor = tensor_info["tensor"]
            device = tensor.device
            del tensor_info["tensor"]
            del tensor
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
        except Exception as e:
            print(f"Error cleaning up tensor {tensor_id}: {e}")
    print("Cleanup completed.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
