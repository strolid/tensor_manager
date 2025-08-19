#!/usr/bin/env python3

import torch
import base64
import requests
from typing import Optional

class TensorClient:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
    
    def get_tensor_handle(self, tensor_id: str) -> dict:
        response = requests.get(f"{self.server_url}/tensors/{tensor_id}/handle")
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get tensor handle: {response.text}")
        return response.json()
    
    def access_shared_tensor(self, tensor_id: str) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        handle_data = self.get_tensor_handle(tensor_id)
        
        ipc_handle_bytes = base64.b64decode(handle_data['ipc_handle'])
        
        shape = tuple(handle_data['shape'])
        dtype_str = handle_data['dtype']
        device_str = handle_data['device']
        
        dtype_map = {
            'torch.float32': torch.float32,
            'torch.float64': torch.float64,
            'torch.int32': torch.int32,
            'torch.int64': torch.int64,
            'torch.float16': torch.float16
        }
        dtype = dtype_map.get(dtype_str, torch.float32)
        
        try:
            storage = torch.cuda.UntypedStorage._new_shared_cuda(ipc_handle_bytes)
            
            typed_storage = storage._typed_storage(dtype)
            
            tensor = torch.tensor([], dtype=dtype, device=device_str)
            tensor.set_(typed_storage, 0, shape)
            
            return tensor
        
        except Exception as e:
            raise RuntimeError(f"Failed to access shared tensor: {e}")
    
    def upload_wav_file(self, wav_file_path: str, cuda_device: int) -> str:
        with open(wav_file_path, 'rb') as f:
            files = {'wav_file': (wav_file_path, f, 'audio/wav')}
            data = {'cuda_device': cuda_device}
            response = requests.post(f"{self.server_url}/tensors", files=files, data=data)
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to upload WAV file: {response.text}")
        
        return response.json()['tensor_id']
    
    def delete_tensor(self, tensor_id: str) -> bool:
        response = requests.delete(f"{self.server_url}/tensors/{tensor_id}")
        return response.status_code == 200
    
    def list_tensors(self) -> list:
        response = requests.get(f"{self.server_url}/tensors")
        if response.status_code != 200:
            raise RuntimeError(f"Failed to list tensors: {response.text}")
        return response.json()['tensors']

def example_usage():
    print("=== Tensor Client Example ===\n")
    
    client = TensorClient()
    
    import tempfile
    import soundfile as sf
    import numpy as np
    
    sample_rate = 16000
    duration = 1.0
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
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
        import os
        try:
            os.unlink(wav_path)
        except:
            pass
    
    print("\n=== Example completed ===")

if __name__ == "__main__":
    example_usage()
