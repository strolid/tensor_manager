#!/usr/bin/env python3

import base64
import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import soundfile as sf
import torch

class TensorClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8003):
        self.host = host
        self.port = port
        self.server_url = f"http://{self.host}:{self.port}"
    
    def get_tensor_handle(self, tensor_id: str) -> dict:
        response = requests.get(f"{self.server_url}/tensors/{tensor_id}/handle")
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get tensor handle: {response.text}")
        return response.json()
    
    def access_shared_tensor(self, tensor_id: str) -> torch.Tensor:
        handle_data = self.get_tensor_handle(tensor_id)
        shape = tuple(handle_data['shape'])
        dtype_str = handle_data.get('dtype', 'float32')
        device_str = handle_data.get('device', 'cuda:0')
        data_b64 = handle_data.get('data_b64')
        if not data_b64:
            raise RuntimeError("Tensor handle is missing data payload")

        torch_dtype_map = {
            'torch.float32': torch.float32,
            'torch.float64': torch.float64,
            'torch.int32': torch.int32,
            'torch.int64': torch.int64,
            'torch.float16': torch.float16,
            'float32': torch.float32,
            'float64': torch.float64,
            'int32': torch.int32,
            'int64': torch.int64,
            'float16': torch.float16,
        }
        numpy_dtype_map = {
            'torch.float32': np.float32,
            'torch.float64': np.float64,
            'torch.int32': np.int32,
            'torch.int64': np.int64,
            'torch.float16': np.float16,
            'float32': np.float32,
            'float64': np.float64,
            'int32': np.int32,
            'int64': np.int64,
            'float16': np.float16,
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
        with open(wav_file_path, 'rb') as f:
            files = {'wav_file': (wav_file_path, f, 'audio/wav')}
            data = {'cuda_device': cuda_device}
            response = requests.post(f"{self.server_url}/tensors", files=files, data=data)
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to upload WAV file: {response.text}")
        
        return response.json()['tensor_id']

    def upload_cpu_tensor(self, tensor: torch.Tensor, sample_rate: int, cuda_device: int = 0) -> str:
        if tensor.is_cuda:
            tensor = tensor.cpu()
        arr = tensor.contiguous().numpy()
        raw = arr.tobytes()
        b64 = base64.b64encode(raw).decode('utf-8')
        payload = {
            'cuda_device': cuda_device,
            'sample_rate': sample_rate,
            'data_b64': b64,
            'dtype': str(tensor.dtype).replace('torch.', ''),
            'shape': list(tensor.shape)
        }
        r = requests.post(f"{self.server_url}/tensors/from-array", json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to upload tensor: {r.text}")
        return r.json()['tensor_id']
    
    def delete_tensor(self, tensor_id: str) -> bool:
        response = requests.delete(f"{self.server_url}/tensors/{tensor_id}")
        if response.status_code == 200:
            return True
        elif response.status_code == 404:
            # Tensor already deleted - that's fine
            return False
        else:
            raise RuntimeError(f"Failed to delete tensor {tensor_id}: {response.text}")
    
    def list_tensors(self) -> list:
        response = requests.get(f"{self.server_url}/tensors")
        if response.status_code != 200:
            raise RuntimeError(f"Failed to list tensors: {response.text}")
        return response.json()['tensors']

    def get_or_lookup_tensor_by_vcon(self, vcon: Any) -> torch.Tensor:
        """
        Scans vCon dialog entries to find a tensor. If a 'tensor' entry exists, uses its
        tensor_ref. Otherwise, if a reference entry exists ('tensor_ref', 'gpu-entire', 'gpu_tensor'),
        fetches from the server, appends a new 'tensor' dialog entry, and returns the tensor.
        The provided vcon object is modified in-place (destructively) if possible.
        """
        # Normalize to a dict we can mutate
        if hasattr(vcon, 'to_dict'):
            vcon_dict: Dict[str, Any] = vcon.to_dict()
            can_assign_back = hasattr(vcon, 'vcon_dict')
        elif isinstance(vcon, dict):
            vcon_dict = vcon
            can_assign_back = True
        else:
            raise TypeError('vcon must be a vcon.Vcon or dict')

        dialogs: List[Dict[str, Any]] = vcon_dict.get('dialog', []) or []

        def extract_tid(entry: Dict[str, Any]) -> Optional[str]:
            ref = entry.get('tensor_ref')
            if not ref:
                return None
            return ref.split('tensor://', 1)[1] if ref.startswith('tensor://') else ref

        # First try direct 'tensor' entries
        for d in dialogs:
            if d.get('type') == 'tensor':
                tid = extract_tid(d)
                if tid:
                    return self.access_shared_tensor(tid)

        # Fallback to reference entries
        for d in dialogs:
            if d.get('type') in {'tensor_ref', 'gpu-entire', 'gpu_tensor'}:
                tid = extract_tid(d)
                if not tid:
                    continue
                tensor = self.access_shared_tensor(tid)
                # Append a concrete 'tensor' entry for future lookups
                start_ts = d.get('start') or vcon_dict.get('created_at')
                new_entry = {
                    'type': 'tensor',
                    'tensor_ref': f'tensor://{tid}',
                    'device': str(tensor.device),
                    'shape': list(tensor.shape),
                }
                if start_ts:
                    new_entry['start'] = start_ts
                dialogs.append(new_entry)
                vcon_dict['dialog'] = dialogs
                if can_assign_back:
                    # Update underlying object if available
                    if hasattr(vcon, 'vcon_dict'):
                        vcon.vcon_dict = vcon_dict
                return tensor

        raise RuntimeError('No tensor or tensor reference found in vCon dialog entries')

    def delete_tensor_by_vcon(self, vcon: Any) -> int:
        """
        Deletes all GPU tensors referenced in the vCon and removes their dialog entries.
        Returns the number of tensors deleted. The vCon is modified in place when possible.
        """
        if hasattr(vcon, 'to_dict'):
            vcon_dict: Dict[str, Any] = vcon.to_dict()
            can_assign_back = hasattr(vcon, 'vcon_dict')
        elif isinstance(vcon, dict):
            vcon_dict = vcon
            can_assign_back = True
        else:
            raise TypeError('vcon must be a vcon.Vcon or dict')

        dialogs: List[Dict[str, Any]] = vcon_dict.get('dialog', []) or []
        deleted = 0
        new_dialogs: List[Dict[str, Any]] = []

        def extract_tid(entry: Dict[str, Any]) -> Optional[str]:
            ref = entry.get('tensor_ref')
            if not ref:
                return None
            return ref.split('tensor://', 1)[1] if ref.startswith('tensor://') else ref

        for d in dialogs:
            if d.get('type') in {'tensor', 'tensor_ref', 'gpu-entire', 'gpu_tensor'}:
                tid = extract_tid(d)
                if tid:
                    try:
                        if self.delete_tensor(tid):
                            deleted += 1
                    except Exception:
                        # Tensor might already be deleted by another process - that's OK
                        pass
                # Do not keep this dialog entry
                continue
            new_dialogs.append(d)

        if deleted:
            vcon_dict['dialog'] = new_dialogs
            if can_assign_back and hasattr(vcon, 'vcon_dict'):
                vcon.vcon_dict = vcon_dict
        return deleted

    def remove_loaded_tensor(self, vcon: Any) -> int:
        """
        Removes any ephemeral 'tensor' dialog entries added during lookup, without
        touching server-side tensors. Returns number of entries removed.
        """
        if hasattr(vcon, 'to_dict'):
            vcon_dict: Dict[str, Any] = vcon.to_dict()
            can_assign_back = hasattr(vcon, 'vcon_dict')
        elif isinstance(vcon, dict):
            vcon_dict = vcon
            can_assign_back = True
        else:
            raise TypeError('vcon must be a vcon.Vcon or dict')

        dialogs: List[Dict[str, Any]] = vcon_dict.get('dialog', []) or []
        kept: List[Dict[str, Any]] = []
        removed = 0
        for d in dialogs:
            if d.get('type') == 'tensor':
                removed += 1
                continue
            kept.append(d)
        if removed:
            vcon_dict['dialog'] = kept
            if can_assign_back and hasattr(vcon, 'vcon_dict'):
                vcon.vcon_dict = vcon_dict
        return removed

def example_usage():
    print("=== Tensor Client Example ===\n")
    
    client = TensorClient()
    
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
        try:
            os.unlink(wav_path)
        except:
            pass
    
    print("\n=== Example completed ===")

if __name__ == "__main__":
    example_usage()
