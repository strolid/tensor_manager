#!/usr/bin/env python3

import requests
import numpy as np
import soundfile as sf
import tempfile
import os

def create_sample_wav():
    sample_rate = 16000
    duration = 2.0
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        sf.write(tmp_file.name, audio_data, sample_rate)
        return tmp_file.name

def main():
    server_url = "http://localhost:8000"
    
    print("=== GPU Tensor Server Usage Example ===\n")
    
    print("1. Checking server status...")
    response = requests.get(f"{server_url}/")
    print(f"Server response: {response.json()}")
    
    print("\n2. Checking CUDA info...")
    response = requests.get(f"{server_url}/cuda/info")
    cuda_info = response.json()
    print(f"CUDA available: {cuda_info['cuda_available']}")
    if cuda_info['cuda_available']:
        print(f"Device count: {cuda_info['device_count']}")
        for device in cuda_info['devices']:
            print(f"  Device {device['device_id']}: {device['name']}")
    
    if not cuda_info['cuda_available']:
        print("CUDA not available, cannot proceed with tensor operations")
        return
    
    print("\n3. Creating sample WAV file...")
    wav_file_path = create_sample_wav()
    print(f"Created: {wav_file_path}")
    
    print("\n4. Uploading WAV file as tensor...")
    with open(wav_file_path, 'rb') as f:
        files = {'wav_file': ('example.wav', f, 'audio/wav')}
        data = {'cuda_device': 0}
        response = requests.post(f"{server_url}/tensors", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        tensor_id = result['tensor_id']
        print(f"Tensor created successfully!")
        print(f"  Tensor ID: {tensor_id}")
        print(f"  Shape: {result['shape']}")
        print(f"  Device: {result['device']}")
        print(f"  Sample rate: {result['sample_rate']}")
        
        print("\n5. Getting tensor info...")
        response = requests.get(f"{server_url}/tensors/{tensor_id}")
        info = response.json()
        print(f"  Original filename: {info['original_filename']}")
        print(f"  Dtype: {info['dtype']}")
        
        print("\n6. Listing all tensors...")
        response = requests.get(f"{server_url}/tensors")
        tensors_list = response.json()
        print(f"  Total tensors in memory: {tensors_list['count']}")
        
        print("\n7. Getting tensor IPC handle...")
        response = requests.get(f"{server_url}/tensors/{tensor_id}/handle")
        handle_data = response.json()
        print(f"  Retrieved IPC handle (first 50 chars): {handle_data['ipc_handle'][:50]}...")
        print(f"  Data pointer: 0x{handle_data['data_ptr']:x}")
        print(f"  Element size: {handle_data['element_size']} bytes")
        print(f"  Number of elements: {handle_data['numel']}")
        print(f"  Tensor shape: {handle_data['shape']}")
        print("  NOTE: This handle can be used by other processes to access the GPU tensor directly!")
        
        print("\n8. Deleting tensor...")
        response = requests.delete(f"{server_url}/tensors/{tensor_id}")
        if response.status_code == 200:
            print("  Tensor deleted successfully!")
        
        print("\n9. Verifying deletion...")
        response = requests.get(f"{server_url}/tensors/{tensor_id}")
        if response.status_code == 404:
            print("  Confirmed: tensor no longer exists")
    
    else:
        print(f"Failed to create tensor: {response.text}")
    
    os.unlink(wav_file_path)
    print("\n=== Example completed ===")

if __name__ == "__main__":
    main()
