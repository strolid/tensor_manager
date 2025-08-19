#!/usr/bin/env python3

import os
import time
import torch
import numpy as np
import soundfile as sf
import tempfile
import multiprocessing as mp
from tensor_client import TensorClient

def create_sample_wav():
    sample_rate = 16000
    duration = 2.0
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        sf.write(tmp_file.name, audio_data, sample_rate)
        return tmp_file.name

def uploader_process():
    """Process A: Upload the tensor"""
    print("🔼 UPLOADER PROCESS: Starting...")
    
    client = TensorClient()
    wav_path = create_sample_wav()
    
    try:
        tensor_id = client.upload_wav_file(wav_path, cuda_device=0)
        print(f"🔼 UPLOADER: Uploaded tensor with ID: {tensor_id}")
        
        shared_tensor = client.access_shared_tensor(tensor_id)
        print(f"🔼 UPLOADER: Initial mean value: {shared_tensor.mean().item():.6f}")
        
        time.sleep(3)
        
        print("🔼 UPLOADER: Multiplying tensor by 10...")
        shared_tensor.mul_(10.0)
        print(f"🔼 UPLOADER: New mean value: {shared_tensor.mean().item():.6f}")
        
        time.sleep(5)
        
        print("🔼 UPLOADER: Cleaning up...")
        client.delete_tensor(tensor_id)
        print("🔼 UPLOADER: Tensor deleted. Exiting.")
        
        return tensor_id
        
    finally:
        os.unlink(wav_path)

def reader_process(process_id, tensor_id):
    """Processes B, C, D: Just read the tensor"""
    print(f"👁️  READER {process_id}: Starting...")
    
    client = TensorClient()
    
    try:
        time.sleep(1)
        
        shared_tensor = client.access_shared_tensor(tensor_id)
        print(f"👁️  READER {process_id}: Connected! Tensor shape: {shared_tensor.shape}")
        print(f"👁️  READER {process_id}: Initial mean: {shared_tensor.mean().item():.6f}")
        
        time.sleep(4)
        
        print(f"👁️  READER {process_id}: Checking for changes...")
        new_mean = shared_tensor.mean().item()
        print(f"👁️  READER {process_id}: Updated mean: {new_mean:.6f}")
        
        if abs(new_mean) > 1.0:
            print(f"✅ READER {process_id}: SUCCESS! Saw changes from uploader process!")
        else:
            print(f"❌ READER {process_id}: No changes detected")
            
        time.sleep(2)
        print(f"👁️  READER {process_id}: Exiting...")
        
    except Exception as e:
        print(f"❌ READER {process_id}: Error: {e}")

def modifier_process(process_id, tensor_id):
    """Process E: Modify the tensor"""
    print(f"✏️  MODIFIER {process_id}: Starting...")
    
    client = TensorClient()
    
    try:
        time.sleep(2)
        
        shared_tensor = client.access_shared_tensor(tensor_id)
        print(f"✏️  MODIFIER {process_id}: Connected! Initial mean: {shared_tensor.mean().item():.6f}")
        
        time.sleep(6)
        
        print(f"✏️  MODIFIER {process_id}: Adding 100 to all values...")
        shared_tensor.add_(100.0)
        print(f"✏️  MODIFIER {process_id}: New mean: {shared_tensor.mean().item():.6f}")
        
        time.sleep(1)
        print(f"✏️  MODIFIER {process_id}: Exiting...")
        
    except Exception as e:
        print(f"❌ MODIFIER {process_id}: Error: {e}")

def demo_multi_process_sharing():
    """Demonstrate multiple processes accessing the same GPU tensor"""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Cannot run multi-process demo.")
        return
    
    print("🚀 === MULTI-PROCESS GPU TENSOR SHARING DEMO ===\n")
    print("This demo shows multiple processes accessing the SAME GPU tensor:")
    print("  - Process A (Uploader): Creates tensor, modifies it")
    print("  - Processes B,C,D (Readers): Read tensor, see changes in real-time")  
    print("  - Process E (Modifier): Also modifies tensor")
    print("  - All processes access THE SAME GPU MEMORY via IPC handles\n")
    
    client = TensorClient()
    wav_path = create_sample_wav()
    
    try:
        print("📤 Main process: Uploading initial tensor...")
        tensor_id = client.upload_wav_file(wav_path, cuda_device=0)
        print(f"📤 Main process: Tensor ID: {tensor_id}\n")
        
        processes = []
        
        reader_proc_1 = mp.Process(target=reader_process, args=(1, tensor_id))
        reader_proc_2 = mp.Process(target=reader_process, args=(2, tensor_id))
        reader_proc_3 = mp.Process(target=reader_process, args=(3, tensor_id))
        modifier_proc = mp.Process(target=modifier_process, args=(1, tensor_id))
        
        processes = [reader_proc_1, reader_proc_2, reader_proc_3, modifier_proc]
        
        for p in processes:
            p.start()
            time.sleep(0.2)
        
        time.sleep(10)
        
        for p in processes:
            if p.is_alive():
                p.join(timeout=2)
        
        print("\n🧹 Main process: Cleaning up...")
        client.delete_tensor(tensor_id)
        print("✅ Demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    finally:
        os.unlink(wav_path)

if __name__ == "__main__":
    demo_multi_process_sharing()
