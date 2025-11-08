# GPU Tensor Server

A REST API server for managing GPU tensors across processes using **CUDA IPC (Inter-Process Communication)** when available, while gracefully falling back to CPU tensors when GPUs are not present.

## Key Features

🚀 **Zero-Copy Sharing**: When CUDA is available tensors stay on GPU and are shared by handle  
🧠 **CPU Fallback**: Runs even on CPU-only hosts by serialising tensors as base64 payloads  
🎯 **CUDA IPC**: Uses PyTorch's CUDA IPC for true cross-process GPU memory sharing  
🔒 **Device Control**: Force explicit CUDA device allocation  
🆔 **Unique References**: UUID-based tensor IDs for reliable access  
⚡ **REST API**: Simple HTTP interface for any language/process  

## Installation

```bash
# Install dependencies
./install-stuff.sh
source .venv/bin/activate
```

## Usage

### Start the Server

```bash
python tensor_server.py
```

The server will run on `http://localhost:8000`

### API Endpoints

#### 1. Load WAV file as tensor
```bash
curl -X POST \
  -F "wav_file=@your_audio.wav" \
  -F "cuda_device=0" \
  http://localhost:8000/tensors
```

Returns:
```json
{
  "tensor_id": "550e8400-e29b-41d4-a716-446655440000",
  "shape": [32000],
  "dtype": "torch.float32", 
  "device": "cuda:0",
  "sample_rate": 16000,
  "message": "Tensor loaded successfully on cuda:0"
}
```

#### 2. Get tensor information
```bash
curl http://localhost:8000/tensors/{tensor_id}
```

#### 3. Get tensor handle (IPC or base64 payload)
```bash
curl http://localhost:8000/tensors/{tensor_id}/handle
```

If CUDA is available you'll receive an IPC handle and metadata for zero-copy access:
```json
{
  "tensor_id": "550e8400-e29b-41d4-a716-446655440000",
  "ipc_handle": "base64-encoded-cuda-ipc-handle",
  "shape": [32000],
  "dtype": "torch.float32",
  "device": "cuda:0",
  "data_ptr": 140000000000000,
  "element_size": 4,
  "numel": 32000,
  "sample_rate": 16000
}
```

On CPU-only systems you'll instead receive a base64 payload that can be rebuilt client-side:
```json
{
  "tensor_id": "550e8400-e29b-41d4-a716-446655440000",
  "data_b64": "base64-encoded-tensor",
  "shape": [32000],
  "dtype": "torch.float32",
  "device": "cpu",
  "element_size": 4,
  "numel": 32000,
  "sample_rate": 16000
}
```

#### 4. Delete tensor
```bash
curl -X DELETE http://localhost:8000/tensors/{tensor_id}
```

#### 5. List all tensors
```bash
curl http://localhost:8000/tensors
```

#### 6. Get CUDA device info
```bash
curl http://localhost:8000/cuda/info
```

## Client Usage

Use the `TensorClient` class for easy integration:

```python
from tensor_client import TensorClient

client = TensorClient()

# Upload WAV file
tensor_id = client.upload_wav_file("audio.wav", cuda_device=0)

# Access the tensor (zero-copy on CUDA, decoded otherwise)
shared_tensor = client.access_shared_tensor(tensor_id)

# Tensor is moved to CUDA automatically if available
shared_tensor.mul_(2.0)

# Cleanup
client.delete_tensor(tensor_id)
```

## Testing

```bash
pytest test_tensor_server.py -v
```

## Examples

```bash
# Basic server usage
python usage_example.py

# Client library demonstration  
python tensor_client.py
```

## How It Works

1. **Server loads WAV** → Tensor lives on GPU when CUDA+CuPy are available, otherwise on CPU.
2. **Client requests handle** → Server returns a CUDA IPC handle or a base64 payload depending on the backend.
3. **Client consumes handle** → Client performs zero-copy mapping on CUDA or reconstructs the CPU tensor from bytes.

## Features

- ✅ **True zero-copy GPU tensor sharing**
- ✅ **CUDA IPC memory handles**  
- ✅ **Explicit CUDA device allocation**
- ✅ **Unique tensor IDs for reference**
- ✅ **RESTful API interface**
- ✅ **Client library for easy integration**
- ✅ **Comprehensive test suite**
- ✅ **Manual memory management (no refcounting)**
- ✅ **Cross-process tensor access**
