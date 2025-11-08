# GPU Tensor Server

A REST API server for managing GPU tensors across processes using **CUDA IPC (Inter-Process Communication)**. Loads WAV files as tensors on specific CUDA devices and shares them via memory handles - **tensors never leave the GPU**.

## Key Features

🚀 **Zero-Copy Sharing**: Tensors stay on GPU, only memory handles are shared  
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

#### 3. Get CUDA IPC handle (for accessing tensor)
```bash
curl http://localhost:8000/tensors/{tensor_id}/handle
```

Returns IPC handle and metadata needed to access the GPU tensor:
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

client = TensorClient(host="localhost", port=8000)

# Upload WAV file
tensor_id = client.upload_wav_file("audio.wav", cuda_device=0)

# Access the shared GPU tensor directly (zero-copy!)
shared_tensor = client.access_shared_tensor(tensor_id)

# Tensor is on GPU and can be modified in-place
shared_tensor.mul_(2.0)  # All processes see this change!

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

1. **Server loads WAV** → GPU tensor with `tensor.share_memory_()`
2. **Client requests handle** → Server returns CUDA IPC handle (base64 encoded)
3. **Client decodes handle** → Maps same GPU memory in client process
4. **Direct GPU access** → Client can read/write tensor without copying data

**Important**: The tensor data **never** moves off the GPU. Only memory handles are transmitted over HTTP.

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
