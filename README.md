# GPU Tensor Server

A REST API server for managing GPU tensors across processes. Loads WAV files as tensors on specific CUDA devices and exposes them through serialized handles for easy retrieval.

## Key Features

🎯 **GPU-backed tensors**: Load audio directly into CUDA tensors  
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

#### 3. Get tensor handle (base64 payload plus metadata)
```bash
curl http://localhost:8000/tensors/{tensor_id}/handle
```

Returns encoded tensor data and metadata needed to reconstruct it:
```json
{
  "tensor_id": "550e8400-e29b-41d4-a716-446655440000",
  "data_b64": "base64-encoded-tensor-bytes",
  "shape": [32000],
  "dtype": "torch.float32",
  "device": "cuda:0",
  "element_size": 4,
  "numel": 32000,
  "sample_rate": 16000,
  "nbytes": 128000
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

# Access the tensor
shared_tensor = client.access_shared_tensor(tensor_id)

# Tensor is loaded to GPU if available
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

1. **Server loads WAV** → tensor is moved onto the requested CUDA device.
2. **Client requests handle** → server returns metadata plus a base64 payload of tensor contents.
3. **Client decodes handle** → tensor is rebuilt locally (on GPU if available).

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
