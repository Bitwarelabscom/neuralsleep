"""
Tensor Serialization for NeuralSleep

Efficient conversion between PyTorch tensors and bytes for database storage.
Uses native PyTorch serialization with optional zlib compression.
"""

import io
import zlib
import struct
from typing import Optional, Tuple
import torch
import numpy as np


# Magic bytes to identify compressed data
COMPRESSION_MAGIC = b'ZLIB'

# Maximum tensor size before auto-compression (10KB)
AUTO_COMPRESS_THRESHOLD = 10240


def tensor_to_bytes(
    tensor: torch.Tensor,
    compress: bool = False,
    auto_compress: bool = True
) -> bytes:
    """
    Serialize a PyTorch tensor to bytes.

    Args:
        tensor: The tensor to serialize
        compress: Force compression regardless of size
        auto_compress: Automatically compress if > threshold

    Returns:
        Serialized bytes (optionally compressed)
    """
    # Move to CPU if on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Serialize using PyTorch's native format
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    data = buffer.getvalue()

    # Determine if we should compress
    should_compress = compress or (auto_compress and len(data) > AUTO_COMPRESS_THRESHOLD)

    if should_compress:
        compressed = zlib.compress(data, level=6)
        # Only use compression if it actually reduces size
        if len(compressed) < len(data):
            return COMPRESSION_MAGIC + compressed

    return data


def bytes_to_tensor(
    data: bytes,
    device: str = 'cpu',
    expected_shape: Optional[Tuple[int, ...]] = None
) -> torch.Tensor:
    """
    Deserialize bytes back to a PyTorch tensor.

    Args:
        data: Serialized tensor bytes
        device: Target device ('cpu' or 'cuda')
        expected_shape: Optional shape validation

    Returns:
        Deserialized tensor on specified device

    Raises:
        ValueError: If shape doesn't match expected_shape
    """
    # Check for compression magic
    if data[:4] == COMPRESSION_MAGIC:
        data = zlib.decompress(data[4:])

    # Deserialize
    buffer = io.BytesIO(data)
    tensor = torch.load(buffer, map_location=device, weights_only=True)

    # Validate shape if specified
    if expected_shape is not None:
        if tensor.shape != expected_shape:
            raise ValueError(
                f"Tensor shape mismatch: expected {expected_shape}, got {tensor.shape}"
            )

    return tensor


def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_size: int,
    allow_batch: bool = True
) -> bool:
    """
    Validate tensor has expected dimensions.

    Args:
        tensor: Tensor to validate
        expected_size: Expected last dimension size
        allow_batch: Allow batch dimension (2D tensor)

    Returns:
        True if valid, raises ValueError otherwise
    """
    if tensor.dim() == 1:
        if tensor.shape[0] != expected_size:
            raise ValueError(
                f"Expected tensor size {expected_size}, got {tensor.shape[0]}"
            )
    elif tensor.dim() == 2 and allow_batch:
        if tensor.shape[1] != expected_size:
            raise ValueError(
                f"Expected tensor feature size {expected_size}, got {tensor.shape[1]}"
            )
    else:
        raise ValueError(
            f"Expected 1D or 2D tensor, got {tensor.dim()}D"
        )
    return True


def compress_bytes(data: bytes, level: int = 6) -> bytes:
    """
    Compress bytes using zlib.

    Args:
        data: Raw bytes to compress
        level: Compression level (1-9, 6 is default)

    Returns:
        Compressed bytes with magic header
    """
    compressed = zlib.compress(data, level=level)
    return COMPRESSION_MAGIC + compressed


def decompress_bytes(data: bytes) -> bytes:
    """
    Decompress bytes if compressed.

    Args:
        data: Potentially compressed bytes

    Returns:
        Decompressed bytes
    """
    if data[:4] == COMPRESSION_MAGIC:
        return zlib.decompress(data[4:])
    return data


def tensor_to_numpy_bytes(tensor: torch.Tensor) -> bytes:
    """
    Alternative serialization using numpy (smaller for simple tensors).

    Args:
        tensor: PyTorch tensor

    Returns:
        Serialized bytes in numpy format
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()

    arr = tensor.numpy()
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    return buffer.getvalue()


def numpy_bytes_to_tensor(
    data: bytes,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Deserialize numpy bytes to tensor.

    Args:
        data: Numpy serialized bytes
        device: Target device

    Returns:
        PyTorch tensor
    """
    buffer = io.BytesIO(data)
    arr = np.load(buffer, allow_pickle=False)
    return torch.from_numpy(arr).to(device)


def pack_tensor_compact(tensor: torch.Tensor) -> bytes:
    """
    Ultra-compact serialization for float32 tensors.

    Format: [4 bytes ndim][4 bytes per dim shape][float32 data]

    Args:
        tensor: Float32 tensor to pack

    Returns:
        Packed bytes
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Convert to contiguous float32
    tensor = tensor.contiguous().float()

    # Pack header: ndim + shape
    ndim = tensor.dim()
    header = struct.pack('<I', ndim)
    for dim in tensor.shape:
        header += struct.pack('<I', dim)

    # Pack data
    data = tensor.numpy().tobytes()

    return header + data


def unpack_tensor_compact(data: bytes, device: str = 'cpu') -> torch.Tensor:
    """
    Unpack ultra-compact tensor format.

    Args:
        data: Packed bytes
        device: Target device

    Returns:
        PyTorch tensor
    """
    # Read header
    ndim = struct.unpack('<I', data[:4])[0]
    shape = []
    offset = 4
    for _ in range(ndim):
        dim = struct.unpack('<I', data[offset:offset+4])[0]
        shape.append(dim)
        offset += 4

    # Read data
    arr = np.frombuffer(data[offset:], dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr.copy()).to(device)


# Size estimates for different tensor dimensions
TENSOR_SIZES = {
    'semantic_state': 256,      # 256 floats = 1024 bytes
    'episodic_state': 128,      # 128 floats = 512 bytes
    'working_state': 256,       # 256 floats = 1024 bytes
    'working_input': 512,       # 512 floats = 2048 bytes
    'working_output': 128,      # 128 floats = 512 bytes
    'experience': 128,          # 128 floats = 512 bytes
}


def estimate_storage_size(
    tensor_type: str,
    count: int = 1,
    compressed: bool = False
) -> int:
    """
    Estimate storage size for tensors.

    Args:
        tensor_type: Type from TENSOR_SIZES
        count: Number of tensors
        compressed: Whether compression is used

    Returns:
        Estimated bytes
    """
    if tensor_type not in TENSOR_SIZES:
        raise ValueError(f"Unknown tensor type: {tensor_type}")

    # 4 bytes per float32 + ~100 bytes overhead
    base_size = TENSOR_SIZES[tensor_type] * 4 + 100
    total = base_size * count

    if compressed:
        # Typical compression ratio for neural network weights
        total = int(total * 0.6)

    return total
