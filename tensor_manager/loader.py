"""Shared helpers for turning vCons into tensors.

This module centralises the logic for locating audio assets referenced by a vCon,
loading them into torch tensors, and moving them to a requested device.  Both
the tensor manager client and server import these helpers so that fallback paths
stay consistent across processes.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import requests
import torch

try:
    import torchaudio
except Exception:  # pragma: no cover - torchaudio may be unavailable in minimal envs
    torchaudio = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover - soundfile optional
    sf = None

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16_000
_AUDIO_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"}


@dataclass
class LoadedTensor:
    tensor: torch.Tensor
    sample_rate: int


def normalize_vcon(vcon: Any) -> Tuple[Dict[str, Any], Optional[callable]]:
    """Return a mutable dict representation of a vCon and an assign-back callable.

    If ``vcon`` is already a dict we simply return it and ``None`` – callers can
    mutate the dict in-place.  If ``vcon`` looks like a ``vcon.Vcon`` instance
    we use ``to_dict``/``vcon_dict`` to round-trip updates.
    """
    if hasattr(vcon, "to_dict"):
        vcon_dict = vcon.to_dict()

        def assign_back(updated: Dict[str, Any]) -> None:
            if hasattr(vcon, "vcon_dict"):
                vcon.vcon_dict = updated

        return vcon_dict, assign_back

    if isinstance(vcon, dict):
        return vcon, None

    raise TypeError("vcon must be a dict-like object or expose to_dict()/vcon_dict")


def move_tensor_to_device(tensor: torch.Tensor, device: Optional[str]) -> torch.Tensor:
    if device is None:
        return tensor
    device_obj = torch.device(device)
    if tensor.device == device_obj:
        return tensor
    return tensor.to(device_obj, non_blocking=True)


def load_tensor_from_vcon(vcon: Any, device: Optional[str] = None) -> LoadedTensor:
    """Load the primary audio tensor referenced by a vCon.

    The loader works in three passes:
      1. look for ``dialog`` entries of type ``tensor`` that already embed PCM data
         (``data_b64``) – this allows downstream components to persist local caches;
      2. inspect attachments for base64-encoded audio blobs;
      3. fall back to downloading the first ``recording`` dialog entry via HTTP.

    Returns a ``LoadedTensor`` containing the waveform as a 1-D ``torch.float32``
    tensor and the associated sample rate.  The tensor is moved to ``device`` if
    provided (e.g. ``"cuda:0"``).
    """
    vcon_dict, _ = normalize_vcon(vcon)

    loaders = (
        _load_from_embedded_tensor(vcon_dict),
        _load_from_attachments(vcon_dict),
        _load_from_recording_urls(vcon_dict),
    )

    for loader in loaders:
        try:
            loaded = next(loader)
        except StopIteration:
            continue
        if not isinstance(loaded.tensor, torch.Tensor):
            continue
        target = move_tensor_to_device(loaded.tensor, device)
        return LoadedTensor(tensor=target, sample_rate=loaded.sample_rate)

    raise RuntimeError("Unable to locate audio data for vCon")


def _load_from_embedded_tensor(vcon_dict: Dict[str, Any]) -> Iterable[LoadedTensor]:
    dialogs = vcon_dict.get("dialog") or []
    for entry in dialogs:
        if entry.get("type") != "tensor":
            continue
        data_b64 = entry.get("data_b64")
        if not data_b64:
            continue
        dtype = entry.get("dtype", "float32")
        np_dtype = _numpy_dtype(dtype)
        raw = base64.b64decode(data_b64)
        np_array = np.frombuffer(raw, dtype=np_dtype)
        shape = entry.get("shape")
        if shape:
            try:
                np_array = np_array.reshape(shape)
            except Exception:
                logger.warning("Failed to reshape embedded tensor to %s", shape)
        tensor = torch.from_numpy(np_array.copy()).to(torch.float32)
        tensor = tensor.reshape(-1).contiguous()
        sr = int(entry.get("sample_rate") or DEFAULT_SAMPLE_RATE)
        yield LoadedTensor(tensor=tensor, sample_rate=sr)


def _load_from_attachments(vcon_dict: Dict[str, Any]) -> Iterable[LoadedTensor]:
    attachments = vcon_dict.get("attachments") or []
    for attachment in attachments:
        encoding = (attachment.get("encoding") or "").lower()
        if encoding != "base64":
            continue
        ctype = (attachment.get("type") or "").lower()
        if ctype and ctype not in _AUDIO_CONTENT_TYPES:
            continue
        body = attachment.get("body")
        if not body:
            continue
        raw = base64.b64decode(body)
        for loaded in _tensor_from_audio_bytes(raw, attachment.get("sample_rate")):
            yield loaded


def _load_from_recording_urls(vcon_dict: Dict[str, Any]) -> Iterable[LoadedTensor]:
    dialogs = vcon_dict.get("dialog") or []
    for entry in dialogs:
        if entry.get("type") != "recording":
            continue
        url = entry.get("url")
        if not url:
            continue
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Failed to download recording from %s: %s", url, exc)
            continue
        filename = entry.get("filename") or url
        for loaded in _tensor_from_audio_bytes(response.content, entry.get("sample_rate"), filename):
            yield loaded


def _tensor_from_audio_bytes(
    blob: bytes,
    sample_rate: Optional[int],
    filename_hint: Optional[str] = None,
) -> Iterable[LoadedTensor]:
    sample_rate = int(sample_rate or DEFAULT_SAMPLE_RATE)

    if torchaudio is not None:
        suffix = ""
        if filename_hint:
            _, ext = os.path.splitext(filename_hint)
            suffix = ext or ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(blob)
            tmp.flush()
            tmp_path = tmp.name
        try:
            try:
                waveform, sr = torchaudio.load(tmp_path)
                sample_rate = sr or sample_rate
                tensor = waveform.mean(dim=0) if waveform.dim() > 1 else waveform.squeeze(0)
                tensor = tensor.contiguous()
                yield LoadedTensor(tensor=tensor, sample_rate=sample_rate)
                return
            except Exception as exc:
                logger.debug("torchaudio failed to load %s: %s", tmp_path, exc)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if sf is not None:
        try:
            data, sr = sf.read(io.BytesIO(blob), dtype="float32")
            sample_rate = sr or sample_rate
            if data.ndim > 1:
                data = data.mean(axis=1)
            tensor = torch.from_numpy(data.copy())
            yield LoadedTensor(tensor=tensor, sample_rate=sample_rate)
            return
        except Exception as exc:
            logger.debug("soundfile failed to load audio bytes: %s", exc)

    # Fallback: interpret bytes as float32 PCM.
    try:
        np_array = np.frombuffer(blob, dtype=np.float32)
        tensor = torch.from_numpy(np_array.copy())
        yield LoadedTensor(tensor=tensor, sample_rate=sample_rate)
    except Exception as exc:
        logger.error("Unable to interpret audio payload: %s", exc)


def _numpy_dtype(dtype: str):
    mapping = {
        "torch.float32": np.float32,
        "torch.float64": np.float64,
        "torch.int32": np.int32,
        "torch.int64": np.int64,
        "torch.float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
        "float16": np.float16,
    }
    return mapping.get(dtype, np.float32)

