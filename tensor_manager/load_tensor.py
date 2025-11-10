"""Link helper that ensures a vCon tensor is present on the tensor-manager server."""

from typing import Dict, Optional

from lib.vcon_redis import VconRedis

from .tensor_client import TensorClient


def run(vcon_uuid: str, link_name: str, opts: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Ensure the associated vCon tensor is loaded on the tensor manager server."""
    client = TensorClient()
    if not client.is_server_available():
        return vcon_uuid

    redis = VconRedis()
    vcon = redis.get_vcon(vcon_uuid)
    if vcon is None:
        return None

    options = opts or {}
    device = options.get("device", "cuda:0")
    client.ensure_remote_tensor(vcon, device=device)
    redis.store_vcon(vcon)
    return vcon_uuid

