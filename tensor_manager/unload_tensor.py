"""Link helper that unloads a vCon tensor from the tensor-manager server."""

from typing import Dict, Optional

from lib.vcon_redis import VconRedis

from .tensor_client import TensorClient


def run(vcon_uuid: str, link_name: str, opts: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Remove the associated vCon tensor from the tensor manager server if present."""
    client = TensorClient()
    if not client.is_server_available():
        return vcon_uuid

    redis = VconRedis()
    vcon = redis.get_vcon(vcon_uuid)
    if vcon is None:
        return None

    client.unload_remote_tensor(vcon)
    redis.store_vcon(vcon)
    return vcon_uuid

