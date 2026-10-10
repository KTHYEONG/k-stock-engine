"""KIS integration exports."""
from __future__ import annotations

from pathlib import Path

from src.config.providers import KisPolicy, load_provider_policy
from src.config.runtime import load_runtime_config
from src.config.secrets import read_secret
from src.integrations.kis.admission import AdmissionClass, SharedTokenStore, limiter_for
from src.integrations.kis.client import KisClient, KisCredentials

__all__ = ["KisClient", "KisCredentials", "build_pool_client", "default_client"]


def build_pool_client(policy: KisPolicy, *, admission_class: AdmissionClass | None = None) -> KisClient:
    """Client on the configured pool slot with the shared limiter and token store."""
    slot = policy.pool_slot
    if slot is None:
        return KisClient(KisCredentials.from_env(policy), policy=policy)
    resolved = admission_class if admission_class is not None else AdmissionClass(policy.admission_class)
    cache_dir = Path(read_secret(policy.admission_dir_env, default=str(Path.home() / ".cache/kis")))
    creds = KisCredentials.from_pool_slot(int(slot))
    limiter = limiter_for(cache_dir, creds.app_key, resolved, require_marker=policy.require_shared_admission)
    store = SharedTokenStore(cache_dir, creds.app_key)
    return KisClient(creds, policy=policy, token_cache_dir=cache_dir, limiter=limiter, token_store=store)


def default_client() -> KisClient:
    """Pool client when the loaded policy configures a pool slot, else the primary-key client."""
    return build_pool_client(load_provider_policy(load_runtime_config()).kis)
