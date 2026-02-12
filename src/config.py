import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class Secrets:
    telegram_api_id: int
    telegram_api_hash: str
    telegram_session: str
    openai_api_key: str
    openai_base_url: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    lmstudio_base_url: Optional[str] = None
    lmstudio_model: Optional[str] = None
    email_config: Optional[Dict[str, Any]] = None

@dataclass
class Config:
    secrets: Secrets
    channels: Dict[str, List[str]]
    base_dir: Path

def load_secrets(path: Path) -> Secrets:
    """Load secrets from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Secrets file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def pick(*keys, default=None):
        for k in keys:
            cur = data
            ok = True
            for part in k.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            if ok and cur not in (None, "", []):
                return cur
        return default

    # Telegram
    tg_api_id = pick("TELEGRAM_API_ID", "telegram_api_id", "api_id", "telegram.api_id")
    tg_api_hash = pick("TELEGRAM_API_HASH", "telegram_api_hash", "api_hash", "telegram.api_hash")
    tg_session = pick("TELEGRAM_SESSION", "telegram_session", "session", "telegram.session", default="telegramdaily")

    # OpenAI / LLM
    oa_key = pick("OPENAI_API_KEY", "openai_api_key", "OPENAI_KEY", "openai_key", "openai.api_key", default="lm-studio")
    oa_base = pick("OPENAI_BASE_URL", "openai_base_url", "openai.base_url")
    oa_model = pick("OPENAI_MODEL", "openai_model", "openai.model", default="gpt-4o-mini")

    lm_base = pick("LMSTUDIO_BASE_URL", "lmstudio_base_url", "lmstudio.base_url", "lm_base_url")
    lm_model = pick("LMSTUDIO_MODEL", "lmstudio_model", "lmstudio.model", "lm_model")

    # Email
    email_cfg = pick("EMAIL", "email", default={})

    if not tg_api_id or not tg_api_hash:
        raise ValueError("Missing TELEGRAM_API_ID or TELEGRAM_API_HASH in secrets.")

    return Secrets(
        telegram_api_id=int(tg_api_id),
        telegram_api_hash=str(tg_api_hash),
        telegram_session=str(tg_session),
        openai_api_key=str(oa_key),
        openai_base_url=oa_base,
        openai_model=str(oa_model),
        lmstudio_base_url=lm_base,
        lmstudio_model=lm_model,
        email_config=email_cfg
    )

def load_channels(path: Path) -> Dict[str, List[str]]:
    """Load channel lists from a JSON file."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_config(base_dir: Optional[Path] = None) -> Config:
    """Load the full configuration."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    secrets_path = base_dir / "config" / "secrets.json"
    channels_path = base_dir / "config" / "channels.json"

    return Config(
        secrets=load_secrets(secrets_path),
        channels=load_channels(channels_path),
        base_dir=base_dir
    )
