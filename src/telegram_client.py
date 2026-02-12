import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, AsyncGenerator
from telethon import TelegramClient
from telethon.tl.types import Message
from src.config import Secrets

class TelegramFetcher:
    def __init__(self, secrets: Secrets):
        self.client = TelegramClient(
            secrets.telegram_session,
            secrets.telegram_api_id,
            secrets.telegram_api_hash
        )

    async def _fetch_channel_messages(
        self, channel: str, since_dt: datetime, limit: int = 200
    ) -> List[Message]:
        messages = []
        if since_dt.tzinfo is None:
            since_dt = since_dt.replace(tzinfo=timezone.utc)

        try:
            async for msg in self.client.iter_messages(channel, limit=limit):
                if not msg.date:
                    continue
                
                msg_dt = msg.date
                if msg_dt.tzinfo is None:
                    msg_dt = msg_dt.replace(tzinfo=timezone.utc)
                
                if msg_dt < since_dt:
                    break
                
                if getattr(msg, "message", None):
                    messages.append(msg)
        except Exception as e:
            print(f"  ⚠️ Error fetching {channel}: {e}")
        
        return messages

    async def fetch_all(self, channels: List[str], lookback_hours: int = 24) -> AsyncGenerator[tuple, None]:
        """
        Yields (channel_name, messages_list) tuples.
        """
        now_utc = datetime.now(timezone.utc)
        since_dt = now_utc - timedelta(hours=lookback_hours)

        await self.client.start()
        try:
            for channel in channels:
                print(f"Fetching from {channel}...")
                msgs = await self._fetch_channel_messages(channel, since_dt)
                print(f"  -> fetched {len(msgs)} messages")
                yield channel, msgs
        finally:
            await self.client.disconnect()
