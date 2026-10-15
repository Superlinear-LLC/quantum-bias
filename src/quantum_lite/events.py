"""SSE event bus: bounded per-client queues; overflow is counted and reported."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import AsyncIterator, Callable

QUEUE_SIZE = 256
HEARTBEAT_SECONDS = 15.0
GLOBAL_TYPES = frozenset({"job", "pool", "error"})
CLOSE = "__close__"  # sentinel event: the stream generator returns when it reads this


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Subscription:
    def __init__(self, session_id: str | None) -> None:
        self.session_id = session_id  # None = global feed
        self.queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue(maxsize=QUEUE_SIZE)
        self.dropped = 0

    def close(self) -> None:
        """End this subscriber's stream: pending events are dropped, the sentinel always fits."""
        while True:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.queue.put_nowait((CLOSE, {}))

    def offer(self, event: str, data: dict) -> None:
        try:
            self.queue.put_nowait((event, data))
        except asyncio.QueueFull:
            self.dropped += 1


class EventBus:
    def __init__(self) -> None:
        self._subs: set[Subscription] = set()
        self._closing = False

    def subscribe(self, session_id: str | None = None) -> Subscription:
        sub = Subscription(session_id)
        self._subs.add(sub)
        if self._closing:
            sub.close()  # a client that connects during shutdown is let go at once
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        self._subs.discard(sub)

    def close_all(self) -> None:
        """Shutdown: end every open stream so the server can close its connections."""
        self._closing = True
        for sub in list(self._subs):
            sub.close()

    def has_subscribers(self, session_id: str) -> bool:
        return any(sub.session_id == session_id for sub in self._subs)

    def publish(self, event: str, data: dict, session_id: str | None = None) -> None:
        payload = {**data, "ts": now_iso()}
        if session_id is not None:
            payload["session_id"] = session_id
        for sub in list(self._subs):
            if sub.session_id is None:
                if event in GLOBAL_TYPES:
                    sub.offer(event, payload)
            elif sub.session_id == session_id:
                sub.offer(event, payload)


def format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def sse_stream(sub: Subscription, bus: EventBus, heartbeat: float = HEARTBEAT_SECONDS,
                     first: tuple[str, dict] | None = None,
                     on_close: Callable[[], None] | None = None) -> AsyncIterator[str]:
    try:
        if first is not None:
            yield format_sse(*first)
        while True:
            try:
                event, data = await asyncio.wait_for(sub.queue.get(), timeout=heartbeat)
            except asyncio.TimeoutError:
                yield format_sse("heartbeat", {"ts": now_iso(), "dropped": _take_dropped(sub)})
                continue
            if event == CLOSE:
                return
            dropped = _take_dropped(sub)
            if dropped:
                data = {**data, "dropped": dropped}
            yield format_sse(event, data)
    finally:
        bus.unsubscribe(sub)
        if on_close is not None:
            on_close()


def _take_dropped(sub: Subscription) -> int:
    dropped, sub.dropped = sub.dropped, 0
    return dropped
