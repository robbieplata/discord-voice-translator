from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, TypeVar

from .config import (
    OPENAI_CHAT_MAX_INPUT_CHARS,
    OPENAI_MAX_CONCURRENT_REQUESTS,
    OPENAI_MAX_REQUESTS_PER_SEC,
    OPENAI_TTS_MAX_INPUT_CHARS,
)

T = TypeVar("T")


@dataclass(frozen=True)
class TruncationResult:
    text: str
    truncated: bool
    original_len: int


def truncate_chars(text: str, max_chars: int) -> TruncationResult:
    original_len = len(text)
    if original_len <= max_chars:
        return TruncationResult(text=text, truncated=False, original_len=original_len)
    return TruncationResult(text=text[:max_chars], truncated=True, original_len=original_len)


def truncate_for_tts(text: str) -> TruncationResult:
    return truncate_chars(text, OPENAI_TTS_MAX_INPUT_CHARS)


def truncate_for_chat(text: str) -> TruncationResult:
    return truncate_chars(text, OPENAI_CHAT_MAX_INPUT_CHARS)


class AsyncOpenAILimiter:
    """Global async limiter for OpenAI calls (pacing + concurrency)."""

    def __init__(self, max_rps: float, max_concurrency: int):
        self._max_rps = float(max_rps)
        self._min_interval = (1.0 / self._max_rps) if self._max_rps > 0 else 0.0
        self._pacer = asyncio.Lock()
        self._last_at = 0.0
        self._sem = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def acquire(self) -> None:
        await self._sem.acquire()
        try:
            if self._min_interval <= 0:
                return
            async with self._pacer:
                now = time.monotonic()
                wait = self._min_interval - (now - self._last_at)
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_at = time.monotonic()
        except Exception:
            self._sem.release()
            raise

    def release(self) -> None:
        self._sem.release()

    async def __aenter__(self) -> "AsyncOpenAILimiter":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.release()


class ThreadOpenAILimiter:
    """Thread-safe limiter for OpenAI calls in worker threads."""

    def __init__(self, max_rps: float):
        self._max_rps = float(max_rps)
        self._min_interval = (1.0 / self._max_rps) if self._max_rps > 0 else 0.0
        self._lock = threading.Lock()
        self._last_at = 0.0

    def pace(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_at)
            if wait > 0:
                time.sleep(wait)
            self._last_at = time.monotonic()


_async_limiter: Optional[AsyncOpenAILimiter] = None
_thread_limiter: Optional[ThreadOpenAILimiter] = None


def get_async_limiter() -> AsyncOpenAILimiter:
    global _async_limiter
    if _async_limiter is None:
        _async_limiter = AsyncOpenAILimiter(
            max_rps=OPENAI_MAX_REQUESTS_PER_SEC,
            max_concurrency=OPENAI_MAX_CONCURRENT_REQUESTS,
        )
    return _async_limiter


def get_thread_limiter() -> ThreadOpenAILimiter:
    global _thread_limiter
    if _thread_limiter is None:
        _thread_limiter = ThreadOpenAILimiter(max_rps=OPENAI_MAX_REQUESTS_PER_SEC)
    return _thread_limiter


async def limited_call(fn: Callable[[], Awaitable[T]]) -> T:
    """Run an awaitable under the global async limiter."""
    limiter = get_async_limiter()
    async with limiter:
        return await fn()


def thread_limited_call(fn: Callable[[], T]) -> T:
    """Run a synchronous function under the global thread pacer."""
    get_thread_limiter().pace()
    return fn()
