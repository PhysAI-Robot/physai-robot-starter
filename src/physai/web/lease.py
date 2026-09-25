"""Who may command each robot instance, and the one command waiting for it.

Manual control uses a short per-instance lease: the first client to submit a
command becomes that instance's controller, and another client's command is
rejected while the lease is alive. Letting it expire, or the client
disconnecting, returns the instance to its hold action or policy. Each
instance keeps at most one pending command; a newer one replaces it.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterable

from ..contracts import Action

DEFAULT_TIMEOUT = 0.35


class ControlLease:
    def __init__(
        self,
        instance_ids: Iterable[str],
        *,
        timeout: float = DEFAULT_TIMEOUT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._commands: dict[str, queue.Queue[Action]] = {
            instance_id: queue.Queue(maxsize=1) for instance_id in instance_ids
        }
        self._owner: dict[str, str] = {}
        self._deadline: dict[str, float] = {}
        self._timeout = timeout
        self._clock = clock
        self._lock = threading.Lock()

    def submit(self, instance_id: str, action: Action, source: str) -> None:
        """Claim (or renew) the instance for ``source`` and queue its command."""
        now = self._clock()
        with self._lock:
            owner = self._owner.get(instance_id)
            deadline = self._deadline.get(instance_id, 0.0)
            if owner not in (None, source) and now < deadline:
                raise PermissionError(
                    f"robot instance {instance_id!r} is controlled by another client"
                )
            self._owner[instance_id] = source
            self._deadline[instance_id] = now + self._timeout
        self._discard(instance_id)
        try:
            self._commands[instance_id].put_nowait(action)
        except queue.Full:
            pass

    def latest(self, instance_id: str) -> Action | None:
        """The waiting command, or None; an expired lease drops its command."""
        expired = False
        with self._lock:
            deadline = self._deadline.get(instance_id)
            if deadline is not None and self._clock() >= deadline:
                self._owner.pop(instance_id, None)
                self._deadline.pop(instance_id, None)
                expired = True
        if expired:
            self._discard(instance_id)
            return None
        try:
            return self._commands[instance_id].get_nowait()
        except queue.Empty:
            return None

    def release(self, source: str) -> None:
        """Drop every instance ``source`` owns, and only those."""
        with self._lock:
            owned = [
                instance_id
                for instance_id, owner in self._owner.items()
                if owner == source
            ]
            for instance_id in owned:
                self._owner.pop(instance_id, None)
                self._deadline.pop(instance_id, None)
        for instance_id in owned:
            self._discard(instance_id)

    def discard_all(self) -> None:
        """Forget every waiting command (the leases themselves are kept)."""
        for instance_id in self._commands:
            self._discard(instance_id)

    def _discard(self, instance_id: str) -> None:
        try:
            self._commands[instance_id].get_nowait()
        except queue.Empty:
            pass


__all__ = ["DEFAULT_TIMEOUT", "ControlLease"]
