"""Slow dashboard actions off the server's event loop (owner 11-04: clicking an agent froze the dashboard while the new
voice warmed up on the loop). Standard library only, so the dashboard replay harness (tools/ui_bench) uses it too.

Actions run in groups, one at a time per group, in arrival order, each on a worker thread:
  "pipeline"  agent switch, voice switch, clear history, agent create/delete, voice import (they abort the reply or
              rebuild the agent and voice tables the others read)
  "devices"   audio device list and change (they touch only the call's audio streams)
Only agent switches are replaceable: a newer one replaces one that is still waiting, which is answered at once with
REPLACED and never runs (the owner changed their mind; a stale switch would abort and warm up for nothing).

One ActionGroups per server app, created on its event loop (asyncio locks belong to one loop)."""
from __future__ import annotations

import asyncio

REPLACED = "replaced by a newer switch"


class ActionFailed(Exception):
    """Raised by an action's work to report a failure with a message for the dashboard."""


class ActionGroups:
    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._waiting: asyncio.Future | None = None      # the agent switch waiting for the "pipeline" lock
        self._tasks: set = set()

    def _lock(self, group: str) -> asyncio.Lock:
        if group not in self._locks:
            self._locks[group] = asyncio.Lock()
        return self._locks[group]

    def busy(self, group: str) -> bool:
        return self._lock(group).locked()

    async def run(self, group: str, work, replaceable: bool = False):
        """Run `work()` on a worker thread when the group is free. Returns (ok, result, error)."""
        lock = self._lock(group)
        me = None
        if replaceable:
            me = asyncio.get_running_loop().create_future()
            if self._waiting is not None and not self._waiting.done():
                self._waiting.set_result(True)        # the older waiting switch is replaced
            self._waiting = me
        acquire = asyncio.ensure_future(lock.acquire())
        try:
            if me is not None:
                await asyncio.wait({acquire, me}, return_when=asyncio.FIRST_COMPLETED)
                if me.done():                          # replaced while waiting
                    if acquire.done() and not acquire.cancelled():
                        lock.release()
                    else:
                        acquire.cancel()
                    return False, None, REPLACED
            else:
                await acquire
        except asyncio.CancelledError:
            if acquire.done() and not acquire.cancelled():
                lock.release()
            else:
                acquire.cancel()
            if me is not None and self._waiting is me:
                self._waiting = None
            raise
        if me is not None and self._waiting is me:
            self._waiting = None
        # The lock is ours. The work runs in a task that releases it only when the thread has finished, and the caller
        # waits through shield: a cancelled caller (closed connection, shutdown) never lets the next action overlap.
        task = asyncio.ensure_future(self._work(lock, work))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return await asyncio.shield(task)

    @staticmethod
    async def _work(lock: asyncio.Lock, work):
        try:
            result = await asyncio.to_thread(work)
            return True, result, None
        except ActionFailed as e:
            return False, None, str(e)[:160]
        except Exception as e:  # noqa: BLE001
            return False, None, f"{type(e).__name__}: {e}"[:160]
        finally:
            lock.release()
