import asyncio
from typing import Callable, Any

class RequestQueue:
    def __init__(self, max_concurrent: int = 1):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue_size = 0

    async def run(self, func: Callable, *args, **kwargs) -> Any:
        self.queue_size += 1
        async with self.semaphore:
            self.queue_size -= 1
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    @property
    def waiting(self) -> int:
        return self.queue_size

request_queue = RequestQueue(max_concurrent=1)
