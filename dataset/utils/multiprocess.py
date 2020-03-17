# Copyright 2020 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Callable, List, Iterable
import queue
from multiprocessing import Queue, Process


def process(
        func: Callable,
        in_queue: Queue,
        out_queue: Queue) -> None:
    """Process target function.

    Args:
        func (Callable): processing function.
        in_queue (Queue): process target queue.
        out_queue (Queue): process result queue.

    """
    while True:
        item = in_queue.get()
        try:
            ret = func(*item)
        except Exception:
            continue
        out_queue.put(ret)


class Map:
    """Multiprocessing map class."""

    def __init__(
            self,
            func: Callable,
            num: int,
            queue_size: int = 1000):
        """Initialize process."""
        self.in_queue: Queue = Queue()
        self.out_queue: Queue = Queue(maxsize=queue_size)
        self.processes = [
            Process(target=process, args=(func, self.in_queue, self.out_queue))
            for _ in range(num)]
        for p in self.processes:
            p.start()

    def __del__(self):
        self.close()

    def close(self):
        """End processing."""
        self.in_queue.close()
        self.in_queue.join_thread()

        self.out_queue.close()
        self.out_queue.join_thread()

        for p in self.processes:
            if p.is_alive():
                p.terminate()
        for p in self.processes:
            if p.is_alive():
                p.join()
        for p in self.processes:
            if p.is_alive():
                p.close()

    def put(
            self,
            items: Iterable) -> None:
        """Put items to process queue."""
        for item in items:
            self.in_queue.put(item)

    def get(
            self,
            num: int) -> List:
        """Get items to process queue."""
        items = []
        try:
            for _ in range(num):
                items.append(self.out_queue.get(False))
        except queue.Empty:
            pass
        return items

    def in_queue_empty(self):
        """Inqueue is empty."""
        return self.in_queue.empty()

    def out_queue_empty(self):
        """Outqueue is empty."""
        return self.out_queue.empty()
