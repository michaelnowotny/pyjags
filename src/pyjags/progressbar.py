# Copyright (C) 2016 Tomasz Miasko
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

__all__ = ['const_time_partition', 'progressbar']

import math
import sys
import threading
import time
from datetime import timedelta
from functools import wraps


default_timer = getattr(time, 'monotonic', time.time)


def synchronized(func):
    lock = threading.Lock()
    @wraps(func)
    def inner(*args, **kwargs):
        with lock:
            func(*args, **kwargs)
    return inner


def const_time_partition(iterations, period, timer=default_timer):
    """
    Divides iterations into roughly constant time sub-iterations. Time
    necessary to complete a single iteration is estimated as elapsed time
    divided by all already completed iterations.

    Parameters
    ----------
    steps : int
        A non-negative integer specifying total number of steps to execute.
    period : float
        A positive float number describing desired period between yields from
        generator.
    timer : callable, optional
        Monotonic clock, i.e., function returning number of elapsed seconds
        since some arbitrary point in time. Uses ``time.monotonic`` by default,
        if not available falls back to ``time.time``.

    Examples
    --------

    Following example demonstrates how to display information about progress,
    roughly every 5 seconds:

    >>> for steps in const_time_partition(20, 5.0):
    ...     for step in range(steps):
    ...         print('Working')
    ...         time.sleep(1.0)
    ...     print('Progress')
    ...

    """
    start = timer()
    left = iterations
    next = 1
    while left > 0:
        yield next
        elapsed = timer() - start
        left -= next
        done = iterations - left
        next = int(period * done / elapsed) if elapsed > 0 else 2 * next
        if next < 1:
            next = 1
        if next > left:
            next = left


class EmptyProgressBar:

    def __init__(self, *args, **kwargs):
        pass

    def update(self, n):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ProgressBar:

    FORMAT = 'iterations {self.iterations_done} ' \
             'of {self.iterations_total}, ' \
             'elapsed {self.elapsed}, ' \
             'remaining {self.remaining}'

    def __init__(self, steps, header='', refresh_seconds=0.5,
                 file=sys.stdout, timer=default_timer):
        self.format = header + self.FORMAT
        self.file = file
        self.isatty = file.isatty()
        self.timer = timer
        self.start_seconds = self.timer()
        self.last_seconds = self.start_seconds
        self.refresh_seconds = refresh_seconds
        self.iterations_done = 0
        self.iterations_total = steps
        self.previous_length = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.update(0, force=True)
        if self.isatty:
            self.file.write('\n')
            self.file.flush()

    @synchronized
    def update(self, steps, force=False):
        self.iterations_done += steps
        seconds = self.timer()
        if self.refresh_seconds <= seconds - self.last_seconds or force:
            self.last_seconds = seconds
            self.write(self.render())

    def render(self):
        return self.format.format(self=self)

    def write(self, line):
        if self.isatty:
            # 1. Move to the beginning of the line
            # 2. Overwrite previous content (necessary when new line is shorter)
            # 3. Move to the beginning again.
            n = self.previous_length
            self.file.write('\b' * n + ' ' * n + '\b' * n)
            self.file.write(line)
            self.previous_length = len(line)
        else:
            self.file.write(line)
            self.file.write('\n')
        self.file.flush()

    @property
    def iterations_remaining(self):
        return self.iterations_total - self.iterations_done

    @property
    def percentage(self):
        if self.iterations_total:
            return 100 * self.iterations_done / self.iterations_total
        else:
            return 100

    @property
    def elapsed(self):
        elapsed_seconds = self.last_seconds - self.start_seconds
        return timedelta(seconds=round(elapsed_seconds, 0))

    @property
    def time_per_iteration(self):
        elapsed_seconds = self.last_seconds - self.start_seconds
        return elapsed_seconds / self.iterations_done if self.iterations_done else float('Inf')

    @property
    def remaining(self):
        remaining_seconds = self.iterations_remaining * self.time_per_iteration
        if math.isinf(remaining_seconds):
            return timedelta.max
        else:
            return timedelta(seconds=round(remaining_seconds, 0))


def progress_bar_factory(enable, *args, **kwargs):
    type = ProgressBar if enable else EmptyProgressBar
    def factory(steps, *fargs, **fkwargs):
        all_args = fargs + args
        all_kwargs = dict(kwargs)
        all_kwargs.update(fkwargs)
        return type(steps, *all_args, **all_kwargs)
    return factory
