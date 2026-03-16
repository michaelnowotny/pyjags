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

"""Terminal progress bar for MCMC sampling iterations."""

__all__ = ["const_time_partition", "progress_bar_factory"]

import math
import sys
import threading
import time
from datetime import timedelta
from functools import wraps

default_timer = getattr(time, "monotonic", time.time)


def synchronized(func):
    """Decorator that serializes access to the wrapped function using a lock.

    Creates a ``threading.Lock`` per decorated function and acquires it before
    each invocation, ensuring that concurrent calls from multiple threads are
    executed one at a time.

    Parameters
    ----------
    func : callable
        The function to be synchronized.

    Returns
    -------
    callable
        A wrapper that calls *func* while holding the lock.
    """
    lock = threading.Lock()

    @wraps(func)
    def inner(*args, **kwargs):
        """Call the wrapped function while holding the lock."""
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
    iterations : int
        A non-negative integer specifying total number of iterations to
        execute.
    period : float
        A positive float number describing desired period between yields from
        generator.
    timer : callable, optional
        Monotonic clock, i.e., function returning number of elapsed seconds
        since some arbitrary point in time. Uses ``time.monotonic`` by default,
        if not available falls back to ``time.time``.

    Yields
    ------
    int
        The number of iterations to execute in the next sub-batch before the
        generator should be advanced again.

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
    """No-op progress bar that silently ignores all updates.

    Implements the same interface as `ProgressBar` so it can be used as a
    drop-in replacement when progress reporting is disabled.  All methods
    are no-ops.
    """

    def __init__(self, *args, **kwargs):
        """Create an empty progress bar, ignoring all arguments."""
        pass

    def update(self, n):
        """Accept and discard an iteration count update.

        Parameters
        ----------
        n : int
            Number of completed iterations (ignored).
        """
        pass

    def __enter__(self):
        """Enter the context manager and return self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager (no-op)."""
        pass


class ProgressBar:
    """Terminal progress bar for MCMC sampling iterations.

    Displays a single, periodically refreshed status line showing completed
    iterations, total iterations, elapsed time, and estimated remaining time.
    On interactive terminals (TTYs) the line is overwritten in place; on
    non-interactive streams each refresh appends a new line.

    The progress bar is intended to be used as a context manager so that a
    final update and trailing newline are written on exit.
    """

    FORMAT = (
        "iterations {self.iterations_done} "
        "of {self.iterations_total}, "
        "elapsed {self.elapsed}, "
        "remaining {self.remaining}"
    )
    """str : Default format template for the status line.

    The template is evaluated with ``self`` available for attribute
    interpolation.
    """

    def __init__(
        self,
        steps,
        header="",
        refresh_seconds=0.5,
        file=sys.stdout,
        timer=default_timer,
    ):
        """Initialize the progress bar.

        Parameters
        ----------
        steps : int
            Total number of iterations to track.
        header : str, optional
            Text prepended to the default format string (default ``""``).
        refresh_seconds : float, optional
            Minimum interval in seconds between visual refreshes
            (default ``0.5``).
        file : file-like, optional
            Writable stream for output (default ``sys.stdout``).
        timer : callable, optional
            Monotonic clock returning elapsed seconds.  Defaults to
            ``time.monotonic`` when available, otherwise ``time.time``.
        """
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
        """Enter the context manager and return self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, writing a final status line and newline."""
        self.update(0, force=True)
        if self.isatty:
            self.file.write("\n")
            self.file.flush()

    @synchronized
    def update(self, steps, force=False):
        """Record completed iterations and optionally refresh the display.

        The display is refreshed only when at least ``refresh_seconds`` have
        elapsed since the last refresh, unless *force* is ``True``.  Access
        is serialized across threads via the `synchronized` decorator.

        Parameters
        ----------
        steps : int
            Number of newly completed iterations to add to the running total.
        force : bool, optional
            If ``True``, refresh the display regardless of the elapsed time
            since the last refresh (default ``False``).
        """
        self.iterations_done += steps
        seconds = self.timer()
        if self.refresh_seconds <= seconds - self.last_seconds or force:
            self.last_seconds = seconds
            self.write(self.render())

    def render(self):
        """Render the current progress status into a string.

        Returns
        -------
        str
            The formatted status line produced by interpolating the progress
            bar's format template with the current state.
        """
        return self.format.format(self=self)

    def write(self, line):
        """Write a status line to the output stream.

        On TTY streams the previous line is overwritten in place.  On
        non-TTY streams a new line is appended instead.

        Parameters
        ----------
        line : str
            The text to write.
        """
        if self.isatty:
            # 1. Move to the beginning of the line
            # 2. Overwrite previous content (necessary when new line is shorter)
            # 3. Move to the beginning again.
            n = self.previous_length
            self.file.write("\b" * n + " " * n + "\b" * n)
            self.file.write(line)
            self.previous_length = len(line)
        else:
            self.file.write(line)
            self.file.write("\n")
        self.file.flush()

    @property
    def iterations_remaining(self):
        """Number of iterations not yet completed.

        Returns
        -------
        int
            ``iterations_total - iterations_done``.
        """
        return self.iterations_total - self.iterations_done

    @property
    def percentage(self):
        """Percentage of iterations completed.

        Returns
        -------
        float
            A value between 0 and 100.  Returns 100 when
            ``iterations_total`` is zero.
        """
        if self.iterations_total:
            return 100 * self.iterations_done / self.iterations_total
        else:
            return 100

    @property
    def elapsed(self):
        """Wall-clock time elapsed since the progress bar was created.

        Returns
        -------
        datetime.timedelta
            Elapsed duration rounded to the nearest second.
        """
        elapsed_seconds = self.last_seconds - self.start_seconds
        return timedelta(seconds=round(elapsed_seconds, 0))

    @property
    def time_per_iteration(self):
        """Average wall-clock time per completed iteration.

        Returns
        -------
        float
            Seconds per iteration, or ``float('Inf')`` if no iterations
            have been completed yet.
        """
        elapsed_seconds = self.last_seconds - self.start_seconds
        return (
            elapsed_seconds / self.iterations_done
            if self.iterations_done
            else float("Inf")
        )

    @property
    def remaining(self):
        """Estimated wall-clock time remaining until all iterations finish.

        Returns
        -------
        datetime.timedelta
            Estimated remaining duration rounded to the nearest second, or
            ``timedelta.max`` when no iterations have been completed yet.
        """
        remaining_seconds = self.iterations_remaining * self.time_per_iteration
        if math.isinf(remaining_seconds):
            return timedelta.max
        else:
            return timedelta(seconds=round(remaining_seconds, 0))


def progress_bar_factory(enable, *args, **kwargs):
    """Create a factory function that produces progress bars.

    Returns a callable that, when invoked with a step count (and optional
    extra arguments), constructs either a `ProgressBar` or an
    `EmptyProgressBar` depending on *enable*.  Any positional or keyword
    arguments passed here are forwarded to the progress bar constructor on
    every call, and can be overridden at call time.

    Parameters
    ----------
    enable : bool
        If ``True`` the factory produces `ProgressBar` instances; if
        ``False`` it produces `EmptyProgressBar` instances.
    *args
        Extra positional arguments forwarded to the progress bar constructor.
    **kwargs
        Extra keyword arguments forwarded to the progress bar constructor.

    Returns
    -------
    callable
        A function with signature ``factory(steps, *fargs, **fkwargs)``
        that returns a progress bar instance.
    """
    type = ProgressBar if enable else EmptyProgressBar

    def factory(steps, *fargs, **fkwargs):
        """Create a progress bar instance for the given number of steps."""
        all_args = fargs + args
        all_kwargs = dict(kwargs)
        all_kwargs.update(fkwargs)
        return type(steps, *all_args, **all_kwargs)

    return factory
