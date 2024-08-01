
import time


# ---------------------------------------------------------------------
def human_readable_timeslapse(sec_elapsed):
    """
    Helper method to convert time slapse to human readable format
    """
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return f"{h} hours, {m:>2} minutes, {s:>.2} seconds"
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
def func_timer(func):
    """
    A timer decorator.
    Usage:

    @timerfunc
    def long_runner():
        for x in range(5):
            sleep_time = random.choice(range(1, 5))
            time.sleep(sleep_time)
    """
    def function_timer(*args, **kwargs):
        """
        A nested function for timing other functions
        """
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        msg = "The runtime for {func} took {time} seconds to complete"
        print(msg.format(func=func.__name__, time=runtime))
        return value
    return function_timer


class ContextTimer:
    """

    Usage:
    if __name__ == "__main__":
        with ContextTimer('context1'):
            " Some code to time
            do_something1()
            do_something2()
    """

    def __init__(self, identifier='block'):
        self.start = time.time()
        self.identifier = identifier

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        runtime = end - self.start
        msg = f'The {self.identifier} took {runtime} seconds to complete'
        print(msg)
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
class ElapseTimer:

    def __init__(self):
        self.time_accumulate = 0

    def start(self):
        self.time_start = time.time()

    def update(self):
        now = time.time()
        time_elapse = now - self.time_start
        self.time_accumulate += time_elapse
        self.time_start = now
        return time_elapse

    def __str__(self):
        return human_readable_timeslapse(self.time_accumulate)
# ---------------------------------------------------------------------
