
import numpy


class BatchIdxIter:

    def __init__(self, batch_size, N):
        self.batch_size = batch_size
        self.N = N
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        
        if self.current >= self.N:
            raise StopIteration

        if self.current + self.batch_size < self.N:
            bs = self.batch_size
        else:
            bs = self.N - self.current
        _range = list(range(self.current, self.current + bs))

        self.current += bs
        return bs, _range



def print_progress_log(epoch: int, logs: dict, extra=None):
    console_print = f'\x1b[2K\rEpoch {epoch:3}:'
    console_print += ''.join(f" [{key}]{value:5.3f}" for key, value in logs.items())

    if extra is not None:
        if isinstance(extra, str):
            console_print += '| ' + extra
        elif isinstance(extra, list) and len(extra) > 0:
            console_print += '  | ' + "".join(f' {info}' for info in extra)

    print(console_print)
