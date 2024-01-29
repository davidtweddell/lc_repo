import time

class Timer:
    # define a class to time code exeuction using 
    # context manager `with` statement
    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, *args):
        self.end = time.monotonic()
        self.interval = self.end - self.start
