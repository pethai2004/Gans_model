import time

class Timer:
    def __init__(self):
        self.end = 0
        self.elapsed = 0
        self.elapsedH = 0

    def __enter__(self):
        self.start()

    def __exit__(self, *args):
      self.stop()

    def start(self):
        self.begin = time.time()
        return self

    def stop(self):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print('====> [{}] Time: {:7.3f}s or {}'.format(self.elapsed, time.strftime("%H:%M:%S", self.elapsedH)))