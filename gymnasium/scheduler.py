import numpy as np


class ExponentialScheduler:
    def __init__(self, in_min, in_max, out_start, out_end, rate) -> None:
        self.in_min = in_min
        self.in_max = in_max
        self.out_start = out_start
        self.out_end = out_end
        self.rate = rate

    def step(self, progress):
        if progress < self.in_min:
            return self.out_start
        elif progress > self.in_max:
            return self.out_end
        else:
            ratio = (progress - self.in_min) / (self.in_max - self.in_min)
            gap = self.out_end / self.out_start
            return self.out_start * (gap**(ratio**self.rate))

    def __call__(self, progress) -> float:
        return self.step(progress)

    def get_caller(self, progress):
        return lambda x: self.step(progress)


class LinearScheduler:
    def __init__(self, in_min, in_max, out_min, out_max, step_size=-1) -> None:
        self.in_min = in_min
        self.in_max = in_max
        self.out_min = out_min
        self.out_max = out_max
        self.step_size = step_size

    def step(self, progress):
        if progress < self.in_min:
            return self.out_min
        elif progress > self.in_max:
            return self.out_max
        else:
            increment = (self.out_max - self.out_min) * \
                (progress - self.in_min) / (self.in_max - self.in_min)
            if self.step_size == -1:
                return self.out_min + increment
            return int(self.out_min + np.round(increment / self.step_size) * self.step_size)

    def __call__(self, progress) -> float:
        # return self.step(1 - progress_remaining)
        return self.step(progress)

    def get_caller(self, progress):
        return lambda x: self.step(progress)


class MultiStageScheduler:
    def __init__(self, schedulers) -> None:
        self.schedulers = schedulers

    def step(self, progress):
        for scheduler in self.schedulers:
            if progress < scheduler.in_max:
                return scheduler.step(progress)
        return self.schedulers[-1].step(progress)

    def __call__(self, progress) -> float:
        # return self.step(1 - progress_remaining)
        return self.step(progress)
