#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from typing import Optional


class ConstantSchedule:
    def __init__(self, val):
        self.val = val

    def __call__(self, steps=1):
        return self.val


class LinearSchedule:
    def __init__(self, start, end: Optional[int], steps: int):
        if end is None:
            end = start
            steps = 1

        self.current = start
        self.end = start if end is None else end

        self.inc = (end - start) / float(steps)

        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps: int = 1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val
