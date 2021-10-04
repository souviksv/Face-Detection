import time
import collections

class FPS:
    def __init__(self,averageof=50):
        self.frametimestamps = collections.deque(maxlen=averageof)

    def __call__(self):
        self.frametimestamps.append(time.time())
        
        if(len(self.frametimestamps) > 1):
            return int(len(self.frametimestamps)/(self.frametimestamps[-1]-self.frametimestamps[0]))
        else:
            return 0