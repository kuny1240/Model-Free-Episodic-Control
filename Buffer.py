import numpy as np
import random

class Buffer:

    def __init__(self,Buffer_Size):
        self.buffer = []
        self.cur_num = 0
        self.buffer_size = Buffer_Size

    def add(self,states,actions,rewards,done,next_states):

        if self.cur_num < self.buffer_size:
            self.buffer.append(np.hstack([states,actions,rewards,done,next_states]))
            self.cur_num += 1
        else:
            self.buffer.pop(0)
            self.buffer.append(np.hstack([states,actions,rewards,done,next_states]))


    def getBatch(self,batch_size):
        if self.cur_num < batch_size:
            return np.vstack(random.sample(self.buffer,self.cur_num))
        else:
            return np.vstack(random.sample(self.buffer,batch_size))

    def count(self):
        return  self.cur_num