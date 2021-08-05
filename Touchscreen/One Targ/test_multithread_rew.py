import threading 
import time
import serial 

class RewThread(threading.Thread):
    def __init__(self, comport, rew_time):
        super(RewThread, self).__init__()
        
        self.comport = comport
        self.rew_time = rew_time

    def run(self): 
        #self.comport.open()
        rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.rew_time)+' sec\n']
        self.comport.write(rew_str)
        time.sleep(.25)
        run_str = [ord(r) for r in 'run\n']
        self.comport.write(run_str)
        #self.comport.close()
        print('rewarding\n')


reward_port = serial.Serial(port='COM4',
                baudrate=115200)
#reward_port.close()


t0 = time.time()
# Create new threads
t1 = time.time()
# Start new Threads
thread1 = RewThread(reward_port, .3)
thread1.start()
t2 = time.time()
thread1 = RewThread(reward_port, .3)
thread1.start()
t3 = time.time()
thread1 = RewThread(reward_port, .3)
thread1.start()
t4 = time.time()
