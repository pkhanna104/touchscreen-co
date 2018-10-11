from kivy.app import App
from kivy.core.window import Window
from kivy.core.audio import SoundLoader
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, ListProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.vector import Vector
from kivy.clock import Clock
from random import randint
from kivy.config import Config
import serial, time, pickle, datetime, winsound
import time
import numpy as np
import tables

class Data(tables.IsDescription):
    state = tables.StringCol(24)   # 24-character String
    time = tables.Float32Col()
    force = tables.Float32Col()
    beam = tables.Float32Col()
    start_led = tables.Float32Col()
    start_button = tables.Float32Col()

class R2Game(Widget):
    ITI_mean = 1.
    ITI_std = .2

    start_timeout = 10. 
    start_holdtime = .001

    grasp_timeout_time = 5000.
    grasp_holdtime = .001

    # Number of trials: 
    trial_counter = NumericProperty(0)
            
    def init(self, animal_names_dict=None, rew_in=None, rew_del=None,
        test=None, hold=None, autoquit=None, use_start=None, only_start=None):

        holdz = [0., .25, .5, .625, .75]
        for i, val in enumerate(hold['start_hold']):
            if val:
                self.start_hold = holdz[i]
        for i, val in enumerate(hold['grasp_hold']):
            if val:
                self.grasp_hold = holdz[i]

        small_rew_opts = [.1, .3, .5]
        for i, val in enumerate(rew_in['small_rew']):
            if val:
                small_rew = small_rew_opts[i]

        big_rew_opts = [.3, .5, .7]
        for i, val in enumerate(rew_in['big_rew']):
            if val:
                big_rew = big_rew_opts[i]

        if np.logical_or(rew_in['rew_start'], rew_in['rew_start_and_grasp']):
            self.reward_for_start = [True, small_rew]
        else:
            self.reward_for_start = [False, 0]

        if np.logical_or(rew_in['rew_grasp'], rew_in['rew_start_and_grasp']):
            self.reward_for_grasp = [True, big_rew]
        else:
            self.reward_for_grasp = [False, 0]

        if rew_in['snd_only']:
            self.reward_for_grasp = [True, 0.]
            self.skip_juice = True
        else:
            self.skip_juice = False

        for i, (nm, val) in enumerate(animal_names_dict.items()):
            if val:
                animal_name = nm

        try:
            pygame.mixer.init()    
        except:
            pass

        reward_delay_opts = [0., .4, .8, 1.2]
        for i, val in enumerate(rew_del['rew_del']):
            if val:
                self.reward_delay_time = reward_delay_opts[i]

        test_vals = [True, False, False]
        for i, val in enumerate(test['test']):
            if val:
                self.testing = test_vals[i]
        
        autoquit_trls = [25, 50, 10**10]
        for i, val in enumerate(autoquit['autoquit']):
            if val: 
                self.max_trials = autoquit_trls[i]

        start = [True, False]
        for i, val in enumerate(use_start['start']):
            if val:
                self.use_start = start[i]
        for i, val in enumerate(only_start['start']):
            if val:
                self.only_start = start[i]

        self.state = 'ITI'
        self.state_start = time.time()
        self.ITI = np.random.random()*self.ITI_std + self.ITI_mean

        self.start_led = 0
        self.button = 0

        # State transition matrix: 
        self.FSM = dict()
        self.FSM['ITI'] = dict(end_ITI='start_button', stop=None)
        self.FSM['start_button'] = dict(pushed_start='start_hold', start_button_timeout='ITI', stop=None)
        self.FSM['start_hold'] = dict(end_start_hold='grasp_trial_start', start_early_release = 'start_button', stop=None)
        self.FSM['grasp_trial_start'] = dict(clear_LED='grasp_hold', grasp_timeout='ITI', stop=None)
        self.FSM['grasp_hold'] = dict(end_grasp_hold='reward', drop='grasp', grasp_timeout='ITI', stop=None)
        self.FSM['grasp'] = dict(clear_LED='grasp_hold', grasp_timeout='ITI', stop=None) # state to indictate 'grasp' w/o resetting timer
        self.FSM['reward'] = dict(end_reward='ITI', stop=None)

        if not self.use_start:
            self.FSM['ITI'] = dict(end_ITI='grasp_trial_start', stop=None)

        if self.only_start:
            self.FSM = dict()
            self.FSM['ITI'] = dict(end_ITI='start_button', stop=None)
            self.FSM['start_button'] = dict(pushed_start='start_hold', start_button_timeout='ITI', stop=None)
            self.FSM['start_hold'] = dict(end_start_hold='reward', start_early_release = 'start_button', stop=None)
            self.FSM['reward'] = dict(end_reward='ITI', stop=None)

        try:
            self.reward_port = serial.Serial(port='COM6',
                baudrate=115200)
            self.reward_port.close()
            reward_fcn = True
        except:
            reward_fcn = False
            pass

        # save parameters: 
        d = dict(animal_name=animal_name,
            ITI_mean=self.ITI_mean, ITI_std = self.ITI_std, start_hold=self.start_hold,
            grasp_hold = self.grasp_hold, start_timeout = self.start_timeout, 
            grasp_timeout = self.grasp_timeout_time, big_rew=big_rew, 
            small_rew = small_rew, rew_for_start = self.reward_for_start[0], 
            reward_for_grasp = self.reward_for_grasp[0], skip_juice=self.skip_juice,
            rew_delay = self.reward_delay_time, use_start = self.use_start, 
            only_start = self.only_start, reward_fcn=reward_fcn)

        # Open task arduino
        self.task_ard = serial.Serial(port='COM12')

        if self.testing:
            pass
        else:
            import os
            path = os.getcwd()
            path = path.split('\\')
            path_data = [p for p in path if np.logical_and('Touchscreen' not in p, 'Targ' not in p)]
            p = ''
            for ip in path_data:
                p += ip+'/'
            p += 'data/'
            print ('')
            print ('')
            print('Data saving PATH: ', p)
            print ('')
            print ('')
            self.filename = p+ animal_name+'_Grasp_'+datetime.datetime.now().strftime('%Y%m%d_%H%M')

            pickle.dump(d, open(self.filename+'_params.pkl', 'wb'))
            self.h5file = tables.open_file(self.filename + '_data.hdf', mode='w', title = 'NHP data')
            self.h5_table = self.h5file.create_table('/', 'task', Data, '')
            self.h5_table_row = self.h5_table.row

            # Note in python 3 to open pkl files: 
            #with open('xxxx_params.pkl', 'rb') as f:
            #    data_params = pickle.load(f)

    def close_app(self):
        App.get_running_app().stop()
        Window.close()

    def update(self, ts):
        self.state_length = time.time() - self.state_start
        
        # Read from task arduino: 
        ser = self.task_ard.flushInput()
        _ = self.task_ard.readline()
        port_read = self.task_ard.readline()
        port_splits = port_read.decode('ascii').split('/t')
        #print(port_splits)
        print(self.state)
        if len(port_splits) != 4:
            ser = self.task_ard.flushInput()
            _ = self.task_ard.readline()
            port_read = self.task_ard.readline()
            port_splits = port_read.decode('ascii').split('/t')            
        self.force = int(port_splits[0])
        self.beam = int(port_splits[1])
        self.button = int(port_splits[2])
        
        # Run task update functions: 
        for f, (fcn_test_name, next_state) in enumerate(self.FSM[self.state].items()):
            kw = dict(ts=self.state_length)
            
            fcn_test = getattr(self, fcn_test_name)
            if fcn_test(**kw):
                # if stop: close the app
                if fcn_test_name == 'stop':
                    self.close_app()

                else:
                    # Run any 'end' fcns from prevoius state: 
                    end_state_fn_name = "_end_%s" % self.state
                    if hasattr(self, end_state_fn_name):
                        end_state_fn = getattr(self, end_state_fn_name)
                        end_state_fn()
                    self.prev_state = self.state
                    self.state = next_state
                    self.state_start = time.time()

                    # Run any starting functions: 
                    start_state_fn_name = "_start_%s" % self.state
                    if hasattr(self, start_state_fn_name):
                        start_state_fn = getattr(self, start_state_fn_name)
                        start_state_fn()
            
        if self.testing:
            pass
        else:
            if self.state == 'idle_exit':
                pass
            else:
                self.write_to_h5file()

    def write_to_h5file(self):
        self.h5_table_row['state']= self.state
        self.h5_table_row['time'] = time.time()
        self.h5_table_row['force'] = self.force
        self.h5_table_row['beam'] = self.beam
        self.h5_table_row.append()

    def stop(self, **kwargs):
        # If past number of max trials then auto-quit: 
        if np.logical_and(self.trial_counter >= self.max_trials, self.state == 'ITI'):
            self.idle = True
            return True
        else:
            return False

    def _start_ITI(self, **kwargs):
        self.ITI = np.random.random()*self.ITI_std + self.ITI_mean
        
    def end_ITI(self, **kwargs):
        return kwargs['ts'] > self.ITI

    def _start_start_button(self, **kwargs):
        self.task_ard.flushInput()
        self.task_ard.write('m'.encode()) #morning

    def pushed_start(self, **kwargs):
        return self.button

    def _end_start_button(self, **kwargs):
        self.task_ard.flushInput()
        self.task_ard.write('n'.encode()) #night

    def start_button_timeout(self, **kwargs):
        return kwargs['ts'] > self.start_timeout

    def end_start_hold(self, **kwargs):
        if kwargs['ts'] > self.start_hold:
            if self.reward_for_start:
                self._start_rew_start()
            return True
        else:
            return False

    def start_early_release(self, **kwargs):
        if self.button == 0:
            if kwargs['ts'] < self.start_hold:
                return True
        return False

    def clear_LED(self, **kwargs):
        return self.beam

    def grasp_timeout(self, **kwargs):
        return kwargs['ts'] > self.grasp_timeout_time

    def end_grasp_hold(self, **kwargs):
        return kwargs['ts'] > self.grasp_hold

    def drop(self, **kwargs):
        return # not beam clear

    def _start_reward(self, **kwargs):
        self.trial_counter += 1

        try:
            if self.reward_for_grasp[0]:
                #winsound.PlaySound('beep1.wav', winsound.SND_ASYNC)
                sound = SoundLoader.load('reward1.wav')
                sound.play()

                if not self.skip_juice:
                    self.reward_port.open()
                    rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.reward_for_grasp[1])+' sec\n']
                    self.reward_port.write(rew_str)
                    time.sleep(.5 + self.reward_delay_time)
                    run_str = [ord(r) for r in 'run\n']
                    self.reward_port.write(run_str)
                    self.reward_port.close()
        except:
            pass
        
    def _start_rew_start(self, **kwargs):
        try:
            if self.reward_for_start[0]:
                sound = SoundLoader.load('reward2.wav')
                sound.play()
                
                self.reward_port.open()
                rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.reward_for_start[1])+' sec\n']
                self.reward_port.write(rew_str)
                time.sleep(.5)
                run_str = [ord(r) for r in 'run\n']
                self.reward_port.write(run_str)
                self.reward_port.close()
        except:
            pass

    def end_reward(self, **kwargs):
        return True

class Manager(ScreenManager):
    pass

class R2GApp(App):
    def build(self, **kwargs):
        return Manager()
        
if __name__ == '__main__':
    R2GApp().run()