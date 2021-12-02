from kivy.app import App
from kivy.core.window import Window
from kivy.core.audio import SoundLoader
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, ListProperty, StringProperty, BooleanProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.vector import Vector
from kivy.clock import Clock
from random import randint
from kivy.config import Config
import serial, time, pickle, datetime, winsound, struct
import time
import numpy as np
import tables
import subprocess, signal

Config.set('graphics', 'resizable', False)
fixed_window_size = (1800, 1000)
Config.set('graphics', 'width', str(fixed_window_size[0]))
Config.set('graphics', 'height', str(fixed_window_size[1]))

import threading

### Load last parameters ###
import os 

path = os.getcwd()
path_data = path.split('\\')
path_root = ''
for ip in path_data: 
    path_root += ip + '/'
if os.path.exists(path_root + 'last_params.pkl'): 
    with open(path_root + 'last_params.pkl', 'rb') as f:
        data_params = pickle.load(f)
else: 
    data_params = {}

class RewThread(threading.Thread):
    def __init__(self, comport, rew_time, juicer):
        super(RewThread, self).__init__()

        self.comport = comport
        self.rew_time = rew_time
        self.juicer = juicer

    def run(self):

        if self.juicer == 'yellow': 
            rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.rew_time)+' sec\n']
            
            try: # commented out comport open/close -- was giving errors in spinning pal
                self.comport.open()
                self.comport.write(rew_str)
                time.sleep(.25)
                run_str = [ord(r) for r in 'run\n']
                self.comport.write(run_str)
                self.comport.close()
            except:
                pass            
        

        elif self.juicer == 'red':
            try:
                volume2dispense = self.rew_time * 50 / 60 #mL/min x 1 min / 60 sec --> sec x mL/sec 
                self.comport.write(b"VOL %.1f \r"%volume2dispense)
                time.sleep(.25)
                self.comport.write(b"RUN\r")
            except:
                pass
class Data(tables.IsDescription):
    state = tables.StringCol(24)   # 24-character String
    time = tables.Float32Col()
    time_abs = tables.Float32Col()
    fsr1 = tables.Float32Col()
    fsr2 = tables.Float32Col()
    wheel = tables.Float32Col()
    beam = tables.Float32Col()
    button = tables.Float32Col()
    start_led = tables.Float32Col()
    start_button = tables.Float32Col()
    trial_type = tables.StringCol(24)
    trial_pos = tables.Float32Col()
    door_pos = tables.Float32Col()
    door_state = tables.StringCol(7)

class R2Game(Widget):
    pre_start_vid_ts = 0.1

    ITI_mean = 1.
    ITI_std = .2

    start_timeout = 5000. 
    start_holdtime = .001

    grasp_timeout_time = 5000.
    grasp_holdtime = .001

    fsr_threshold = -1

    # Number of trials: 
    trial_counter = NumericProperty(0)
    t0 = time.time()

    big_reward_cnt = NumericProperty(0)
    small_reward_cnt = NumericProperty(0)
    tried = NumericProperty(0)
    what_notch_going_to = NumericProperty(0)
    what_notch_currently_at = NumericProperty(0)
    speed = NumericProperty(0)
    hall_cnt = NumericProperty(0)
    end_cnt = NumericProperty(0)

    # Set relevant params text: 
    grasp_rew_txt = StringProperty('')
    grasp_rew_param = StringProperty('')

    button_rew_txt = StringProperty('')
    button_rew_param = StringProperty('')

    grasp_hold_param = StringProperty('')
    grasp_hold_txt = StringProperty('')   
    button_hold_txt = StringProperty('')
    button_hold_param = StringProperty('')

    n_trials_txt = StringProperty('')
    n_trials_param = StringProperty('')

    def init(self, animal_names_dict=None, rew_in=None, rew_var=None,
        test=None, hold=None, autoquit=None,
        grasp_to=None, use_cap=None, tsk_opt=None, trials_active=None,
        juicer=None):

        self.h5_table_row_cnt = 0
        self.idle = False

        cap = [True, False]
        for i, val in enumerate(use_cap['use_cap']):
            if val:
                self.use_cap_not_button = cap[i]

        button_holdz = [0., 0.1, 0.2, 0.3, 0.4]
        grasp_holdz = [0., 0.05, 0.1, 0.15, .2, .25, .35, .50, '.09-.12', '.11-.15']

        for i, val in enumerate(hold['start_hold']):
            if val:
                if type(button_holdz[i]) is str:
                    self.start_hold_type = button_holdz[i]
                    self.start_hold = 0.
                else:
                    self.start_hold_type = 0
                    self.start_hold = button_holdz[i]

        for i, val in enumerate(hold['grasp_hold']):
            if val:
                if type(grasp_holdz[i]) is str:
                    self.grasp_hold_type = grasp_holdz[i]
                    self.grasp_hold = 0.
                else:
                    self.grasp_hold_type = grasp_holdz[i]
                    self.grasp_hold = grasp_holdz[i]

        small_rew_opts = [.1, .3, .5]
        for i, val in enumerate(rew_in['small_rew']):
            if val:
                small_rew = small_rew_opts[i]

        big_rew_opts = [.3, .5, .7]
        for i, val in enumerate(rew_in['big_rew']):
            if val:
                big_rew = big_rew_opts[i]

        ### We always want sound for start reward ###
        if np.logical_or(rew_in['rew_start'], rew_in['rew_start_and_grasp']):
            self.reward_for_start = [True, small_rew]
        else:
            self.reward_for_start = [True, 0]

        if np.logical_or(rew_in['rew_grasp'], rew_in['rew_start_and_grasp']):
            self.reward_for_grasp = [True, big_rew]
        else:
            self.reward_for_grasp = [True, 0]

        if rew_in['snd_only']:
            self.reward_for_grasp = [True, 0.]
            self.reward_for_start = [True, 0.]
            self.skip_juice = True
        else:
            self.skip_juice = False

        print('reward for start')
        print(self.reward_for_start)
        print('reward for grasp')
        print(self.reward_for_grasp)
        for i, (nm, val) in enumerate(animal_names_dict.items()):
            if val:
                animal_name = nm

        try:
            pygame.mixer.init()    
        except:
            pass

        reward_var_opt = [1.0, .5, .33]
        for i, val in enumerate(rew_var['rew_var']):
            if val:
                self.percent_of_trials_rewarded = reward_var_opt[i]
                if self.percent_of_trials_rewarded == 0.33:
                    self.percent_of_trials_doubled = 0.1
                else:
                    self.percent_of_trials_doubled = 0.0
        
        self.reward_generator = self.gen_rewards(self.percent_of_trials_rewarded, self.percent_of_trials_doubled,
            self.reward_for_grasp)

        self.reward_delay_time = 0. #reward_delay_opts[i]

        test_vals = [True, False, False]
        for i, val in enumerate(test['test']):
            if val:
                self.testing = test_vals[i]
        
        autoquit_trls = [25, 50, 10**10]
        for i, val in enumerate(autoquit['autoquit']):
            if val: 
                self.max_trials = autoquit_trls[i]

        self.use_start = True
        self.only_start = False

        grasp_tos = [5., 10., 20., 30., 999999999.]

        for i, val in enumerate(grasp_to['gto']):
            if val:
                self.grasp_timeout_time = grasp_tos[i]

        # Preload reward buttons: 
        self.reward1 = SoundLoader.load('reward1.wav')
        self.reward2 = SoundLoader.load('reward2.wav')
        self.reward_started = False

        self.state = 'ITI'
        self.state_start = time.time()
        self.ITI = np.random.random()*self.ITI_std + self.ITI_mean

        self.start_led = 0
        self.button = 0
        self.cap_button = 0.
        self.door_state = 'closed'

        task_opt = ['grip', 'button', 'both']
        for i, val in enumerate(tsk_opt['tsk_opt']):
            if val: 
                self.task_opt = task_opt[i]

        ### Trials to include ###
        trials_active_list = ['power_1', 'tripod_1', 'pinch_1', 'tiny_1', 'pinch_3']
        trials_position_list = [2, 6, 10, 8, 10]


        ###  juicer option ### 
        juicer_opts = ['yellow', 'red']
        for i, val in enumerate(juicer['juicer']): 
            if val: 
                self.juicer = juicer_opts[i]


        self.trial_num = 0; 
        self.trials_list_valid = []
        self.trial_labels_active = []
        for i, val in enumerate(trials_active['trials']):
            if val: 
                ### label 
                self.trial_labels_active.append(trials_active_list[i])
                ### how many of this trial type should we add? 
                trial_type, num = trials_active_list[i].split('_')
                num = int(num)
                for _ in range(num): 
                    self.trials_list_valid.append([trial_type, trials_position_list[i]])

        #### Generate trials list ####
        self.get_trials_order()
        self.current_trial = self.generated_trials[0]

        # State transition matrix: 
        self.FSM = dict()
        self.FSM['ITI'] = dict(end_ITI='vid_trig', stop=None)
        self.FSM['vid_trig'] = dict(end_vid_trig='start_button')
        self.FSM['start_button'] = dict(pushed_start='start_hold', start_button_timeout='ITI', opened_door='ITI', stop=None)
        self.FSM['start_hold'] = dict(end_start_hold='grasp_trial_start', start_early_release = 'start_button', stop=None)
        self.FSM['grasp_trial_start'] = dict(door_opened='grasp', stop=None)
        self.FSM['grasp'] = dict(clear_LED='grasp_hold', grasp_timeout='prep_next_trial', stop=None) # state to indictate 'grasp' w/o resetting timer
        self.FSM['grasp_hold'] = dict(end_grasp_hold='reward', drop='grasp', grasp_timeout='prep_next_trial', stop=None)
        self.FSM['reward'] = dict(end_reward='prep_next_trial', stop=None)
        self.FSM['prep_next_trial'] = dict(start_next_trial='ITI')
        self.FSM['idle_exit'] = dict(stop=None)

        if self.task_opt == 'both':
            pass

        elif self.task_opt == 'grip':
            self.FSM['ITI'] = dict(end_ITI='grasp_trial_start', stop=None)

        elif self.task_opt == 'button':
            self.FSM = dict()
            self.FSM['ITI'] = dict(end_ITI='start_button', stop=None)
            self.FSM['start_button'] = dict(pushed_start='start_hold', start_button_timeout='ITI', stop=None)
            self.FSM['start_hold'] = dict(end_start_hold='reward', start_early_release = 'start_button', stop=None)
            self.FSM['reward'] = dict(end_reward='ITI', stop=None)

        try:
            if self.juicer == 'yellow':
                self.reward_port = serial.Serial(port='COM5',
                    baudrate=115200)
                reward_fcn = True
                self.reward_port.close()
            
            elif self.juicer == 'red': 
                prolific_com = None
                import serial.tools.list_ports
                coms = serial.tools.list_ports.comports()
                for c in coms: 
                    if 'Prolific USB-to-Serial' in c.description:
                        prolific_com_end = c.description.split('(')
                        prolific_com_beg = prolific_com_end[1].split(')')
                        prolific_com = prolific_com_beg[0]
                    
                self.reward_port = serial.Serial(port=prolific_com, 
                    baudrate=19200)
                reward_fcn = True
                
                ### setup the flow rate
                time.sleep(.5) 
                ### set volume value and units and rate units
                self.reward_port.write(b"VOL 0.5\r")
                self.reward_port.write(b"VOL ML\r")
                self.reward_port.write(b"RAT 50MM\r") # 50 ml / min

        except:
            self.reward_port = None
            reward_fcn = False
            pass

        try:
            self.dio_port = serial.Serial(port='COM6', baudrate=115200)
            time.sleep(4.)
        except:
            pass

        try:
            self.cam_trig_port = serial.Serial(port='COM11', baudrate=9600)
            time.sleep(3.)
            # Say hello: 
            self.cam_trig_port.write('a'.encode())

            # Pause
            time.sleep(1.)

            # Start cams @ 50 Hz
            self.cam_trig_port.write('1'.encode())
        except:
            pass

        if self.use_cap_not_button:
            self.cap_ard = serial.Serial(port=None, baudrate=9600)

        # save parameters: 
        d = dict(animal_name=animal_name,
            ITI_mean=self.ITI_mean, ITI_std = self.ITI_std, start_timeout = self.start_timeout, rew_delay = self.reward_delay_time,
            reward_fcn=reward_fcn, use_cap=self.use_cap_not_button,
            start_hold=self.start_hold,
            grasp_hold = self.grasp_hold_type, 
            grasp_timeout = self.grasp_timeout_time, 
            big_rew=big_rew, 
            small_rew = small_rew, 
            rew_manual = rew_in['rew_manual'],
            rew_start = rew_in['rew_start'], 
            rew_start_and_grasp = rew_in['rew_start_and_grasp'],
            rew_grasp = rew_in['rew_grasp'],
            snd_only = rew_in['snd_only'], 
            task_opt = self.task_opt, 
            rew_all = rew_var['rew_var'][0], 
            rew_50 = rew_var['rew_var'][1], 
            rew_30 = rew_var['rew_var'][2], 
            trls_25 = autoquit['autoquit'][0], 
            trls_50 = autoquit['autoquit'][1], 
            trls_inf = autoquit['autoquit'][2], 
            trials_active = self.trials_list_valid, 
            trials_labels = self.trial_labels_active,
            juicer = self.juicer,
            rew_for_start = self.reward_for_start[0], 
            reward_for_grasp = self.reward_for_grasp[0], 
            skip_juice=self.skip_juice, 
            use_start = self.use_start, 
            only_start = self.only_start)

        ### save now 
        import os
        path_root = os.getcwd()
        path_root = path_root.split('\\')
        path_ = ''
        for p in path_root: path_ += p+'/'
        pickle.dump(d, open(path_ + 'last_params.pkl', 'wb'))

        ## Open task arduino - IR sensor, button, wheel position ### 
        self.task_ard = serial.Serial('COM10', baudrate=115200)
        self.going_to_targ = 0; 
        self.abortclose = 0; 
        self.try_to_close = False; 

        baseline_values = []
        
        ### Get baseline FSR data 
        for _ in range(100): 
            
            ### read data from FSR 
            # Read from task arduino: 
            ser = self.task_ard.flushInput()
            _ = self.task_ard.readline()
            port_read = self.task_ard.readline()
            port_splits = port_read.decode('ascii').split('\t')

            if len(port_splits) != 5:
                ser = self.task_ard.flushInput()
                _ = self.task_ard.readline()
                port_read = self.task_ard.readline()
                port_splits = port_read.decode('ascii').split('\t')  

        ### Beam / FSR 1 / FSR 2 / current count position           
        fsr1 = int(port_splits[1])
        fsr2 = int(port_splits[2])
        baseline_values.append(fsr1 + fsr2)
        time.sleep(.005)

        ### FSR threshold 
        self.fsr_threshold = 1.5*np.max(np.hstack((baseline_values)))
        if self.fsr_threshold < 30: 
            self.fsr_threshold += 25

        ### Close door 
        self.task_ard.write('c'.encode())

        time.sleep(2.)

        ### Go to next trial
        self.what_notch_going_to = self.current_trial[1] 
        word = b'd'+struct.pack('<H', self.current_trial[1]) 
        self.task_ard.write(word)

        d['fsr_threshold'] = self.fsr_threshold
        d['trials'] = self.generated_trials

        if self.testing:
            pass
        else:
            # import os
            # path = os.getcwd()
            # path = path.split('\\')
            # path_data = [p for p in path if np.logical_and('Touchscreen' not in p, 'Targ' not in p)]
            # path_root = ''
            # for ip in path_data:
            #     path_root += ip+'/'
            # p = path_root+'data/'

            path = os.getcwd()
            path = path.split('\\')
            print(path)
            for p in path:
                if p == 'BasalGangulia':
                    laptop_name = 'BasalGangulia'
                elif p == 'Ganguly':
                    laptop_name = 'Ganguly'
                elif p == 'stim':
                    laptop_name = 'stim'
                elif p == 'cortex':
                    laptop_name = 'cortex'

            p = "C:/Users/%s/Box/Data/NHP_BehavioralData/"%laptop_name

            # Check if this directory exists: 
            if os.path.exists(p):
                p = p + "spal/"
                print('path: %s'%p)
            else:
                p = path_root+ 'data_tmp_'+datetime.datetime.now().strftime('%Y%m%d_%H%M')+'/'
                if os.path.exists(p):
                    pass
                else:
                    os.mkdir(p)
                    print('Making temp directory: ', p)
                    
            print ('')
            print ('')
            print('Data saving PATH: ', p)
            print ('')
            print ('')

            self.filename = p+ animal_name+'_Grasp_'+datetime.datetime.now().strftime('%Y%m%d_%H%M')
            pickle.dump(d, open(self.filename+'_params.pkl', 'wb'))

            ## Save as 'last params'
            path_root = os.getcwd()
            pickle.dump(d, open(os.path.join(path_root, 'last_params.pkl'), 'wb'))

            self.h5file = tables.open_file(self.filename + '_data.hdf', mode='w', title = 'NHP data')
            self.h5_table = self.h5file.create_table('/', 'task', Data, '')
            self.h5_table_row = self.h5_table.row

            # Get the task to start the accelerometer process
            # Start the accelerometer: 
            #self.acc_process = subprocess.Popen(['python run_acc.py', p + animal_name + '_Grasp'])

            # Note in python 3 to open pkl files: 
            #with open('xxxx_params.pkl', 'rb') as f:
            #    data_params = pickle.load(f)

    def get_trials_order(self): 
        self.generated_trials = []
        for i in range(1000): 
            ix = np.random.permutation(len(self.trials_list_valid))
            for j in ix: 
                self.generated_trials.append(self.trials_list_valid[j])
        ### Now expand out 

    def gen_rewards(self, perc_trials_rew, perc_trials_2x, reward_for_grasp):
        mini_block = int(2*(np.round(1./self.percent_of_trials_rewarded)))
        rew = []
        mini_block_count = 0
        for i in range(500):
            mini_block_array = np.zeros((mini_block))
            ix = np.random.permutation(mini_block)
            mini_block_array[ix[:2]] = reward_for_grasp[1]
            mini_block_count += mini_block
            if (perc_trials_2x * perc_trials_rew) > 0:
                if mini_block_count > int(1./(perc_trials_2x*perc_trials_rew)):
                    mini_block_array[ix[0]] = 2*reward_for_grasp[1]
                    mini_block_count = 0
            rew.append(mini_block_array)
        return np.hstack((rew))

    def close_app(self):
        try:
            self.cam_trig_port.write('0'.encode())
        except:
            pass

        # Stop the accelerometer: 
        #self.acc_process.send_signal(signal.CTRL_C_EVENT)

        # Make sure motor goes to sleep at night ;)  
        self.task_ard.flushInput()

        # move to position zero 
        word = b'd'+struct.pack('<H', 0) 
        self.task_ard.write(word)
        going_to_targ = 1 
        while going_to_targ: 
            _ = self.task_ard.readline()
            port_read = self.task_ard.readline()
            port_splits = port_read.decode('ascii').split('\t')

            if len(port_splits) != 10:
                ser = self.task_ard.flushInput()
                _ = self.task_ard.readline()
                port_read = self.task_ard.readline()
                port_splits = port_read.decode('ascii').split('\t')  
            
            going_to_targ = int(port_splits[4])
            time.sleep(.01) # wait 10 ms; 

        ### turn off solenoid 
        self.task_ard.write('n'.encode()) 

        ### close teh door 
        self.task_ard.write('c'.encode()) 
        

        if self.idle:
            self.state = 'idle_exit'
            self.trial_counter = -1

            # Set relevant params text: 
            self.grasp_rew_txt = 'Grasp Rew Time: '
            self.grasp_rew_param = str(self.reward_for_grasp[1])

            self.button_rew_txt = 'Button Rew Time: '
            self.button_rew_param = str(self.reward_for_start[1])


            self.grasp_hold_txt = 'Grasp Hold Time: '
            if type(self.grasp_hold_type) is str:
                self.grasp_hold_param = self.grasp_hold_type
            else:
                self.grasp_hold_param = str(self.grasp_hold)
                
            self.button_hold_txt = 'Button Hold Time: '
            if type(self.start_hold_type) is str:
                self.button_hold_param = self.start_hold_type
            else:
                self.button_hold_param = str(self.start_hold)

            self.n_trials_txt = '# Trials: '
            self.n_trials_param = str(self.big_reward_cnt)
        else:
            App.get_running_app().stop()
            Window.close()

    def quit_from_app(self):
        # If second click: 

        if self.idle:
            self.idle = False

        # If first click: 
        else:
            self.idle = True

        self.close_app()

    def update(self, ts):
        print(self.state, self.going_to_targ, self.door_state, self.try_to_close, self.abortclose)
        self.state_length = time.time() - self.state_start
        
        # Read from task arduino: 
        ser = self.task_ard.flushInput()
        _ = self.task_ard.readline()
        port_read = self.task_ard.readline()
        port_splits = port_read.decode('ascii').split('\t')

        if len(port_splits) != 10:
            ser = self.task_ard.flushInput()
            _ = self.task_ard.readline()
            port_read = self.task_ard.readline()
            port_splits = port_read.decode('ascii').split('\t')  

        ### Beam / FSR 1 / FSR 2 / current count position           
        self.beam = int(port_splits[0])
        self.fsr1 = int(port_splits[1])
        self.fsr2 = int(port_splits[2])
        self.wheel_pos = int(port_splits[3])
        self.what_notch_currently_at = self.wheel_pos
        self.going_to_targ = int(port_splits[4])
        self.speed = float(port_splits[5])
        self.hall_cnt = int(port_splits[6])
        self.end_cnt = int(port_splits[7])
        self.door_pos = int(port_splits[8])
        self.abortclose = int(port_splits[9])
        if self.door_pos < 100:
            self.door_state = 'open'
        elif self.door_pos > 950: 
            self.door_state = 'closed'

        ### Buttons #####
        if self.fsr1 + self.fsr2 > self.fsr_threshold: 
            self.button = True
        else:
            self.button = False

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
        self.h5_table_row['time'] = time.time() - self.t0
        self.h5_table_row['time_abs'] = time.time()
        self.h5_table_row['beam'] = self.beam
        self.h5_table_row['fsr1'] = self.fsr1
        self.h5_table_row['fsr2'] = self.fsr2
        self.h5_table_row['button'] = self.button
        self.h5_table_row['wheel'] = self.wheel_pos
        self.h5_table_row['trial_type'] = self.current_trial[0]
        self.h5_table_row['trial_pos'] = self.current_trial[1]
        self.h5_table_row['door_pos'] = self.door_pos
        self.h5_table_row['door_state'] = self.door_state
        self.h5_table_row.append()

        # Write DIO 
        try:
            self.write_row_to_dio()
        except:
            pass
            
        # Upgrade table row: 
        self.h5_table_row_cnt += 1

    def write_row_to_dio(self):
        ### FROM TDT TABLE, 5 is GND, BYTE A #
        row_to_write = self.h5_table_row_cnt % 256

        ### write to arduino: 
        word_str = b'd' + struct.pack('<H', int(row_to_write))
        self.dio_port.write(word_str)

    def stop(self, **kwargs):
        # If past number of max trials then auto-quit: 
        if np.logical_and(self.trial_counter >= self.max_trials, self.state == 'ITI'):
            self.idle = True
            return True
        else:
            return False

    def _start_ITI(self, **kwargs):
        # Stop video
        t0 = time.time()
        try:
            self.cam_trig_port.write('0'.encode())
        except:
            pass
        self.ITI = np.random.random()*self.ITI_std + self.ITI_mean
        
        if type(self.start_hold_type) is str:
            cht_min, cht_max = self.start_hold_type.split('-')
            self.start_hold = ((float(cht_max) - float(cht_min)) * np.random.random()) + float(cht_min)

        if type(self.grasp_hold_type) is str:
            tht_min, tht_max = self.grasp_hold_type.split('-')
            self.grasp_hold = ((float(tht_max) - float(tht_min)) * np.random.random()) + float(tht_min) 

        #### flag to make sure that while reward is blocking loops, that 'drop' doesn't get triggered ###
        self.reward_started = False

        #### Get the current trial 
        self.trial_num += 1
        self.current_trial = self.generated_trials[self.trial_num]

    def opened_door(self, **kwargs): 
        if self.door_pos < 950: 
            ### close the door and return true 
            self.task_ard.write('c'.encode())
            return True 
        else:
            return False

    def _start_grasp_trial_start(self, **kwargs):
        ### open the door
        self.task_ard.write('o'.encode())

    def _start_grasp(self, **kwargs):
        self.start_grasp = time.time()

    def door_opened(self, **kwargs): 
        ### Is the wheel in the right spot ?? 
        if self.going_to_targ == 0 and self.door_state == 'open': 
            return True

    def end_ITI(self, **kwargs):
        ''' Only end the ITI if we've finished getting to the rest state'''
        if self.going_to_targ == 0: 
            if self.door_state == 'closed': 
                return kwargs['ts'] > self.ITI
            else:
                return False 
        else:
            return False
        
    def _start_vid_trig(self, **kwargs):
        try:
            self.cam_trig_port.write('1'.encode())
        except:
            pass
    
    def end_vid_trig(self, **kwargs):
        return kwargs['ts'] > self.pre_start_vid_ts

    def _start_start_button(self, **kwargs):
        pass

    def pushed_start(self, **kwargs):
        if self.use_cap_not_button:
            return self.cap_button
        else:
            return self.button

    def start_button_timeout(self, **kwargs):
        return kwargs['ts'] > self.start_timeout

    def end_start_hold(self, **kwargs):
        if kwargs['ts'] > self.start_hold:
            self._start_rew_start()
            return True
        else:
            return False

    def start_early_release(self, **kwargs):
        but = self.button
        if self.use_cap_not_button:
            but = self.cap_button
        if but == 0:
            if kwargs['ts'] < self.start_hold:
                return True
        return False

    def clear_LED(self, **kwargs):
        return self.beam

    def grasp_timeout(self, **kwargs):
        if (time.time() - self.start_grasp) > self.grasp_timeout_time: 
            self.try_to_close = False 
            return True 
        else: 
            return False

    def end_grasp_hold(self, **kwargs):
        return kwargs['ts'] > self.grasp_hold

    def drop(self, **kwargs):
        if self.reward_started:
            return False
        else:
            if self.beam == 0:
                return True
            else:
                return False

    def _start_reward(self, **kwargs):
        self.reward_started = True
        if self.task_opt == 'button':
            pass 
        else:
            if self.reward_for_grasp[0]:
                self.reward1.play()
                if self.reward_for_grasp[1] > 0:
                    thread1 = RewThread(self.reward_port, self.reward_generator[self.trial_counter],
                        self.juicer)
                    thread1.run()

        self.trial_counter += 1
        self.big_reward_cnt += 1
        self.try_to_close = False
        
    def _start_rew_start(self, **kwargs):
        self.small_reward_cnt += 1
        if self.reward_for_start[0]:
            self.reward2.play()
            
            if self.reward_for_start[1] > 0.:
                thread1 = RewThread(self.reward_port, self.reward_for_start[1], self.juicer)
                thread1.run()

    def end_reward(self, **kwargs):
        return True 

    def start_next_trial(self, **kwargs): 
        
        ready = False
        if self.try_to_close: 
            ## still trying to close door so don't start on the wheel yet 
            if self.door_state == 'open': 
                pass 

            elif self.door_state == 'closed': 
                if self.waiting_on_wheel: 
                    if self.going_to_targ: 
                        pass
                    else: 
                        ready = True 
                else:
                    ## try to move wheel to next position 
                    next_trl_rest_pos = self.generated_trials[self.trial_num+1][1]
                    self.what_notch_going_to = next_trl_rest_pos
                    word = b'd'+struct.pack('<H', next_trl_rest_pos) 
                    self.task_ard.write(word)
                    self.waiting_on_wheel = True

        else: 
            self.waiting_on_wheel = False # not yet waiting on the wheel 
            ### if the beam indicates dropped griopper 
            if self.beam == 0: 

                ## Try to close the door 
                self.task_ard.flushInput()
                self.task_ard.write('c'.encode())
                self.try_to_close = True 

        return ready

class Manager(ScreenManager):
    ### Reward setup ### 

    if len(data_params.keys()) > 0: 
        rew_manual = BooleanProperty(data_params['rew_manual'])
        rew_start = BooleanProperty(data_params['rew_start'])
        rew_start_and_grasp = BooleanProperty(data_params['rew_start_and_grasp'])
        rew_grasp = BooleanProperty(data_params['rew_grasp'])
        rew_snd = BooleanProperty(data_params['snd_only'])

        # if data_params['rew_manual']: 
        #     rew_manual = BooleanProperty(True)
        # elif data_params['rew_start']: 
        #     rew_start = BooleanProperty(True)
        # elif data_params['rew_start_and_grasp']: 
        #     rew_start_and_grasp = BooleanProperty(True)
        # elif data_params['rew_grasp']: 
        #     rew_grasp = BooleanProperty(True)
        # elif data_params['snd_only']: 
        #     rew_snd = BooleanProperty(True) 

        only_gripper = BooleanProperty(data_params['task_opt'] == 'grip')
        only_button = BooleanProperty(data_params['task_opt'] == 'button')
        button_grip = BooleanProperty(data_params['task_opt'] == 'both')

        # if data_params['task_opt'] == 'grip': 
        #     only_gripper = BooleanProperty(True)
        # elif data_params['task_opt'] == 'button': 
        #     only_button = BooleanProperty(True)
        # elif data_params['task_opt'] == 'both': 
        #     button_grip = BooleanProperty(True)

        monk_haribo = BooleanProperty(data_params['animal_name'] == 'haribo')
        monk_butters = BooleanProperty(data_params['animal_name'] == 'butters')
        monk_nike = BooleanProperty(data_params['animal_name'] == 'nike')
        monk_fifi = BooleanProperty(data_params['animal_name'] == 'fifi')
        monk_test = BooleanProperty(data_params['animal_name'] == 'testing')

        small_rew_1 = BooleanProperty(data_params['small_rew'] == 0.1)
        small_rew_3 = BooleanProperty(data_params['small_rew'] == 0.3)
        small_rew_5 = BooleanProperty(data_params['small_rew'] == 0.5)

        # if data_params['small_rew'] == 0.1: 
        #     small_rew_1 = BooleanProperty(True)
        # elif data_params['small_rew'] == 0.3: 
        #     small_rew_3 = BooleanProperty(True)
        # elif data_params['small_rew'] == 0.5: 
        #     small_rew_5 = BooleanProperty(True)

        big_rew_3 = BooleanProperty(data_params['big_rew'] == 0.3)
        big_rew_5 = BooleanProperty(data_params['big_rew'] == 0.5)
        big_rew_7 = BooleanProperty(data_params['big_rew'] == 0.7)

        # if data_params['big_rew'] == 0.3: 
        #     big_rew_3 = BooleanProperty(True)
        # elif data_params['big_rew'] == 0.5: 
        #     big_rew_5 = BooleanProperty(True)
        # elif data_params['big_rew'] == 0.7: 
        #     big_rew_7 = BooleanProperty(True)

        juicer_y = BooleanProperty(data_params['juicer'] ==  'yellow')
        juicer_r = BooleanProperty(data_params['juicer'] == 'red')

        # if data_params['juicer'] ==  'yellow': 
        #     juicer_y = BooleanProperty(True)
        # elif data_params['juicer'] == 'red': 
        #     juicer_r = BooleanProperty(True)

        rew_all = BooleanProperty(data_params['rew_all'])
        rew_50 = BooleanProperty(data_params['rew_50'])
        rew_30 = BooleanProperty(data_params['rew_30'])

        button_0 = BooleanProperty(data_params['start_hold'] == 0.)
        button_1 = BooleanProperty(data_params['start_hold'] == 0.1)
        button_2 = BooleanProperty(data_params['start_hold'] == 0.2)
        button_3 = BooleanProperty(data_params['start_hold'] == 0.3)
        button_4 = BooleanProperty(data_params['start_hold'] == 0.4)

        grasp_0 = BooleanProperty(data_params['grasp_hold'] == 0.)
        grasp_5 = BooleanProperty(data_params['grasp_hold'] == 0.05)
        grasp_10 = BooleanProperty(data_params['grasp_hold'] == 0.10)
        grasp_15 = BooleanProperty(data_params['grasp_hold'] == 0.15)
        grasp_20 = BooleanProperty(data_params['grasp_hold'] == 0.20)
        grasp_25 = BooleanProperty(data_params['grasp_hold'] == 0.25)
        grasp_35 = BooleanProperty(data_params['grasp_hold'] == 0.35)
        grasp_50 = BooleanProperty(data_params['grasp_hold'] == 0.50)
        grasp_r_90_120 = BooleanProperty(data_params['grasp_hold'] == '.09-.12')
        grasp_r_110_150 = BooleanProperty(data_params['grasp_hold'] == '.11-.15')

        power = BooleanProperty(False)
        tripod = BooleanProperty(False)
        pinch = BooleanProperty(False)
        tiny = BooleanProperty(False) 
        pinch3 = BooleanProperty(False)

        for trl in data_params['trials_labels']: 
            
            if trl == 'power_1': 
                power = BooleanProperty(True)
            elif trl == 'tripod_1': 
                tripod = BooleanProperty(True)
            elif trl == 'pinch_1': 
                pinch = BooleanProperty(True)
            elif trl == 'tiny_1': 
                tiny = BooleanProperty(True)
            elif trl == 'pinch_3': 
                pinch3 = BooleanProperty(True)

        grasp_to5 = BooleanProperty(data_params['grasp_timeout'] == 5.)
        grasp_to10 = BooleanProperty(data_params['grasp_timeout'] == 10.)
        grasp_to20 = BooleanProperty(data_params['grasp_timeout'] == 20.)
        grasp_to30 = BooleanProperty(data_params['grasp_timeout'] == 30.)
        grasp_toinf = BooleanProperty(data_params['grasp_timeout'] > 40)

        trls_25 = BooleanProperty(data_params['trls_25'])
        trls_50 = BooleanProperty(data_params['trls_50'])
        trls_inf = BooleanProperty(data_params['trls_inf'])
    else: 
        rew_manual = BooleanProperty(False)
        rew_start = BooleanProperty(False)
        rew_start_and_grasp = BooleanProperty(True)
        rew_grasp = BooleanProperty(False)
        rew_snd = BooleanProperty(False)

        only_gripper =BooleanProperty(False)
        only_button = BooleanProperty(False)
        button_grip = BooleanProperty(True)

        monk_haribo = BooleanProperty(True)
        monk_butters = BooleanProperty(False)
        monk_nike = BooleanProperty(False)

        small_rew_1 = BooleanProperty(True)
        small_rew_3 = BooleanProperty(False)
        small_rew_5 = BooleanProperty(False)

        big_rew_3 = BooleanProperty(False)
        big_rew_5 = BooleanProperty(False)
        big_rew_7 = BooleanProperty(True)

        juicer_y = BooleanProperty(True)
        juicer_r = BooleanProperty(False)

        rew_all = BooleanProperty(True)
        rew_50 = BooleanProperty(False)
        rew_30 = BooleanProperty(False)

        button_0 = BooleanProperty(False)
        button_1 = BooleanProperty(False)
        button_2 = BooleanProperty(True)
        button_3 = BooleanProperty(False)
        button_4 = BooleanProperty(False)

        grasp_0 = BooleanProperty(False)
        grasp_15 = BooleanProperty(False)
        grasp_20 = BooleanProperty(False)
        grasp_25 = BooleanProperty(True)
        grasp_35 = BooleanProperty(False)
        grasp_50 = BooleanProperty(False)

        power = BooleanProperty(False)
        tripod = BooleanProperty(False)
        pinch = BooleanProperty(False)
        tiny = BooleanProperty(False) 
        pinch3 = BooleanProperty(False)

        grasp_to5 = BooleanProperty(False)
        grasp_to10 = BooleanProperty(True)
        grasp_toinf = BooleanProperty(False)

        trls_25 = BooleanProperty(False)
        trls_50 = BooleanProperty(True)
        trls_inf = BooleanProperty(False)


class R2GApp(App):
    def build(self, **kwargs):
        from win32api import GetSystemMetrics
        screenx = GetSystemMetrics(0)
        screeny = GetSystemMetrics(1)
        Window.size = (1800, 1000)
        Window.left = (screenx - 1800)/2
        Window.top = (screeny - 1000)/2
        return Manager()
        
if __name__ == '__main__':
    R2GApp().run()