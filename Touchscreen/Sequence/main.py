from kivy.app import App
from kivy.core.window import Window
from kivy.core.audio import SoundLoader
from kivy.core.text import Label as CoreLabel
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, ListProperty, StringProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.vector import Vector
from kivy.clock import Clock
from random import randint
from kivy.config import Config
import serial, time, pickle, datetime #, winsound
from numpy import binary_repr
import struct
import time
import numpy as np
import tables


Config.set('kivy', 'exit_on_escape', 1)
fixed_window_size = (1000, 1000)
pix_per_cm = 85.
Config.set('graphics', 'width', str(fixed_window_size[0]))
Config.set('graphics', 'height', str(fixed_window_size[1]))

class Data(tables.IsDescription):
    state = tables.StringCol(24)   # 24-character String
    cursor = tables.Float32Col(shape=(10, 2))
    cursor_ids = tables.Float32Col(shape = (10, ))
    set_ix = tables.Float32Col(shape=(1, ))
    cap_touch = tables.Float32Col()
    time = tables.Float32Col()

class TargetSet(Widget):
    pos_x = NumericProperty(0)
    pos_y = NumericProperty(0)
    
    def move(self, pos_x, pos_y):
        self.pos = pos_x, pos_y

class SequenceGame(Widget):
    target1 = ObjectProperty(None)
    target2 = ObjectProperty(None)
    
    # Time to wait after starting the video before getting to the first set display
    pre_start_vid_ts = 0.1
    
    # Sets per hyperset
    nsets_per_hyperset = 5
    
    # Inter Set Interval
    ISetI_mean = .5
    ISetI_std = .1
    
    # Inter HyperSet Interval
    IHSI_mean = 1.
    IHSI_std = .2
    
    # Target Radius 
    target_rad = 1.5 # cm
    
    # Exit Button Settings 
    exit_pos = np.array([7, 4]) # cm
    exit_rad = 1. # cm
    exit_hold = 2 # seconds
    
    # Indicator Light Settings
    indicator_pos = np.array([8, 5]) # cm
    
    # Home Hold Times
    ch_timeout = 10. # ch timeout
    cht = .001 # how long to hold in the center until reward?
    
    # Target Hold Times
    set_timeout_time = 5. # how long does the target stay on the screen until it disappears and move to next trial?
    tht = .001 # how long to hold in target until reward?
    
    # Initialize variables for tracking cursor position 
    cursor = {}
    cursor_start = {}
    cursor_ids = []
    
    # Initialize Touch and Prev Touch Boolean
    anytouch_prev = False
    touch = False
    
    # Error Timeout Times
    touch_error_timeout = 0.
    timeout_error_timeout = 0.
    hold_error_timeout = 0.
    drag_error_timeout = 0.
    
    # T/F Done with Initialization
    done_init = False
    
    # Initialize array of previous exit times?
    prev_exit_ts = np.array([0,0])

    # Number of trials: 
    trial_counter = NumericProperty(0)
    #indicator_txt = StringProperty('o')
    #indicator_txt_color = ListProperty([.5, .5, .5, 1.])

    t0 = time.time()

    cht_text = StringProperty('')
    tht_text = StringProperty('')
    targ_size_text = StringProperty('')
    big_rew_text = StringProperty('')
    cht_param = StringProperty('')
    tht_param = StringProperty('')
    targ_size_param = StringProperty('')
    big_rew_time_param = StringProperty('')
    
    
    # Define what to do when there is a touch on the screen
    def on_touch_down(self, touch):
        #handle many touchs:
        ud = touch.ud

        # Add new touch to ids: 
        self.cursor_ids.append(touch.uid)

        # Add cursor
        curs = pix2cm(np.array([touch.x, touch.y]))
        self.cursor[touch.uid] =  curs.copy()
        self.cursor_start[touch.uid] = curs.copy()

        # set self.touch to True
        self.touch = True
    
    
    def on_touch_move(self, touch):
        self.cursor[touch.uid] = pix2cm(np.array([touch.x, touch.y]))
        self.touch = True
    
    def on_touch_up(self, touch):
        try:
            self.cursor_ids.remove(touch.uid)
            _ = self.cursor.pop(touch.uid)
        except:
            print('removing touch from pre-game screen')
    
    
    # Get the initial parameters
    def init(self, animal_names_dict=None, rew_in=None, task_in=None,
        test=None, hold=None, 
        autoquit=None, rew_var=None, set_timeout = None, targ_pos=None):
        
        # Initialize a count of the number of rewards
        self.rew_cnt = 0
        self.small_rew_cnt = 0

        # Do you want to use an exxternal capacitive touch sensor?
        self.use_cap_sensor = False

        if self.use_cap_sensor:
            self.serial_port_cap = serial.Serial(port='COM5')
            
        self.rhtouch_sensor = 0.
        
        # Initiralize the 16 target positions
        # self.possible_target_pos_x = np.array([self.center_x-0.375*self.width, self.center_x-0.125*self.width, self.center_x+0.125*self.width, self.center_x+0.375*self.width, 
        #     self.center_x-0.375*self.width, self.center_x-0.125*self.width, self.center_x+0.125*self.width, self.center_x+0.375*self.width, 
        #     self.center_x-0.375*self.width, self.center_x-0.125*self.width, self.center_x+0.125*self.width, self.center_x+0.375*self.width, 
        #     self.center_x-0.375*self.width, self.center_x-0.125*self.width, self.center_x+0.125*self.width, self.center_x+0.375*self.width])
        # self.possible_target_pos_y = np.array([self.center_y+0.375*self.height, self.center_y+0.375*self.height, self.center_y+0.375*self.height, self.center_y+0.375*self.height, 
        #     self.center_y+0.125*self.height, self.center_y+0.125*self.height, self.center_y+0.125*self.height, self.center_y+0.125*self.height, 
        #     self.center_y-0.125*self.height, self.center_y-0.125*self.height, self.center_y-0.125*self.height, self.center_y-0.125*self.height, 
        #     self.center_y-0.375*self.height, self.center_y-0.375*self.height, self.center_y-0.375*self.height, self.center_y-0.375*self.height])
        
        self.possible_target_pos_x = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
        self.possible_target_pos_y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

        # How long does the set stay on the screen until it disappears and move to next trial? 
        set_timeout_opts = [15, 30, 45, 60]
        for i, val in enumerate(set_timeout['tt']):
            if val:
                self.set_timeout_time = set_timeout_opts[i]
        
        # How long to give small rewards?
        small_rew_opts = [0., .1, .3, .5]
        for i, val in enumerate(rew_in['small_rew']):
            if val:
                small_rew_opts = small_rew_opts[i]
        
        # How long to give big rewards?
        big_rew_opts = [.3, .5, .7]
        for i, val in enumerate(rew_in['big_rew']):
            if val:
                big_rew = big_rew_opts[i]
                
        # Reward for simply touching the target?
        self.reward_for_targtouch = [True, big_rew]
        
        # NOT SURE WHAT THESE DO
        self.reward_for_center = [False, 0]
        self.skip_juice = False
        self.reward_for_anytouch = [False, 0]
        self.testing = False
        self.in_cage = False
        
        # How big are the targets?
        target_rad_opts = [.5, .75, .82, .91, 1.0, 1.5, 2.25, 3.0]
        for i, val in enumerate(task_in['targ_rad']):
            if val:
                self.target_rad = target_rad_opts[i]
        
        # What is the animal's name?
        for i, (nm, val) in enumerate(animal_names_dict.items()):
            if val:
                animal_name = nm
        
        # How long to hold the home and target buttons?
        holdz = [0.0, 0.1, .375, .5, .575, .6, '.4-.6']
        
        self.cht_type = None
        self.tht_type = None

        for i, val in enumerate(hold['hold']):
            if val:
                if type(holdz[i]) is str:
                    mx, mn = holdz[i].split('-')
                    self.tht_type = holdz[i]
                    self.tht =  (float(mn)+float(mx))*.5
                else:
                    self.tht = holdz[i]

        for i, val in enumerate(hold['chold']):
            if val:
                if type(holdz[i]) is str:
                    mx, mn = holdz[i].split('-')
                    self.cht_type = holdz[i]
                    self.cht = (float(mn)+float(mx))*.5
                else:
                    self.cht = holdz[i]
        try:
            pygame.mixer.init()    
        except:
            pass

        # How long is the delay from achieve to receive reward
        self.reward_delay_time = 0.0
        
        # What is the reward variance?
        reward_var_opt = [1.0, .5, .33]
        for i, val in enumerate(rew_var['rew_var']):
            if val:
                self.percent_of_trials_rewarded = reward_var_opt[i]
                if self.percent_of_trials_rewarded == 0.33:
                    self.percent_of_trials_doubled = 0.1
                else:
                    self.percent_of_trials_doubled = 0.0
        
        # Generate the rewards
        self.reward_generator = self.gen_rewards(self.percent_of_trials_rewarded, self.percent_of_trials_doubled,
            self.reward_for_targtouch)
        
        # Auto-quit after how many trials?
        autoquit_trls = [10, 25, 50, 100, 10**10]
        for i, val in enumerate(autoquit['autoquit']):
            if val: 
                self.max_trials = autoquit_trls[i]
        
        # Is it okay to drag fingers into target?
        # drag_ok = [True, False]
        # for i, val in enumerate(drag['drag']):
        #     if val:
        #         self.drag_ok = drag_ok[i]
        self.drag_ok = False;
    
        # Preload sounds: 
        self.reward1 = SoundLoader.load('reward1.wav')
        self.reward2 = SoundLoader.load('reward2.wav')
        
        # Initialize what state we are in
        self.state = 'IHSI'
        self.state_start = time.time()
        self.IHSI = self.IHSI_std + self.IHSI_mean
        
        # Initialize the set number
        self.set_ix = 1

        # Initialize targets
        self.target1.set_size(2*self.target_rad)
        self.target1.move(np.array([0., 0.]))
        self.target2.set_size(2*self.target_rad)
        self.target2.move(np.array([0., 0.]))
        
        # self.exit_target1.set_size(2*self.exit_rad)
        # self.exit_target2.set_size(2*self.exit_rad)
        # self.indicator_targ.set_size(self.exit_rad)
        # self.indicator_targ.move(self.indicator_pos)
        # self.indicator_targ.color = (0., 0., 0., 1.)

        # self.exit_target1.move(self.exit_pos)
        # self.exit_pos2 = np.array([self.exit_pos[0], -1*self.exit_pos[1]])
        # self.exit_target2.move(self.exit_pos2)
        # self.exit_target1.color = (.15, .15, .15, 1)
        # self.exit_target2.color = (.15, .15, .15, 1)
        
        # Determine the hypersets
        self.target_list = np.array([[1, 2, 4, 5, 7, 8, 10, 11, 13, 14], [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]])
        
        # Initialize the Target and Home position
        self.target_HS_index = 0
        self.target_index = np.array([2*(self.set_ix-1), 2*(self.set_ix-1)+1])
        self.repeat = False
        self.home_position = np.array([0., 0.])
        self.target1_position = np.array([self.possible_target_pos_x[self.target_list[self.target_HS_index, self.target_index[0]]], self.possible_target_pos_y[self.target_list[self.target_HS_index, self.target_index[0]]]])
        self.target2_position = np.array([self.possible_target_pos_x[self.target_list[self.target_HS_index, self.target_index[1]]], self.possible_target_pos_y[self.target_list[self.target_HS_index, self.target_index[1]]]])
        
        # Initialize FSM Dictionary
        self.FSM = dict()
        
        # Determine the relative task update functions for each task state
        self.FSM['IHSI'] = dict(end_IHSI='vid_trig', stop=None)
        self.FSM['vid_trig'] = dict(rhtouch='set', stop=None)
        
        self.FSM['set'] = dict(touch_target1 = 'targ1_hold', touch_target2 = 'targ2_hold', set_timeout='timeout_error', stop=None,
            non_rhtouch='RH_touch')#,touch_not_target='touch_error')
        
        self.FSM['targ1_hold'] = dict(finish_targ_hold='targ1_pressed', early_leave_target1_hold = 'hold_error_nopress', 
            targ1_drag_out = 'drag_error_nopress', stop=None, non_rhtouch='RH_touch')
        self.FSM['targ2_hold'] = dict(finish_targ_hold='targ2_pressed', early_leave_target2_hold = 'hold_error_nopress',
            targ2_drag_out = 'drag_error_nopress', stop=None, non_rhtouch='RH_touch')
        
        self.FSM['targ1_pressed'] = dict(touch_target2 ='t1p_targ2_hold', set_timeout='timeout_error', stop=None,
            non_rhtouch='RH_touch')
        self.FSM['targ2_pressed'] = dict(touch_target1 ='t2p_targ1_hold', set_timeout='timeout_error', stop=None,
            non_rhtouch='RH_touch')
        
        self.FSM['t1p_targ2_hold'] = dict(finish_targ_hold='set_complete', early_leave_target2_hold = 'hold_error_t1p',
            targ2_drag_out = 'drag_error_t1p', stop=None, non_rhtouch='RH_touch')
        self.FSM['t2p_targ1_hold'] = dict(finish_targ_hold='IHSI', early_leave_target1_hold = 'hold_error_t2p',
            targ1_drag_out = 'drag_error_t2p', stop=None, non_rhtouch='RH_touch')
        
        
        self.FSM['set_complete'] = dict(hyperset_complete = 'reward_hyperset', next_set = 'reward_set', stop=None, non_rhtouch='RH_touch')
        self.FSM['reward_set'] = dict(end_reward_set = 'set', stop=None, non_rhtouch='RH_touch')
        self.FSM['reward_hyperset'] = dict(end_reward_hyperset = 'IHSI', stop=None, non_rhtouch='RH_touch')
        
        self.FSM['timeout_error'] = dict(end_timeout_error='IHSI', stop=None, non_rhtouch='RH_touch')
        self.FSM['hold_error_nopress'] = dict(end_hold_error_nopress='set', stop=None, non_rhtouch='RH_touch')
        self.FSM['drag_error_nopress'] = dict(end_drag_error_nopress='set', stop=None, non_rhtouch='RH_touch')
        
        self.FSM['hold_error_t1p'] = dict(end_hold_error_t1p='targ1_pressed', stop=None, non_rhtouch='RH_touch')
        self.FSM['drag_error_t1p'] = dict(end_drag_error_t1p='targ1_pressed', stop=None, non_rhtouch='RH_touch')
        
        self.FSM['hold_error_t2p'] = dict(end_hold_error_t2p='targ2_pressed', stop=None, non_rhtouch='RH_touch')
        self.FSM['drag_error_t2p'] = dict(end_drag_error_t2p='targ2_pressed', stop=None, non_rhtouch='RH_touch')
        
        # self.FSM['rew_anytouch'] = dict(end_rewanytouch='target', stop=None, non_rhtouch='RH_touch')
        self.FSM['idle_exit'] = dict(stop=None)
        
        
        # Test the COM ports for the reward and camera
        try:
            self.reward_port = serial.Serial(port='COM4',
                baudrate=115200)
            self.reward_port.close()
        except:
            pass

        try:
            self.dio_port = serial.Serial(port='COM5', baudrate=115200)
            time.sleep(4.)
        except:
            pass

        try:
            self.cam_trig_port = serial.Serial(port='COM6', baudrate=9600)
            time.sleep(3.)
            # Say hello: 
            self.cam_trig_port.write('a'.encode())

            # Start cams @ 50 Hz
            self.cam_trig_port.write('1'.encode())
        except:
            pass

        # save parameters: 
        d = dict(animal_name=animal_name, target_rad=self.target_rad,
            target_list = self.target_list, 
            ISetI_mean=self.ISetI_mean, ISetI_std = self.ISetI_std, 
            IHSI_mean=self.IHSI_mean, IHSI_std = self.IHSI_std, 
            ch_timeout=self.ch_timeout, 
            cht=self.cht, reward_time_small=self.reward_for_center[1],
            reward_time_big=self.reward_for_targtouch[1],
            reward_for_anytouch=self.reward_for_anytouch[0],
            reward_for_center = self.reward_for_center[0],
            reward_for_targtouch=self.reward_for_targtouch[0], 
            touch_error_timeout = self.touch_error_timeout,
            timeout_error_timeout = self.timeout_error_timeout,
            hold_error_timeout = self.hold_error_timeout,
            drag_error_timeout = self.drag_error_timeout,
            start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M'),
            testing=self.testing,
            rew_delay = self.reward_delay_time,
            use_cap_sensor = self.use_cap_sensor,
            drag_ok = self.drag_ok,
            )

        print(self.reward_for_center)
        print(self.reward_for_targtouch)
        print(self.reward_for_anytouch)

        try:
            if self.testing:
                pass

            else:
                import os
                path = os.getcwd()
                path = path.split('\\')
                path_data = [p for p in path if np.logical_and('Touchscreen' not in p, 'Targ' not in p)]
                path_root = ''
                for ip in path_data:
                    path_root += ip+'/'
                p = path_root + 'data/'

                # Check if this directory exists: 
                if os.path.exists(p):
                    pass
                else:
                    p = path_root+ 'data_tmp_'+datetime.datetime.now().strftime('%Y%m%d')+'/'
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
                self.filename = p+ animal_name+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M')
                if self.in_cage:
                    self.filename = self.filename+'_cage'

                pickle.dump(d, open(self.filename+'_params.pkl', 'wb'))
                self.h5file = tables.open_file(self.filename + '_data.hdf', mode='w', title = 'NHP data')
                self.h5_table = self.h5file.create_table('/', 'task', Data, '')
                self.h5_table_row = self.h5_table.row
                self.h5_table_row_cnt = 0

                # Note in python 3 to open pkl files: 
                #with open('xxxx_params.pkl', 'rb') as f:
                #    data_params = pickle.load(f)
        except:
            pass
    
    # Function for Generating Rewards
    def gen_rewards(self, perc_trials_rew, perc_trials_2x, reward_for_grasp):
        mini_block = int(2*(np.round(1./self.percent_of_trials_rewarded)))
        rew = []
        trial_cnt_bonus = 0

        for i in range(500):
            mini_block_array = np.zeros((mini_block))
            ix = np.random.permutation(mini_block)
            mini_block_array[ix[:2]] = reward_for_grasp[1]

            trial_cnt_bonus += mini_block
            if perc_trials_2x > 0:
                if trial_cnt_bonus > int(1./(perc_trials_rew*perc_trials_2x)):
                    mini_block_array[ix[0]] = reward_for_grasp[1]*2.
                    trial_cnt_bonus = 0

            rew.append(mini_block_array)
        return np.hstack((rew))
    
    
    # Function for Closing the App
    def close_app(self):
        # Save Data Eventually
         #Stop the video: 
        try:
            self.cam_trig_port.write('0'.encode())
        except:
            pass

        if self.use_cap_sensor:
            self.serial_port_cap.close()
        
        if self.idle:
            self.state = 'idle_exit'
            self.trial_counter = -1

            # Set relevant params text: 
            self.cht_text = 'Home Hold Time: '
            self.tht_text = 'Target Hold Time: '
            self.targ_size_text = 'Target Radius: '
            self.big_rew_text = 'Big Reward Time: '

            if type(self.cht_type) is str:
                self.cht_param = self.cht_type
            else:
                self.cht_param = 'Constant: ' + str(self.cht)

            if type(self.tht_type) is str:
                self.tht_param = self.tht_type
            else:
                self.tht_param = 'Constant: ' + str(self.tht)

            self.targ_size_param = str(self.target_rad)
            self.big_rew_time_param = str(self.reward_for_targtouch[1])

        else:
            App.get_running_app().stop()
            Window.close()
    
    def update(self, ts):
        self.state_length = time.time() - self.state_start
        self.rew_cnt += 1
        self.small_rew_cnt += 1
        
        # Run task update functions: 
        for f, (fcn_test_name, next_state) in enumerate(self.FSM[self.state].items()):
            kw = dict(ts=self.state_length)
            print(self.state)
            # import pdb; pdb.set_trace()
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
                        
                    # Advance to the next state
                    self.prev_state = self.state
                    self.state = next_state
                    self.state_start = time.time()

                    # Run any starting functions: 
                    start_state_fn_name = "_start_%s" % self.state
                    if hasattr(self, start_state_fn_name):
                        start_state_fn = getattr(self, start_state_fn_name)
                        start_state_fn()
            else:
                while_state_fn_name = "_while_%s" % self.state
                if hasattr(self, while_state_fn_name):
                    while_state_fn = getattr(self, while_state_fn_name)
                    while_state_fn()
            
        if self.use_cap_sensor:
            try:
                self.serial_port_cap.flushInput()
                port_read = self.serial_port_cap.read(4)
                if str(port_read[:2]) == "b'N1'":
                    self.rhtouch_sensor = False
                elif str(port_read[:2]) == "b'C1'":
                    self.rhtouch_sensor = True
                    print(self.rhtouch_sensor)
            except:
                print('passing state! ')
                pass  
        
        # Save the data if this is not a test or we arent exiting
        if self.testing:
            pass
        else:
            if self.state == 'idle_exit':
                pass
            else:
                self.write_to_h5file()
    
    # Function for saving the data
    def write_to_h5file(self):
        self.h5_table_row['state']= self.state; 
        cursor = np.zeros((10, 2))
        cursor[:] = np.nan
        for ic, curs_id in enumerate(self.cursor_ids):
            cursor[ic, :] = self.cursor[curs_id]

        self.h5_table_row['cursor'] = cursor

        cursor_id = np.zeros((10, ))
        cursor_id[:] = np.nan
        cursor_id[:len(self.cursor_ids)] = self.cursor_ids
        self.h5_table_row['cursor_ids'] = cursor_id

        self.h5_table_row['set_ix'] = self.set_ix
        self.h5_table_row['time'] = time.time() - self.t0
        self.h5_table_row['cap_touch'] = self.rhtouch_sensor
        self.h5_table_row.append()

        # Write DIO 
        try:
            self.write_row_to_dio()
        except:
            pass
            
        # Upgrade table row: 
        self.h5_table_row_cnt += 1
    
    def write_row_to_dio(self):
        ### FROM TDT TABLE, 5 is GND, BYTE A ###
        row_to_write = self.h5_table_row_cnt % 256

        ### write to arduino: 
        word_str = b'd' + struct.pack('<H', int(row_to_write))
        self.dio_port.write(word_str)
    
    def stop(self, **kwargs):
        # If past number of max trials then auto-quit: 
        if np.logical_and(self.trial_counter >= self.max_trials, self.state == 'IHSI'):
            self.idle = True
            return True
        # else:
            # Stop if both exit targets are held
            # e = [0, 0]
            # e[0] = self.check_if_cursors_in_targ(self.exit_pos, self.exit_rad)
            # e[1] = self.check_if_cursors_in_targ(self.exit_pos2, self.exit_rad)
            # t = [0, 0]
            # for i in range(2):
            #     if np.logical_and(self.prev_exit_ts[i] !=0, e[i] == True):
            #         t[i] = time.time() - self.prev_exit_ts[i]
            #     elif np.logical_and(self.prev_exit_ts[i] == 0, e[i]==True):
            #         self.prev_exit_ts[i] = time.time()
            #     else:
            #         self.prev_exit_ts[i] = 0
                    
            # if t[0] > self.exit_hold and t[1] > self.exit_hold:
            #     self.idle = False
            #     return True

            # else:
            #     return False
    
    # IHSI State
    def _start_IHSI(self, **kwargs):
        try:
            self.cam_trig_port.write('0'.encode())
        except:
            pass
        Window.clearcolor = (0., 0., 0., 1.)
        # self.exit_target1.color = (.15, .15, .15, 1.)
        # self.exit_target2.color = (.15, .15, .15, 1.)
        
        # Set the set index to 1
        self.set_ix = 1;
        
        # Set IHSI, CHT, THT
        self.IHSI = np.random.random()*self.IHSI_std + self.IHSI_mean

        if type(self.cht_type) is str:
            cht_min, cht_max = self.cht_type.split('-')
            self.cht = ((float(cht_max) - float(cht_min)) * np.random.random()) + float(cht_min)

        if type(self.tht_type) is str:
            tht_min, tht_max = self.tht_type.split('-')
            self.tht = ((float(tht_max) - float(tht_min)) * np.random.random()) + float(tht_min)            
        
        # Make all of the targets invisible
        self.target1.color = (0., 0., 0., 0.)
        self.target2.color = (0., 0., 0., 0.)
        # self.indicator_targ.color = (0., 0., 0., 0.)
    
    def end_IHSI(self, **kwargs):
        return kwargs['ts'] > self.IHSI
    
    # Video Trigger State
    def _start_vid_trig(self, **kwargs):
        if self.trial_counter == 0:
            time.sleep(1.)
        try:    
            self.cam_trig_port.write('1'.encode())
        except:
            pass
        
        # Sets that come after the trigger must be the first set
        self.first_set_attempt = True

        if np.logical_and(self.use_cap_sensor, not self.rhtouch_sensor):
            self.target1.color = (1., 0., 0., 1.)
            self.target2.color = (1., 0., 0., 1.)
            Window.clearcolor = (1., 0., 0., 1.)

            # # Turn exit buttons redish:
            # self.exit_target1.color = (.9, 0, 0, 1.)
            # self.exit_target2.color = (.9, 0, 0, 1.)

    def end_vid_trig(self, **kwargs):
        return kwargs['ts'] > self.pre_start_vid_ts
    
    # Right hand on touch sensor
    def rhtouch(self, **kwargs):
        if self.use_cap_sensor:
            if self.rhtouch_sensor:
                return True
            else:
                return False
        else:
            return True

    def non_rhtouch(self, **kwargs):
        x = not self.rhtouch()
        # if x:
        #     self.repeat = True
        return x
    
    # Start a new set
    def _start_set(self, **kwargs):
        Window.clearcolor = (0., 0., 0., 1.)
        
        self.target_index = np.array([2*(self.set_ix-1), 2*(self.set_ix-1)+1])
        # Change the position of the targets
        if self.repeat is False:
            self.target1_position = np.array([self.possible_target_pos_x[self.target_list[self.target_HS_index, self.target_index[0]]], self.possible_target_pos_y[self.target_list[self.target_HS_index, self.target_index[0]]]])
            self.target2_position = np.array([self.possible_target_pos_x[self.target_list[self.target_HS_index, self.target_index[1]]], self.possible_target_pos_y[self.target_list[self.target_HS_index, self.target_index[1]]]])
            
            # self.target_set_index += 2
            print(self.target1_position)
            print(self.target2_position)
        
        self.target1.move(self.target1_position)
        self.target2.move(self.target2_position)
        self.target1.color = (1., 1., 0., 1.)
        self.target2.color = (1., 1., 0., 1.)
        self.repeat = False
        # self.exit_target1.color = (.15, .15, .15, 1)
        # self.exit_target2.color = (.15, .15, .15, 1)
        # self.indicator_targ.color = (.25, .25, .25, 1.)
        
        if self.first_set_attempt:
            self.first_set_attempt_t0 = time.time();
            self.first_set_attempt = False
    
    # Touch Targets
    def touch_target1(self, **kwargs):
        if self.drag_ok:
            return self.check_if_cursors_in_targ(self.target1_position, self.target_rad)
        else:
            return np.logical_and(self.check_if_cursors_in_targ(self.target1_position, self.target_rad),
                self.check_if_started_in_targ(self.target1_position, self.target_rad))
    
    # Touch Target
    def touch_target2(self, **kwargs):
        if self.drag_ok:
            return self.check_if_cursors_in_targ(self.target2_position, self.target_rad)
        else:
            return np.logical_and(self.check_if_cursors_in_targ(self.target2_position, self.target_rad),
                self.check_if_started_in_targ(self.target2_position, self.target_rad))
    
    
    # Start Target Holds --> Change the Color of the target to yellow?
    def _start_targ1_hold(self, **kwargs):
        self.target1.color = (0., 1., 0., 1.)
        # self.indicator_targ.color = (0.75, .75, .75, 1.)
    
    def _start_targ2_hold(self, **kwargs):
        self.target2.color = (0., 1., 0., 1.)
        # self.indicator_targ.color = (0.75, .75, .75, 1.)
        
    def _start_t2p_targ1_hold(self, **kwargs):
        self.target1.color = (0., 1., 0., 1.)
        # self.indicator_targ.color = (0.75, .75, .75, 1.)
    
    def _start_t1p_targ2_hold(self, **kwargs):
        self.target2.color = (0., 1., 0., 1.)
        # self.indicator_targ.color = (0.75, .75, .75, 1.)
        
    # End Target Holds --> Change the color of the target to same as background
    def _end_targ1_hold(self, **kwargs):
        self.target1.color = (0., 0., 0., 1.)
        
    def _end_targ2_hold(self, **kwargs):
        self.target2.color = (0., 0., 0., 1.)
    
    # Finish target hold?
    def finish_targ_hold(self, **kwargs):
        return self.tht <= kwargs['ts']
    
    # One target has been pressed --> make it invisible
    def _start_targ1_pressed(self, **kwargs):
        self.target1.color = (0., 0., 0., 1.)
        self.target2.color = (1., 1., 0., 1.)
        # self.indicator_targ.color = (0.75, .75, .75, 1.)
        
    def _start_targ2_pressed(self, **kwargs):
        self.target1.color = (1., 1., 0., 1.)
        self.target2.color = (0., 0., 0., 1.)
        # self.indicator_targ.color = (0.75, .75, .75, 1.)
    
    # Once a set is complete, determine whether to restart or move to the next set
    def hyperset_complete(self, **kwargs):
        return self.set_ix == self.nsets_per_hyperset
    
    def next_set(self, **kwargs):
        return self.set_ix < self.nsets_per_hyperset
    
    def _start_reward_set(self, **kwargs):
        self.trial_counter += 1
        Window.clearcolor = (1., 1., 1., 1.)
        self.target1.color = (1., 1., 1., 1.)
        self.target2.color = (1., 1., 1., 1.)
        # self.exit_target1.color = (1., 1., 1., 1.)
        # self.exit_target2.color = (1., 1., 1., 1.)
        self.rew_cnt = 0
        self.cnts_in_rew = 0
        # self.indicator_targ.color = (1., 1., 1., 1.)
        self.repeat = False
        
    def _while_reward_set(self, **kwargs):
        if self.rew_cnt == 1:
            self.run_small_rew()
            self.rew_cnt += 1
            
    def end_reward_set(self, **kwargs):
        # Advance to the next set
        self.set_ix += 1
        
        # The next set will be the first attempt of that set
        self.first_set_attempt = True
        
        return True
      
    def _start_reward_set(self, **kwargs):
        self.trial_counter += 1
        Window.clearcolor = (1., 1., 1., 1.)
        self.target1.color = (1., 1., 1., 1.)
        self.target2.color = (1., 1., 1., 1.)
        # self.exit_target1.color = (1., 1., 1., 1.)
        # self.exit_target2.color = (1., 1., 1., 1.)
        self.rew_cnt = 0
        self.cnts_in_rew = 0
        # self.indicator_targ.color = (1., 1., 1., 1.)
        self.repeat = False
    
    def _while_reward_hyperset(self, **kwargs):
        if self.rew_cnt == 1:
            self.run_big_rew()
            self.rew_cnt += 1
            
    def end_reward_hyperset(self, **kwargs):
        return True
    
    # Early leave from target --> hold error (buttons turn invisible)
    def early_leave_target1_hold(self, **kwargs):
        return not self.check_if_cursors_in_targ(self.target1_position, self.target_rad)
    
    def early_leave_target2_hold(self, **kwargs):
        return not self.check_if_cursors_in_targ(self.target2_position, self.target_rad)
    
    def _start_hold_error_nopress(self, **kwargs):
        self.target1.color = (0., 0., 0., 1.)
        self.target2.color = (0., 0., 0., 1.)
        self.repeat = True
        
    def end_hold_error_nopress(self, **kwargs):
        return kwargs['ts'] >= self.hold_error_timeout     
    
    def _start_hold_error_t1p(self, **kwargs):
        self.target1.color = (0., 0., 0., 1.)
        self.target2.color = (0., 0., 0., 1.)
        self.repeat = True
        
    def end_hold_error_t1p(self, **kwargs):
        return kwargs['ts'] >= self.hold_error_timeout  
    
    def _start_hold_error_t2p(self, **kwargs):
        self.target1.color = (0., 0., 0., 1.)
        self.target2.color = (0., 0., 0., 1.)
        self.repeat = True
        
    def end_hold_error_t2p(self, **kwargs):
        return kwargs['ts'] >= self.hold_error_timeout 
    
    
    # Drag outside of target --> drag error (buttons turn invisible)
    def targ1_drag_out(self, **kwargs):
        touch = self.touch
        self.touch = True
        stay_in = self.check_if_cursors_in_targ(self.target1_position, self.target_rad)
        self.touch = touch
        return not stay_in
    
    def targ2_drag_out(self, **kwargs):
        touch = self.touch
        self.touch = True
        stay_in = self.check_if_cursors_in_targ(self.target2_position, self.target_rad)
        self.touch = touch
        return not stay_in
    
    def _start_drag_error_nopress(self, **kwargs):
        self.target1.color = (0., 0., 0., 1.)
        self.target2.color = (0., 0., 0., 1.)
        self.repeat = True
    
    def end_drag_error_nopress(self, **kwargs):
        return kwargs['ts'] >= self.drag_error_timeout
    
    def _start_drag_error_t1p(self, **kwargs):
        self.target1.color = (0., 0., 0., 1.)
        self.target2.color = (0., 0., 0., 1.)
        self.repeat = True
        
    def end_drag_error_t1p(self, **kwargs):
        return kwargs['ts'] >= self.hold_error_timeout  
    
    def _start_drag_error_t2p(self, **kwargs):
        self.target1.color = (0., 0., 0., 1.)
        self.target2.color = (0., 0., 0., 1.)
        self.repeat = True
        
    def end_drag_error_t2p(self, **kwargs):
        return kwargs['ts'] >= self.hold_error_timeout 
    
    
    # Timeout error
    def set_timeout(self, **kwargs):
        #return kwargs['ts'] > self.target_timeout_time
        if time.time() - self.first_set_attempt_t0 > self.set_timeout_time:
            self.repeat = False
            return True
    
    def _start_timeout_error(self, **kwargs):
        self.target1.color = (0., 0., 0., 1.)
        self.target2.color = (0., 0., 0., 1.)
        #self.repeat = True
        
    def end_timeout_error(self, **kwargs):
        return kwargs['ts'] >= self.timeout_error_timeout
    
    # Check if the cursor is in the target and started in the target
    def check_if_started_in_targ(self, targ_center, targ_rad):
        startedInTarg = False
        if self.touch:
            for id_ in self.cursor_ids:
                # If in target: 
                if np.linalg.norm(np.array(self.cursor[id_]) - targ_center) < targ_rad:
                    if np.linalg.norm(np.array(self.cursor_start[id_]) - targ_center) < targ_rad:
                        startedInTarg = True
        return startedInTarg

    def check_if_cursors_in_targ(self, targ_center, targ_rad):
        if self.touch:
            inTarg = False
            for id_ in self.cursor_ids:
                if np.linalg.norm(np.array(self.cursor[id_]) - targ_center) < targ_rad:
                    inTarg = True

            return inTarg
        else:
            return False
    
    
    # Run Rewards
    def run_big_rew(self, **kwargs):
        try:
            print('in big reward:')
            self.repeat = False
            if self.reward_for_targtouch[0]:
                #winsound.PlaySound('beep1.wav', winsound.SND_ASYNC)
                #sound = SoundLoader.load('reward1.wav')
                print('in big reward 2')
                #print(str(self.reward_generator[self.trial_counter]))
                #print(self.trial_counter)
                #print(self.reward_generator[:100])
                self.reward1.play()

                if not self.skip_juice:
                    if self.reward_generator[self.trial_counter] > 0:
                        self.reward_port.open()
                        #rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.reward_for_targtouch[1])+' sec\n']
                        rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.reward_generator[self.trial_counter])+' sec\n']
                        self.reward_port.write(rew_str)
                        time.sleep(.25 + self.reward_delay_time)
                        run_str = [ord(r) for r in 'run\n']
                        self.reward_port.write(run_str)
                        self.reward_port.close()
        except:
            pass
        
    def run_small_rew(self, **kwargs):
        try:
            if np.logical_or(self.reward_for_anytouch[0], self.reward_for_center[0]):
                #winsound.PlaySound('beep1.wav', winsound.SND_ASYNC)
                sound = SoundLoader.load('reward2.wav')
                sound.play()

                ### To trigger reward make sure reward is > 0:
                if np.logical_or(np.logical_and(self.reward_for_anytouch[0], self.reward_for_anytouch[1] > 0), 
                    np.logical_and(self.reward_for_center[0], self.reward_for_center[1] > 0)):

                    self.reward_port.open()
                    if self.reward_for_anytouch[0]:
                        rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.reward_for_anytouch[1])+' sec\n']
                    elif self.reward_for_center[0]:
                        rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.reward_for_center[1])+' sec\n']
                    self.reward_port.write(rew_str)
                    time.sleep(.25)
                    run_str = [ord(r) for r in 'run\n']
                    self.reward_port.write(run_str)
                    self.reward_port.close()
        except:
            pass

        #self.repeat = True
    
    
   # def new_set(self):
   #      self.x_grid1 = randint(1,4)
   #      self.y_grid1 = randint(1,4)
   #      self.x_grid2 = randint(1,4)
   #      self.y_grid2 = randint(1,4)
        
   #  def update(self, dt):
   #      pos_x1 = self.center_x+(self.x_grid1-3)*self.height/6
   #      pos_y1 = self.center_y+(self.y_grid1-3)*self.height/6
   #      pos_x2 = self.center_x+(self.x_grid2-3)*self.height/6
   #      pos_y2 = self.center_y+(self.y_grid2-3)*self.height/6
   #      self.target1.move(pos_x1, pos_y1)
   #      self.target2.move(pos_x2, pos_y2)
        
   #      try:
   #          if self.curs[0] < pos_x1:
   #              self.new_set()
   #      except:
   #          pass
    
    
class Splash(Widget):
    def init(self, *args):
        self.args = args
        # from sound import Sound
        # Sound.volume_max() 
    
class Target(Widget):
    color = ListProperty([0., 0., 0., 1.])

    def set_size(self, size):
        size_pix = [cm2pix(size), cm2pix(size)]
        self.size=size_pix

    def move(self, pos):
        pos_pix = cm2pix(pos).astype(int)
        pos_pix_int = tuple((int(pos_pix[0]), int(pos_pix[1])))
        self.center = pos_pix_int    
    
    
class Manager(ScreenManager):
    pass 
    
    
class SequenceApp(App):
    def build(self, **kwargs):
        # from win32api import GetSystemMetrics
        # screenx = GetSystemMetrics(0)
        # screeny = GetSystemMetrics(1)
        screenx = 1000
        screeny = 1000
        Window.size = (800, 800)
        Window.left = (screenx - 800)/2
        Window.top = (screeny - 800)/2
        return Manager()    
    
def cm2pix(pos_cm, fixed_window_size=fixed_window_size, pix_per_cm=pix_per_cm):
    # Convert from CM to pixels: 
    pix_pos = pix_per_cm*pos_cm

    if type(pix_pos) is np.ndarray:
        # Translate to coordinate system w/ 0, 0 at bottom left
        pix_pos[0] = pix_pos[0] + (fixed_window_size[0]/2.)
        pix_pos[1] = pix_pos[1] + (fixed_window_size[1]/2.)

    return pix_pos    
    
def pix2cm(pos_pix, fixed_window_size=fixed_window_size, pix_per_cm=pix_per_cm):
    # First shift coordinate system: 
    pos_pix[0] = pos_pix[0] - (fixed_window_size[0]/2.)
    pos_pix[1] = pos_pix[1] - (fixed_window_size[1]/2.)

    pos_cm = pos_pix*(1./pix_per_cm)
    return pos_cm    
    
def reset():
    import kivy.core.window as window
    from kivy.base import EventLoop
    if not EventLoop.event_listeners:
        from kivy.cache import Cache
        window.Window = window.core_select_lib('window', window.window_impl, True)
        Cache.print_usage()
        for cat in Cache._categories:
            Cache._objects[cat] = {}

if __name__ == '__main__':
   # reset()
   SequenceApp().run()
