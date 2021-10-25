from kivy.app import App
from kivy.core.window import Window
from kivy.core.audio import SoundLoader
from kivy.core.text import Label as CoreLabel
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, ListProperty, StringProperty, BooleanProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.vector import Vector
from kivy.clock import Clock
from random import randint
from kivy.config import Config
import serial, time, pickle, datetime
from numpy import binary_repr
import struct
from sys import platform
import os


Config.set('graphics', 'resizable', False)

if platform == 'darwin': # we are on a Mac
    # This probably means that we are testing on a personal laptop
    
    # settings for MBP 16" 2021
    fixed_window_size = (3072, 1920) # we get this automatically now but here it is anyway
    fixed_window_size_cm = (34.5, 21.5) # this is the important part
    pix_per_cm = 104. # we get this automatically now but here it is anyway
elif platform == 'win32':
    # see if there is an external monitor plugged in
    from screeninfo import get_monitors
    mon = get_monitors()
#    if len(get_monitors()) > 1 or get_monitors()[0].height == 1080:
#        # must be an external monitor plugged in
#        # assume that it is the ViewSonic TD2230
#        fixed_window_size = (1920, 1080) # we get this automatically now but here it is anyway
#        fixed_window_size_cm = (47.6, 26.8) # this is the important part
#        pix_per_cm = 40. # we get this automatically now but here it is anyway
#    else:
        # must just be the Surface Pro
        # These are surface pro settings
    fixed_window_size = (2160, 1440) # we get this automatically now but here it is anyway
    fixed_window_size_cm = (47.6, 26.8)
#        fixed_window_size_cm = (22.8, 15.2) # this is the important part
    pix_per_cm = 95. # we get this automatically now but here it is anyway
    import winsound

Config.set('graphics', 'width', str(fixed_window_size[0]))
Config.set('graphics', 'height', str(fixed_window_size[1]))

import time
import numpy as np
import tables

# DETERMINE WHAT COMPUTER WE ARE ON
path = os.getcwd()
if platform == 'darwin': # we are on a Mac
    path = path.split('/')
elif platform == 'win32': # we are on windows
    path = path.split('\\')
for p in path:
    if p == 'BasalGangulia':
        user_id = 'BasalGangulia'
    elif p == 'Ganguly':
        user_id = 'Ganguly'
    elif p == 'stim':
        user_id = 'stim'
    elif p == 'Sandon':
        user_id = 'Sandon'

# LOAD THE MOST RECENT PARMS TO USE AS DEFAULTS
if user_id == 'Sandon':
    last_param_path = '/Users/Sandon/Documents/'
elif user_id == 'Ganguly':
    last_param_path = 'C:/Users/Ganguly/Documents/'
elif user_id == 'BasalGangulia':
    last_param_path = 'C:/Users/BasalGangulia/Documents/'

last_param_path = last_param_path+'most_recent_snake_chase_params.pkl'
if os.path.exists(last_param_path):
    with open(last_param_path, 'rb') as f:
        data_params = pickle.load(f)
        

class Data(tables.IsDescription):
    state = tables.StringCol(24)   # 24-character String
    cursor = tables.Float32Col(shape=(10, 2))
    cursor_ids = tables.Float32Col(shape = (10, ))
    target_pos = tables.Float32Col(shape=(2, ))
    time = tables.Float32Col()

class COGame(Widget):
    # Time to wait after starting the video before getting to the first target display. 
    pre_start_vid_ts = 0.1
    
    # ITI LENGTH
    ITI_mean = 1.0
    ITI_std = .2
    target_rad = 1.5
    eff_target_rad = 1.5
    
    # SET THE POSITION OF THE EXIT BUTTONS AND THE PHOTODIODE INDICATOR LIGHT
    # positions are in CM measured from the center of the screen
    # if platform == 'darwin':
    
    exit_pos_x = (fixed_window_size_cm[0]/2)-1.5
    exit_pos_y = (fixed_window_size_cm[1]/2)-1.5
                 
    exit_pos = np.array([exit_pos_x, exit_pos_y])
    pd_ind_pos_x = (fixed_window_size_cm[0]/2)-0.5
    pd_ind_pos_y = (fixed_window_size_cm[1]/2)-0.5
                 
    pd_indicator_pos = np.array([pd_ind_pos_x, pd_ind_pos_y])
    
    vid_ind_pos_x = -(fixed_window_size_cm[0]/2)+0.5
    vid_ind_pos_y = -(fixed_window_size_cm[1]/2)+0.5
    vid_indicator_pos = np.array([vid_ind_pos_x, vid_ind_pos_y])
    # elif platform == 'win32':
    #     exit_pos = np.array([7, 4])
    #     pd_indicator_pos = np.array([8, 5])
    exit_rad = 1.
    exit_hold = 2 #seconds
    
    # SET THE HOLD AND TIMEOUT TIMES
    target_timeout_time = 5.
    tht = .001

    cursor = {}
    cursor_start = {}
    cursor_ids = []

    anytouch_prev = False
    touch_error_timeout = 0.
    timeout_error_timeout = 0.
    hold_error_timeout = 0.
    drag_error_timeout = 0.

    touch = False

    target1 = ObjectProperty(None)
    target2 = ObjectProperty(None)

    done_init = False
    prev_exit_ts = np.array([0,0])

    t0 = time.time()
    
    trials_started = 0
    trial_counter = NumericProperty(0)
    percent_correct = StringProperty('')
    tht_text = StringProperty('')
    targ_size_text = StringProperty('')
    big_rew_text = StringProperty('')
    tht_param = StringProperty('')
    targ_size_param = StringProperty('')
    big_rew_time_param = StringProperty('')
    percent_correct_text = StringProperty('')
        
    
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
            
    def init(self, animal_names_dict=None, rew_in=None, task_in=None,
        hold=None, autoquit=None, rew_var=None, targ_timeout = None, 
        drag=None, nudge_x=None, screen_size=None):
        
        self.rew_cnt = 0

        # TARGET TIMEOUT
        targ1_timeout_opts = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 10.0]
        self.target1_timeout_time = 10000
        for i, val in enumerate(targ_timeout['t1tt']):
            if val:
                self.target1_timeout_time = targ1_timeout_opts[i]
        
        targ_timeout_opts = [0.7, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        self.target_timeout_time = 10000
        for i, val in enumerate(targ_timeout['tt']):
            if val:
                self.target_timeout_time = targ_timeout_opts[i]
        
        # JUICE REWARD SETTINGS
        button_rew_opts = [0., .1, .3, .5]
        for i, val in enumerate(rew_in['button_rew']):
            if val:
                button_rew = button_rew_opts[i]

        big_rew_opts = [0., .3, .5, .7]
        for i, val in enumerate(rew_in['big_rew']):
            if val:
                big_rew = big_rew_opts[i]
                
        if big_rew > 0.0:
            self.last_targ_reward = [True, big_rew]
        else:
            self.last_targ_reward = [False, 0]
        
        if button_rew > 0.0:
            self.button_rew = [True, button_rew]
        else:
            self.button_rew = [False, 0]
        
        
        # NUDGE X
        nudge_x_opts = [-6, -4, -2, 0, 2, 4, 6]    
        for i, val in enumerate(nudge_x['nudge_x']):
            if val:
                self.nudge_x = nudge_x_opts[i]
                
        
        # WHERE TO CONSIDER THE TOP AND BOTTOM OF THE SCREEN (HOW MUCH TO SHRINK IT DOWN/UP BY)
        screen_top_opts = [-12, -10, -8, -6, -4, -2, 0]    
        for i, val in enumerate(screen_size['screen_top']):
            if val:
                self.screen_top = screen_top_opts[i]
                
        screen_bot_opts = [0, 2, 4, 6, 8, 10, 12]    
        for i, val in enumerate(screen_size['screen_bot']):
            if val:
                self.screen_bot = screen_bot_opts[i]
        
        # TARGET RADIUS
        target_rad_opts = [.5, .75, .82, .91, 1.0, 1.5, 1.85, 2.25, 3.0, 4.0]
        for i, val in enumerate(task_in['targ_rad']):
            if val:
                self.target_rad = target_rad_opts[i]
                
        eff_target_rad_opts = ['Same As Appears', 1.0, 2.0, 3.0, 4.0, 5.0]
        for i, val in enumerate(task_in['eff_targ_rad']):
            if val:
                self.eff_target_rad = eff_target_rad_opts[i]
        
        if self.eff_target_rad == 'Same As Appears':
            self.eff_target_rad = self.target_rad
            
                
        # TARGET POSITIONS
        seq_opts = ['Straight_Right', 'Right_Up']
        self.seq = False
        for i, val in enumerate(task_in['seq']):
            if val:
                self.seq = seq_opts[i]
        
        if self.seq == 'Straight_Right':
            self.intertarget_angle_deg = np.array([0, 0, 0, 0])
        elif self.seq == 'Right_Up':
            self.intertarget_angle_deg = np.array([0, 0, 90, 90])
            
        self.num_targets = len(self.intertarget_angle_deg) + 1
        self.intertarget_angle_rad = self.intertarget_angle_deg/57.2958
        self.intertarget_y = np.sin(self.intertarget_angle_rad)*self.target_rad*2
        self.intertarget_x = np.cos(self.intertarget_angle_rad)*self.target_rad*2
        
        self.center_position = np.array([0., 0.])
        # lower the center position by half of the total amount the screen height has been shrunk by
        self.center_position[1] = self.center_position[1] + self.screen_top/2 + self.screen_bot/2
        
        d_center2top = (fixed_window_size_cm[1]/2)-((self.screen_top/2)+(self.screen_bot/2))
        d_center2bot = (fixed_window_size_cm[1]/2)+((self.screen_top/2)+(self.screen_bot/2))
        self.max_y_from_center = (fixed_window_size_cm[1]+self.screen_top-self.screen_bot)/2-self.target_rad
        
        # start_target
        target_pos_opts = ['random', 'center', 'upper_left', 'middle_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'middle_right', 'lower_right']
        for i, val in enumerate(task_in['start_pos']):
            if val:
                self.start_pos_str = target_pos_opts[i]
        
        if self.start_pos_str == 'random': # set for now, will get overriden later
            targ_x = self.center_position[0]+self.nudge_x
            targ_y = self.center_position[1]
        elif self.start_pos_str == 'center':
            targ_x = self.center_position[0]+self.nudge_x
            targ_y = self.center_position[1]
        elif self.start_pos_str == 'upper_middle':
            targ_x = self.center_position[0]+self.nudge_x
            targ_y = self.center_position[1] + self.max_y_from_center
        elif self.start_pos_str == 'lower_middle':
            targ_x = self.center_position[0]+self.nudge_x
            targ_y = self.center_position[1] - self.max_y_from_center
        elif self.start_pos_str == 'upper_right':
            targ_x = self.max_y_from_center+self.nudge_x
            targ_y = self.center_position[1] + self.max_y_from_center
        elif self.start_pos_str == 'middle_right':
            targ_x = self.max_y_from_center+self.nudge_x
            targ_y = self.center_position[1]
        elif self.start_pos_str == 'lower_right':
            targ_x = self.max_y_from_center+self.nudge_x
            targ_y = self.center_position[1] - self.max_y_from_center
        elif self.start_pos_str == 'lower_left':
            targ_x = -self.max_y_from_center+self.nudge_x
            targ_y = self.center_position[1] - self.max_y_from_center
        elif self.start_pos_str == 'middle_left':
            targ_x = -self.max_y_from_center+self.nudge_x
            targ_y = self.center_position[1]
        elif self.start_pos_str == 'upper_left':
            targ_x = -self.max_y_from_center+self.nudge_x
            targ_y = self.center_position[1] + self.max_y_from_center
            
        self.start_position = np.array([targ_x, targ_y])
        
        self.target_position = np.zeros([self.num_targets, 2])
        self.target_position[0] = self.start_position
        
        target_pos_lims_x = [2+self.target_rad, (fixed_window_size_cm[0]/2)-2-self.target_rad]
        target_pos_lims_y = [-(fixed_window_size_cm[1]/2)+self.screen_bot+self.target_rad, (fixed_window_size_cm[1]/2)+self.screen_top-self.target_rad]

        for i, val in enumerate(self.intertarget_y):
            self.target_position[i+1] = self.target_position[i] + np.array([self.intertarget_x[i], self.intertarget_y[i]])
            # do not proceed if the full target is not on the screen
            if self.target_position[i+1][0] < target_pos_lims_x[0] or self.target_position[i+1][0] > target_pos_lims_x[1]:
                ValueError("NOT ALL TARGETS ARE WITHIN THE WORKSPACE")
            elif self.target_position[i+1][1] < target_pos_lims_y[0] or self.target_position[i+1][1] > target_pos_lims_y[1]:
                ValueError("NOT ALL TARGETS ARE WITHIN THE WORKSPACE")

        self.target_index = 1
        self.active_target_position = self.target_position[self.target_index-1]
        
        
        # # HOW MUCH TIME TO WAIT UNTIL THE NEXT TARGET APPEARS
        # time_to_next_targ_opts = [False, 0.25, 0.5, 0.75, 1.0, 1.5]
        # for i, val in enumerate(task_in['time_to_next_targ']):
        #     if val:
        #         self.time_to_next_targ = time_to_next_targ_opts[i]
        self.time_to_next_targ = False
        
        # ANIMAL NAME
        for i, (nm, val) in enumerate(animal_names_dict.items()):
            if val:
                animal_name = nm


        # BUTTON HOLD TIME
        holdz = [False, 0.0, 0.1, 0.2, 0.3, 0.4, .5, .6, 0.7, 0.8, 0.9, 1.0, '.2-.4', '.6-.8', '.8-1.0']
        self.button_hold_time_type = None
        for i, val in enumerate(hold['button_hold']):
            if val:
                if type(holdz[i]) is str:
                    mx, mn = holdz[i].split('-')
                    self.button_hold_time_type = holdz[i]
                    self.button_hold_time =  (float(mn)+float(mx))*.5
                else:
                    self.button_hold_time = holdz[i]
                    
        if self.button_hold_time is False:
            self.use_button = False
        else:
            self.use_button = True
        
        # TARGET HOLD TIME
        holdz = [0.0, 0.1, 0.2, 0.3, 0.4, .5, .6, '.1-.3', '.4-.6']
        self.tht_type = None
        for i, val in enumerate(hold['hold']):
            if val:
                if type(holdz[i]) is str:
                    mx, mn = holdz[i].split('-')
                    self.tht_type = holdz[i]
                    self.tht =  (float(mn)+float(mx))*.5
                else:
                    self.tht = holdz[i]
        
        try:
            pygame.mixer.init()    
        except:
            pass

        # reward_delay_opts = [0., .4, .8, 1.2]
        # for i, val in enumerate(rew_del['rew_del']):
        #     if val:
        self.reward_delay_time = 0.0

        reward_var_opt = [1.0, .5, .33]
        for i, val in enumerate(rew_var['rew_var']):
            if val:
                self.percent_of_trials_rewarded = reward_var_opt[i]
                if self.percent_of_trials_rewarded == 0.33:
                    self.percent_of_trials_doubled = 0.1
                else:
                    self.percent_of_trials_doubled = 0.0
        
        self.reward_generator = self.gen_rewards(self.percent_of_trials_rewarded, self.percent_of_trials_doubled,
            self.last_targ_reward)


        # white_screen_opts = [True, False]
        # for i, val in enumerate(white_screen['white_screen']):
        #     if val:
        self.use_white_screen = False

        self.testing = False

        autoquit_trls = [10, 25, 50, 100, 10**10]
        for i, val in enumerate(autoquit['autoquit']):
            if val: 
                self.max_trials = autoquit_trls[i]
        
        # OK to drag into the targets?
        self.drag_ok = False;
        drag_opts = [True, False]
        for i, val in enumerate(drag['drag']):
            if val:
                self.drag_ok = drag_opts[i]
        

        # nudge_9am_dist = [0., .5, 1.]
        # for i, val in enumerate(nudge['nudge']):
        #     if val:
        self.nudge_dist = 0.

        # targ_pos = ['corners', None]
        # for i, val in enumerate(targ_pos['targ_pos']):
        #     if val:
        self.generator_kwarg = 'corners'


        # Preload sounds: 
        self.reward1 = SoundLoader.load('reward1.wav')
        self.reward2 = SoundLoader.load('reward2.wav')

        self.state = 'ITI'
        self.state_start = time.time()
        self.ITI = self.ITI_std + self.ITI_mean

        # Initialize targets: 
        self.target1.set_size(2*self.target_rad)
        self.target1.move(self.target_position[0])
        self.target1.set_size(2*self.target_rad)
        self.target2.set_size(2*self.target_rad)

        self.exit_target1.set_size(2*self.exit_rad)
        self.exit_target2.set_size(2*self.exit_rad)
        self.pd_indicator_targ.set_size(self.exit_rad)
        self.pd_indicator_targ.move(self.pd_indicator_pos)
        self.pd_indicator_targ.color = (0., 0., 0., 1.)
        self.vid_indicator_targ.set_size(self.exit_rad)
        self.vid_indicator_targ.move(self.vid_indicator_pos)
        self.vid_indicator_targ.color = (0., 0., 0., 1.)
        
        self.exit_target1.move(self.exit_pos)
        self.exit_pos2 = np.array([self.exit_pos[0], -1*self.exit_pos[1]])
        self.exit_target2.move(self.exit_pos2)
        self.exit_target1.color = (.15, .15, .15, 1)
        self.exit_target2.color = (.15, .15, .15, 1)

        self.repeat = False

        self.FSM = dict()
        self.FSM['ITI'] = dict(end_ITI='vid_trig', stop=None)
        self.FSM['vid_trig'] = dict(end_vid_trig='button', stop=None)
        self.FSM['button'] = dict(button_held='target', stop=None)

        self.FSM['target'] = dict(touch_target = 'targ_hold', target_timeout='timeout_error', stop=None)
        self.FSM['targ_hold'] = dict(finish_last_targ_hold='reward', finish_targ_hold='target', early_leave_target_hold = 'hold_error',
         targ_drag_out = 'drag_error', stop=None)
        self.FSM['reward'] = dict(end_reward = 'ITI', stop=None)
        
        self.FSM['touch_error'] = dict(end_touch_error='target', stop=None)
        self.FSM['timeout_error'] = dict(end_timeout_error='ITI', stop=None)
        self.FSM['hold_error'] = dict(end_hold_error='target', stop=None)
        self.FSM['drag_error'] = dict(end_drag_error='target', stop=None)
        self.FSM['idle_exit'] = dict(stop=None)
        
        # OPEN PORTS
        try:
            if user_id == 'Ganguly':
                self.reward_port = serial.Serial(port='COM4',
                    baudrate=115200)
            elif user_id == 'BasalGangulia':
                self.reward_port = serial.Serial(port='COM3',
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
        
        # External button
        try:
            self.is_button_ard = True
            if platform == 'darwin':
                self.button_ard = serial.Serial(port='/dev/cu.usbmodem1421301', baudrate=9600)
            else:
                if user_id == 'Ganguly':
                    self.button_ard = serial.Serial(port='COM3', baudrate=9600) 
                elif user_id == 'BasalGangulia':
                    self.button_ard = serial.Serial(port='COM9', baudrate=9600)
        except:
            self.is_button_ard = False

        if self.is_button_ard: 
            baseline_data = []
            for _ in range(100): 
                ser = self.button_ard.flushInput()
                _ = self.button_ard.readline()
                port_read = self.button_ard.readline()
                port_read = port_read.decode('ascii')
                i_slash = port_read.find('/')
                fsr1 = int(port_read[0:i_slash])
                fsr2 = int(port_read[i_slash+1:])
                baseline_data.append([fsr1, fsr2])
                time.sleep(.005)
            baseline_data = np.vstack((baseline_data))
            self.fsr_baseline = 100+1.5*np.max(baseline_data, axis=0)
        else: 
            self.fsr_baseline = np.array([200, 200])

        # save parameters: 
        d = dict(animal_name=animal_name,
            user_id = user_id,
            max_trials = self.max_trials,
            target1_timeout_time = self.target1_timeout_time,
            target_timeout_time = self.target_timeout_time,
            button_rew = button_rew,
            last_targ_reward = self.last_targ_reward[1],
            nudge_x = self.nudge_x,
            screen_top = self.screen_top,
            screen_bot = self.screen_bot,
            target_rad=self.target_rad,
            effective_target_rad=self.eff_target_rad,
            center_position = self.center_position,
            seq = self.seq,
            start_pos_str = self.start_pos_str,
            target_position = self.target_position,
            time_to_next_targ = self.time_to_next_targ,
            button_hold_time = self.button_hold_time,
            target_hold_time = self.tht,
            rew_delay = self.reward_delay_time,
            percent_of_trials_rewarded = self.percent_of_trials_rewarded,
            ITI_mean=self.ITI_mean, ITI_std = self.ITI_std,
            touch_error_timeout = self.touch_error_timeout,
            timeout_error_timeout = self.timeout_error_timeout,
            hold_error_timeout = self.hold_error_timeout,
            drag_error_timeout = self.drag_error_timeout,
            start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M'),
            testing=self.testing,
            drag_ok = self.drag_ok, 
            fsr_baseline = self.fsr_baseline
            )
        
        if self.testing:
            pass

        else:
            # Try saving to Box
            if user_id == 'Sandon':
                box_path = '/Users/Sandon/Box/Data/NHP_BehavioralData/snake_chase/'
                last_param_path = '/Users/Sandon/Documents/'
            elif user_id == 'Ganguly':
                box_path = 'C:/Users/Ganguly/Box/Data/NHP_BehavioralData/snake_chase/'
                last_param_path = 'C:/Users/Ganguly/Documents/'
            elif user_id == 'BasalGangulia':
                box_path = 'C:/Users/BasalGangulia/Box/Data/NHP_BehavioralData/snake_chase/'
                last_param_path = 'C:/Users/BasalGangulia/Documents/'
            
            # Check if the Box directory exists
            if os.path.exists(box_path):
                p = box_path
            else:
                # if there is no path to box, then save in a data_tmp folder within the CWD
                path = os.getcwd()
                path = path.split('\\')
                path_data = [p for p in path]
                path_root = ''
                for ip in path_data:
                    path_root += ip+'/'
                p = path_root+ 'data_tmp_'+datetime.datetime.now().strftime('%Y%m%d')+'/'
                if os.path.exists(p):
                    pass
                else:
                    os.mkdir(p)
                    print('Making temp directory: ', p)
                last_param_path = p
            print('Auto path : %s'%p)

            print ('')
            print ('')
            print('Data saving PATH: ', p)
            print ('')
            print ('')
            self.filename = p+ animal_name+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M')
        
            # save the params for this run in the _last_params
            pickle.dump(d, open(last_param_path+'most_recent_snake_chase_params.pkl', 'wb'))
    
            # save the params for this run in the same place where we're saving the data
            pickle.dump(d, open(self.filename+'_params.pkl', 'wb'))
            self.h5file = tables.open_file(self.filename + '_data.hdf', mode='w', title = 'NHP data')
            self.h5_table = self.h5file.create_table('/', 'task', Data, '')
            self.h5_table_row = self.h5_table.row
            self.h5_table_row_cnt = 0

            # Note in python 3 to open pkl files: 
            # with open('xxxx_params.pkl', 'rb') as f:
            #    data_params = pickle.load(f)
        # except:
        #     pass

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

    def gen_trials(self, seq):

        if seq == 'center out': 
            num_targs = 4
        elif seq == 'button out': 
            num_targs = 5 

        trials = np.arange(num_targs)
        trial_order = []
        for i in range(1000): # max 4000 targets  
            ix = np.random.permutation(num_targs)
            trial_order.append(trials[ix])
        
        return np.hstack((trial_order))

    def close_app(self):
        # Save Data Eventually
         #Stop the video: 
        try:
            self.cam_trig_port.write('0'.encode())
        except:
            pass
        
        if self.idle:
            self.state = 'idle_exit'
            # self.trial_counter -= 1

            # Set relevant params text: 
            self.percent_correct_text = 'Percent Correct:'
            self.tht_text = 'Target Hold Time: '
            self.targ_size_text = 'Target Radius: '
            self.big_rew_text = 'Big Reward Time: '

            if type(self.tht_type) is str:
                self.tht_param = self.tht_type
            else:
                self.tht_param = 'Constant: ' + str(self.tht)

            self.targ_size_param = str(self.target_rad)
            self.big_rew_time_param = str(self.last_targ_reward[1])
            self.percent_correct = str(round(100*self.trial_counter/self.trials_started)) + '%'

        else:
            App.get_running_app().stop()
            Window.close()

    def update(self, ts):
        self.state_length = time.time() - self.state_start
        self.rew_cnt += 1
        
        if self.is_button_ard:
            # Get the button values
            ser = self.button_ard.flushInput()
            _ = self.button_ard.readline()
            port_read = self.button_ard.readline()
            port_read = port_read.decode('ascii')
            i_slash = port_read.find('/')
            fsr1 = int(port_read[0:i_slash])
            fsr2 = int(port_read[i_slash+1:])
        
            # Determine if the button was pressed or not
            if fsr1 > self.fsr_baseline[0] or fsr2 > self.fsr_baseline[1]:
                self.button_pressed = True
                # print('Button Pressed')
            else:
                self.button_pressed = False
                # print('Button NOT Pressed')
        else:
            self.button_pressed = False
        
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
                
                break
            else:
                while_state_fn_name = "_while_%s" % self.state
                if hasattr(self, while_state_fn_name):
                    while_state_fn = getattr(self, while_state_fn_name)
                    while_state_fn()
             
        if self.testing:
            pass
        else:
            if self.state == 'idle_exit':
                pass
            else:
                self.write_to_h5file()

    def write_to_h5file(self):
        self.h5_table_row['state']= self.state; 
        cursor = np.zeros((10, 2))
        cursor[:] = np.nan
        for ic, curs_id in enumerate(self.cursor_ids):
            cursor[ic, :] = self.cursor[curs_id]

        self.h5_table_row['cursor'] = cursor
        if np.isnan(cursor).all():
            self.vid_indicator_targ.color = (.25, .25, .25, 1.)
        else:
            self.vid_indicator_targ.color = (.5, .5, .5, 1.)

        cursor_id = np.zeros((10, ))
        cursor_id[:] = np.nan
        cursor_id[:len(self.cursor_ids)] = self.cursor_ids
        self.h5_table_row['cursor_ids'] = cursor_id

        self.h5_table_row['target_pos'] = self.active_target_position
        self.h5_table_row['time'] = time.time() - self.t0
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
        if np.logical_and(self.trial_counter >= self.max_trials, self.state == 'ITI'):
            self.idle = True
            return True
        else:
            e = [0, 0]
            e[0] = self.check_if_cursors_in_targ(self.exit_pos, self.exit_rad)
            e[1] = self.check_if_cursors_in_targ(self.exit_pos2, self.exit_rad)
            t = [0, 0]
            for i in range(2):
                if np.logical_and(self.prev_exit_ts[i] !=0, e[i] == True):
                    t[i] = time.time() - self.prev_exit_ts[i]
                elif np.logical_and(self.prev_exit_ts[i] == 0, e[i]==True):
                    self.prev_exit_ts[i] = time.time()
                else:
                    self.prev_exit_ts[i] = 0
                    
            if t[0] > self.exit_hold and t[1] > self.exit_hold:
                self.idle = False
                return True

            else:
                return False

    def _start_ITI(self, **kwargs):
        try:
            self.cam_trig_port.write('0'.encode())
        except:
            pass
        Window.clearcolor = (0., 0., 0., 1.)
        self.exit_target1.color = (.15, .15, .15, 1.)
        self.exit_target2.color = (.15, .15, .15, 1.)

        # Set ITI, CHT, THT
        self.ITI = np.random.random()*self.ITI_std + self.ITI_mean

        if type(self.tht_type) is str:
            tht_min, tht_max = self.tht_type.split('-')
            self.tht = ((float(tht_max) - float(tht_min)) * np.random.random()) + float(tht_min)      
        
        if type(self.button_hold_time_type) is str:
            bht_min, bht_max = self.button_hold_time_type.split('-')
            self.button_hold_time = ((float(bht_max) - float(bht_min)) * np.random.random()) + float(bht_min)     
        
        self.target1.color = (0., 0., 0., 0.)
        self.target1.color = (0., 0., 0., 0.)
        self.target2.color = (0., 0., 0., 0.)
        self.pd_indicator_targ.color = (0., 0., 0., 0.)
        self.trials_started += 1
        
    def end_ITI(self, **kwargs):
        return kwargs['ts'] > self.ITI

    def _start_vid_trig(self, **kwargs):
        if self.trial_counter == 0:
            time.sleep(1.)
        try:    
            self.cam_trig_port.write('1'.encode())
        except:
            pass
        self.first_target_attempt = True
        self.first_time_for_this_targ = True
        
        # Reset target index back to 1
        self.target_index = 1

        # Get the start position
        if self.start_pos_str == 'random': 
            i_pos = self.trial_order[self.trials_started]
        
            if i_pos == 0: # 'upper_right':
                targ_x = self.max_y_from_center+self.nudge_x_t2
                targ_y = self.center_position[1] + self.max_y_from_center
            elif i_pos == 1:# 'lower_right':
                targ_x = self.max_y_from_center+self.nudge_x_t2
                targ_y = self.center_position[1] - self.max_y_from_center
            elif i_pos == 2: # 'lower_left':
                targ_x = -self.max_y_from_center+self.nudge_x_t2
                targ_y = self.center_position[1] - self.max_y_from_center
            elif i_pos == 3: # 'upper_left':
                targ_x = -self.max_y_from_center+self.nudge_x_t2
                targ_y = self.center_position[1] + self.max_y_from_center
            elif i_pos == 4: # center positon 
                targ_x = self.center_position[0]
                targ_y = self.center_position[1]

            self.target_position[0] = np.array([targ_x, targ_y])

                

    def end_vid_trig(self, **kwargs):
        return kwargs['ts'] > self.pre_start_vid_ts
    
    def _start_button(self, **kwargs):
        Window.clearcolor = (0., 0., 0., 1.)
        self.target1.color = (0., 0., 0., 0.)
        self.target2.color = (0., 0., 0., 0.)
        self.exit_target1.color = (.15, .15, .15, 1)
        self.exit_target2.color = (.15, .15, .15, 1)
        self.pd_indicator_targ.color = (.25, .25, .25, 1.)
        self.button_pressed_prev = False
        
    def button_held(self, **kwargs):
        if self.use_button is False:
            return True
        else:
            button_pressed_prev = self.button_pressed_prev
            self.button_pressed_prev = self.button_pressed
            if self.button_pressed:
                if button_pressed_prev:
                    if time.time() - self.t_button_hold_start > self.button_hold_time:
                        # Play the button reward sound
                        sound = SoundLoader.load('C.wav')
                        sound.play()
                        # if the button has been held down long enough
                        if self.button_rew[0]:
                            self.run_button_rew()
                        return True
                    else:
                        return False
                else:
                    # this is the first cycle that the button has been pressed for
                    self.t_button_hold_start = time.time()
                    return False
            else:
                return False

    def _start_targ_hold(self, **kwargs):
        self.target1.color = (0., 1., 0., 1.)
        self.pd_indicator_targ.color = (0.25, .25, .25, 1.)

    def _end_targ_hold(self, **kwargs):
        self.target1.color = (0., 0., 0., 0.)
        
        # Need to reset this
        self.first_time_for_this_targ = True

    def _start_touch_error(self, **kwargs):
        self.target1.color = (0., 0., 0., 1.)
        self.target1.color = (0., 0., 0., 1.)
        self.target2.color = (0., 0., 0., 1.)
        self.repeat = True

    def _start_timeout_error(self, **kwargs):
        self.target1.color = (0., 0., 0., 1.)
        self.target1.color = (0., 0., 0., 1.)
        self.target2.color = (0., 0., 0., 1.)
        #self.repeat = True

    def _start_hold_error(self, **kwargs):
        self.target1.color = (0., 0., 0., 1.)
        self.target1.color = (0., 0., 0., 1.)
        self.target2.color = (0., 0., 0., 1.)
        self.repeat = True

    def _start_drag_error(self, **kwargs):
        self.target1.color = (0., 0., 0., 1.)
        self.target1.color = (0., 0., 0., 1.)
        self.target2.color = (0., 0., 0., 1.)
        self.repeat = True
                
    def _start_target(self, **kwargs):
        Window.clearcolor = (0., 0., 0., 1.)
        self.target1.color = (0., 0., 0., 0.)
        
        self.active_target_position = self.target_position[self.target_index-1]
        if self.target_index == self.num_targets:
            self.next_target_position = False
        else:
            self.next_target_position = self.target_position[self.target_index]

        self.target1.move(self.active_target_position)
        self.target1.color = (1., 1., 0., 1.)
        
        self.exit_target1.color = (.15, .15, .15, 1)
        self.exit_target2.color = (.15, .15, .15, 1)
        self.pd_indicator_targ.color = (.75, .75, .75, 1.)
        if self.first_target_attempt:
            self.first_target_attempt_t0 = time.time();
            self.first_target_attempt = False
        
        if self.first_time_for_this_targ:
            self.first_time_for_this_targ_t0 = time.time()
            self.target2.color = (0., 0., 0., 0.)
            self.first_time_for_this_targ = False
            
        self.repeat = False
    
    def _while_target(self, **kwargs):
        # check and see if it is time for the next target to appear
        if self.time_to_next_targ is not False:
            if time.time() - self.first_time_for_this_targ_t0 > self.time_to_next_targ and self.target_index < self.num_targets:
                # illuminate the next target
                self.target2.move(self.next_target_position)
                self.target2.color = (1., 1., 0., 1.)

    def _start_reward(self, **kwargs):
        self.trial_counter += 1
        Window.clearcolor = (1., 1., 1., 1.)
        self.target1.color = (1., 1., 1., 1.)
        self.target2.color = (1., 1., 1., 1.)
        self.exit_target1.color = (1., 1., 1., 1.)
        self.exit_target2.color = (1., 1., 1., 1.)
        self.rew_cnt = 0
        self.cnts_in_rew = 0
        self.pd_indicator_targ.color = (1., 1., 1., 1.)
        self.repeat = False

    def _while_reward(self, **kwargs):
        if self.rew_cnt == 1:
            self.run_big_rew()
            self.rew_cnt += 1

    def run_big_rew(self, **kwargs):
        try:
            self.reward1 = SoundLoader.load('reward1.wav')
            self.reward1.play()
        except:
            pass
        
        try:
            print('in big reward:')
            self.repeat = False
            if self.last_targ_reward[0]:
                #winsound.PlaySound('beep1.wav', winsound.SND_ASYNC)
                #sound = SoundLoader.load('reward1.wav')
                print('in big reward 2')
                #print(str(self.reward_generator[self.trial_counter]))
                #print(self.trial_counter)
                #print(self.reward_generator[:100])

                if self.reward_generator[self.trial_counter] > 0:
                    self.reward_port.open()
                    #rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.last_targ_reward[1])+' sec\n']
                    rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.reward_generator[self.trial_counter])+' sec\n']
                    self.reward_port.write(rew_str)
                    time.sleep(.25 + self.reward_delay_time)
                    run_str = [ord(r) for r in 'run\n']
                    self.reward_port.write(run_str)
                    self.reward_port.close()
        except:
            pass
        
    def run_button_rew(self, **kwargs):
        try:
            ### To trigger reward make sure reward is > 0:
            if np.logical_or(self.button_rew[0], self.button_rew[1] > 0):

                self.reward_port.open()
                rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.button_rew[1])+' sec\n']
                self.reward_port.write(rew_str)
                time.sleep(.25)
                run_str = [ord(r) for r in 'run\n']
                self.reward_port.write(run_str)
                self.reward_port.close()
        except:
            pass

        #self.repeat = True

    def end_reward(self, **kwargs):
        self.pd_indicator_txt_color = (1.,1., 1., 1.)
        if self.use_white_screen:
            if len(self.cursor_ids)== 0:
                return True
        else:
            if self.cnts_in_rew > 30:
                return True
            else:
                self.cnts_in_rew += 1
                return False

    def end_touch_error(self, **kwargs):
        return kwargs['ts'] >= self.touch_error_timeout

    def end_timeout_error(self, **kwargs):
        return kwargs['ts'] >= self.timeout_error_timeout

    def end_hold_error(self, **kwargs):
        return kwargs['ts'] >= self.hold_error_timeout

    def end_drag_error(self, **kwargs):
        return kwargs['ts'] >= self.drag_error_timeout

    def touch_target(self, **kwargs):
        if self.drag_ok:
            return self.check_if_cursors_in_targ(self.active_target_position, self.eff_target_rad)
        else:
            return np.logical_and(self.check_if_cursors_in_targ(self.active_target_position, self.eff_target_rad),
                self.check_if_started_in_targ(self.active_target_position, self.eff_target_rad))

    def target_timeout(self, **kwargs):
        #return kwargs['ts'] > self.target_timeout_time
        if self.target_index == 1:
            if time.time() - self.first_time_for_this_targ_t0 > self.target1_timeout_time:
                self.repeat = False
                return True
            else:
                return False
        else:
            if time.time() - self.first_time_for_this_targ_t0 > self.target_timeout_time:
                self.repeat = False
                return True
            else:
                return False

    def finish_targ_hold(self, **kwargs):
        if not self.target_index == self.num_targets:
            if self.tht <= kwargs['ts']:
                # Play a small reward tone
                sound = SoundLoader.load('reward2.wav')
                sound.play()
                self.target_index += 1
                return True
            else:
                return False
        else:
            return False
        
    def finish_last_targ_hold(self, **kwargs):
        if self.target_index == self.num_targets:
            return self.tht <= kwargs['ts']
        else:
            return False

    def early_leave_target_hold(self, **kwargs):
        return not self.check_if_cursors_in_targ(self.active_target_position, self.eff_target_rad)

    def targ_drag_out(self, **kwargs):
        touch = self.touch
        self.touch = True
        stay_in = self.check_if_cursors_in_targ(self.active_target_position, self.eff_target_rad)
        self.touch = touch
        return not stay_in

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

class Splash(Widget):
    def init(self, *args):
        self.args = args
        if platform =='win32':
            from sound import Sound
            Sound.volume_max()

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
    # DEFINE THE DEFAULTS AS WHATEVER THEY WERE LAST TIME
    # animal name
    is_haribo = BooleanProperty(False)
    is_fifi = BooleanProperty(False)
    is_nike = BooleanProperty(False)
    is_butters = BooleanProperty(False)
    is_testing = BooleanProperty(False)
    try:
        if data_params['animal_name'] == 'haribo':
            is_haribo = BooleanProperty(True)
        elif data_params['animal_name'] == 'fifi':
            is_fifi = BooleanProperty(True)
        elif data_params['animal_name'] == 'nike':
            is_nike = BooleanProperty(True)
        elif data_params['animal_name'] == 'butters':
            is_butters = BooleanProperty(True)
        elif data_params['animal_name'] == 'testing':
            is_testing = BooleanProperty(True)
    except:
        pass
    
    # target 1 timeout
    try:
        is_t1tt0pt8 = BooleanProperty(False)
        is_t1tt1pt0 = BooleanProperty(False)
        is_t1tt1pt5 = BooleanProperty(False)
        is_t1tt2pt0 = BooleanProperty(False)
        is_t1tt2pt5 = BooleanProperty(False)
        is_t1tt3pt0 = BooleanProperty(False)
        is_t1tt3pt5 = BooleanProperty(False)
        is_t1tt4pt0 = BooleanProperty(False)
        is_t1tt10pt0 = BooleanProperty(False)

        if data_params['target1_timeout_time'] == 0.8:
            is_t1tt0pt8 = BooleanProperty(True)
        elif data_params['target1_timeout_time'] == 1.0:
            is_t1tt1pt0 = BooleanProperty(True)
        elif data_params['target1_timeout_time'] == 1.5:
            is_t1tt1pt5 = BooleanProperty(True)
        elif data_params['target1_timeout_time'] == 2.0:
            is_t1tt2pt0 = BooleanProperty(True)
        elif data_params['target1_timeout_time'] == 2.5:
            is_t1tt2pt5 = BooleanProperty(True)
        elif data_params['target1_timeout_time'] == 3.0:
            is_t1tt3pt0 = BooleanProperty(True)
        elif data_params['target1_timeout_time'] == 3.5:
            is_t1tt3pt5 = BooleanProperty(True)
        elif data_params['target1_timeout_time'] == 4.0:
            is_t1tt4pt0 = BooleanProperty(True)    
        elif data_params['target1_timeout_time'] == 10.0: 
            is_t1tt10pt0 = BooleanProperty(True)
    except:
        pass
    
    # target timeout
    is_tt0pt7 = BooleanProperty(False)
    is_tt0pt8 = BooleanProperty(False)
    is_tt1pt0 = BooleanProperty(False)
    is_tt1pt5 = BooleanProperty(False)
    is_tt2pt0 = BooleanProperty(False)
    is_tt2pt5 = BooleanProperty(False)
    is_tt3pt0 = BooleanProperty(False)
    is_tt3pt5 = BooleanProperty(False)
    is_tt4pt0 = BooleanProperty(False)
    try:
        if data_params['target_timeout_time'] == 0.7:
            is_tt0pt7 = BooleanProperty(True)
        elif data_params['target_timeout_time'] == 0.8:
            is_tt0pt8 = BooleanProperty(True)
        elif data_params['target_timeout_time'] == 1.0:
            is_tt1pt0 = BooleanProperty(True)
        elif data_params['target_timeout_time'] == 1.5:
            is_tt1pt5 = BooleanProperty(True)
        elif data_params['target_timeout_time'] == 2.0:
            is_tt2pt0 = BooleanProperty(True)
        elif data_params['target_timeout_time'] == 2.5:
            is_tt2pt5 = BooleanProperty(True)
        elif data_params['target_timeout_time'] == 3.0:
            is_tt3pt0 = BooleanProperty(True)
        elif data_params['target_timeout_time'] == 3.5:
            is_tt3pt5 = BooleanProperty(True)
        elif data_params['target_timeout_time'] == 4.0:
            is_tt4pt0 = BooleanProperty(True)
    except:
        pass
        
    # dragok
    is_dragok = BooleanProperty(False)
    is_dragnotok = BooleanProperty(False)
    try:
        if data_params['drag_ok'] is True:
            is_dragok = BooleanProperty(True)
        elif data_params['drag_ok'] is False:
            is_dragnotok = BooleanProperty(True)
    except:
        pass
    
    # crashbar hold time
    is_bhtfalse = BooleanProperty(False)
    is_bht000 = BooleanProperty(False)
    is_bht100 = BooleanProperty(False)
    is_bht200 = BooleanProperty(False)
    is_bht300 = BooleanProperty(False)
    is_bht400 = BooleanProperty(False)
    is_bht500 = BooleanProperty(False)
    is_bht600 = BooleanProperty(False)
    is_bht700 = BooleanProperty(False)
    is_bht800 = BooleanProperty(False)
    is_bht900 = BooleanProperty(False)
    is_bht1000 = BooleanProperty(False)
    is_bht200to400 = BooleanProperty(False)
    is_bht600to800 = BooleanProperty(False)
    is_bht800to1000 = BooleanProperty(False)
    # is_bhtbigrand = BooleanProperty(False)
    try:
        if data_params['button_hold_time'] == False:
            is_bhtfalse = BooleanProperty(True)
        elif data_params['button_hold_time'] == 0.0:
            is_bht000 = BooleanProperty(True)
        elif data_params['button_hold_time'] == 0.1:
            is_bht100 = BooleanProperty(True)
        elif data_params['button_hold_time'] == 0.2:
            is_bht200 = BooleanProperty(True)
        elif data_params['button_hold_time'] == 0.3:
            is_bht300 = BooleanProperty(True)
        elif data_params['button_hold_time'] == 0.4:
            is_bht400 = BooleanProperty(True)
        elif data_params['button_hold_time'] == 0.5:
            is_bht500 = BooleanProperty(True)
        elif data_params['button_hold_time'] == 0.6:
            is_bht600 = BooleanProperty(True)
        elif data_params['button_hold_time'] == 0.7:
            is_bht700 = BooleanProperty(True)
        elif data_params['button_hold_time'] == 0.8:
            is_bht800 = BooleanProperty(True)
        elif data_params['button_hold_time'] == 0.9:
            is_bht900 = BooleanProperty(True)
        elif data_params['button_hold_time'] == 1.0:
            is_bht1000 = BooleanProperty(True)
        elif data_params['button_hold_time'] == '.2-.4': 
            is_bht200to400 = BooleanProperty(True)
        elif data_params['button_hold_time'] == '.6-.8':
            is_bht600to800 = BooleanProperty(True)
        elif data_params['button_hold_time'] == '.8-1.0':
            is_bht800to1000 = BooleanProperty(True)
    except:
        pass
        
    # crashbar reward
    is_bhrew000 = BooleanProperty(False)
    is_bhrew100 = BooleanProperty(False)
    is_bhrew300 = BooleanProperty(False)
    is_bhrew500 = BooleanProperty(False)
    try:
        if data_params['button_rew'] == 0.0:
            is_bhrew000 = BooleanProperty(True)
        elif data_params['button_rew'] == 0.1:
            is_bhrew100 = BooleanProperty(True)
        elif data_params['button_rew'] == 0.3:
            is_bhrew300 = BooleanProperty(True)
        elif data_params['button_rew'] == 0.5:
            is_bhrew500 = BooleanProperty(True)
    except:
        pass
        
    # target hold time
    is_tht000 = BooleanProperty(False)
    is_tht100 = BooleanProperty(False)
    is_tht200 = BooleanProperty(False)
    is_tht300 = BooleanProperty(False)
    is_tht400 = BooleanProperty(False)
    is_tht500 = BooleanProperty(False)
    is_tht600 = BooleanProperty(False)
    is_tht100to300 = BooleanProperty(False)
    is_tht400to600 = BooleanProperty(False)
    try:
        if data_params['target_hold_time'] == 0.0:
            is_tht000 = BooleanProperty(True)
        elif data_params['target_hold_time'] == 0.1:
            is_tht100 = BooleanProperty(True)
        elif data_params['target_hold_time'] == 0.2:
            is_tht200 = BooleanProperty(True)
        elif data_params['target_hold_time'] == 0.3:
            is_tht300 = BooleanProperty(True)
        elif data_params['target_hold_time'] == 0.4:
            is_tht400 = BooleanProperty(True)
        elif data_params['target_hold_time'] == 0.5:
            is_tht500 = BooleanProperty(True)
        elif data_params['target_hold_time'] == 0.6:
            is_tht600 = BooleanProperty(True)
        elif data_params['target_hold_time'] == '.1-.3':
            is_tht100to300 = BooleanProperty(True)
        elif data_params['target_hold_time'] == '.4-.6':
            is_tht400to600 = BooleanProperty(True)
    except:
        pass
        
    # final target reward
    is_threw000 = BooleanProperty(False)
    is_threw300 = BooleanProperty(False)
    is_threw500 = BooleanProperty(False)
    is_threw700 = BooleanProperty(False)
    try:
        if data_params['last_targ_reward'] == 0.0:
            is_threw000 = BooleanProperty(True)
        elif data_params['last_targ_reward'] == 0.3:
            is_threw300 = BooleanProperty(True)
        elif data_params['last_targ_reward'] == 0.5:
            is_threw500 = BooleanProperty(True)
        elif data_params['last_targ_reward'] == 0.7:
            is_threw700 = BooleanProperty(True)
    except:
        pass
        
    # reward variability
    is_rewvarall = BooleanProperty(False)
    is_rewvar50 = BooleanProperty(False)
    is_rewvar33 = BooleanProperty(False)
    try:
        if data_params['percent_of_trials_rewarded'] == 1.0:
            is_rewvarall = BooleanProperty(True)
        elif data_params['percent_of_trials_rewarded'] == 0.5:
            is_rewvar50 = BooleanProperty(True)
        elif data_params['percent_of_trials_rewarded'] == 0.33:
            is_rewvar33 = BooleanProperty(True)
    except:
        pass
        
    # target radius
    is_trad050 = BooleanProperty(False)
    is_trad075 = BooleanProperty(False)
    is_trad082 = BooleanProperty(False)
    is_trad091 = BooleanProperty(False)
    is_trad100 = BooleanProperty(False)
    is_trad150 = BooleanProperty(False)
    is_trad185 = BooleanProperty(False)
    is_trad225 = BooleanProperty(False)
    is_trad300 = BooleanProperty(False)
    is_trad400 = BooleanProperty(False)
    try:
        if data_params['target_rad'] == 0.5:
            is_trad050 = BooleanProperty(True)
        elif data_params['target_rad'] == 0.75:
            is_trad075 = BooleanProperty(True)
        elif data_params['target_rad'] == 0.82:
            is_trad082 = BooleanProperty(True)
        elif data_params['target_rad'] == 0.91:
            is_trad091 = BooleanProperty(True)
        elif data_params['target_rad'] == 1.0:
            is_trad100 = BooleanProperty(True)
        elif data_params['target_rad'] == 1.5:
            is_trad150 = BooleanProperty(True)
        elif data_params['target_rad'] == 1.85:
            is_trad185 = BooleanProperty(True)
        elif data_params['target_rad'] == 2.25:
            is_trad225 = BooleanProperty(True)
        elif data_params['target_rad'] == 3.0:
            is_trad300 = BooleanProperty(True)
        elif data_params['target_rad'] == 4.0:
            is_trad400 = BooleanProperty(True)
    except:
        pass
        
    # effective target radius
    is_efftradsame = BooleanProperty(False)
    is_efftrad10 = BooleanProperty(False)
    is_efftrad20 = BooleanProperty(False)
    is_efftrad30 = BooleanProperty(False)
    is_efftrad40 = BooleanProperty(False)
    is_efftrad50 = BooleanProperty(False)

    try:
        if data_params['effective_target_rad'] == data_params['target_rad']:
            is_efftradsame = BooleanProperty(True)
        elif data_params['effective_target_rad'] == 1.0:
            is_efftrad10 = BooleanProperty(True)
        elif data_params['effective_target_rad'] == 2.0:
            is_efftrad20 = BooleanProperty(True)
        elif data_params['effective_target_rad'] == 3.0:
            is_efftrad30 = BooleanProperty(True)
        elif data_params['effective_target_rad'] == 4.0:
            is_efftrad40 = BooleanProperty(True)
        elif data_params['effective_target_rad'] == 5.0:
            is_efftrad50 = BooleanProperty(True)
    except: 
        pass
        
    # sequence preselect
    is_seq_straight_r = BooleanProperty(False)
    is_seq_r_up = BooleanProperty(False)
    try:
        if data_params['seq'] == 'Straight_Right':
            is_seq_straight_r = BooleanProperty(True)
        elif data_params['seq'] == 'Right_Up':
            is_seq_r_up = BooleanProperty(True)
    except: 
        pass

    # target 1 position
    is_stcent = BooleanProperty(False)
    is_stum = BooleanProperty(False)
    is_stlm = BooleanProperty(False)
    is_stul = BooleanProperty(False)
    is_stml = BooleanProperty(False)
    is_stll = BooleanProperty(False)
    is_stur = BooleanProperty(False)
    is_stmr = BooleanProperty(False)
    is_stlr = BooleanProperty(False)
    is_strand = BooleanProperty(False) 
    try:
        if data_params['start_pos_str'] == 'center':
            is_stcent = BooleanProperty(True)
        elif data_params['start_pos_str'] == 'upper_middle':
            is_stum = BooleanProperty(True)
        elif data_params['start_pos_str'] == 'lower_middle':
            is_stlm = BooleanProperty(True) 
        elif data_params['start_pos_str'] == 'upper_left':
            is_stul = BooleanProperty(True)
        elif data_params['start_pos_str'] == 'middle_left':
            is_stml = BooleanProperty(True)
        elif data_params['start_pos_str'] == 'lower_left':
            is_stll = BooleanProperty(True)
        elif data_params['start_pos_str'] == 'upper_right':
            is_stur = BooleanProperty(True)
        elif data_params['start_pos_str'] == 'middle_right':
            is_stmr = BooleanProperty(True)
        elif data_params['start_pos_str'] == 'lower_right':
            is_stlr = BooleanProperty(True)
        elif data_params['start_pos_str'] == 'random': 
            is_strand = BooleanProperty(True)
    except:
        pass
        
    # target 1 nudge
    is_nudgeneg6 = BooleanProperty(False)
    is_nudgeneg4 = BooleanProperty(False)
    is_nudgeneg2 = BooleanProperty(False)
    is_nudgezero = BooleanProperty(False)
    is_nudgepos2 = BooleanProperty(False)
    is_nudgepos4 = BooleanProperty(False)
    is_nudgepos6 = BooleanProperty(False)
    try:
        if data_params['nudge_x'] == -6:
            is_nudgeneg6 = BooleanProperty(True)
        elif data_params['nudge_x'] == -4:
            is_nudgeneg4 = BooleanProperty(True)
        elif data_params['nudge_x'] == -2:
            is_nudgeneg2 = BooleanProperty(True)
        elif data_params['nudge_x'] == 0:
            is_nudgezero = BooleanProperty(True)
        elif data_params['nudge_x'] == 2:
            is_nudgepos2 = BooleanProperty(True)
        elif data_params['nudge_x'] == 4:
            is_nudgepos4 = BooleanProperty(True)
        elif data_params['nudge_x'] == 6:
            is_nudgepos6 = BooleanProperty(True)
    except:
        pass
        
    # lower screen top by
    is_screentopzero = BooleanProperty(False)
    is_screentop2 = BooleanProperty(False)
    is_screentop4 = BooleanProperty(False)
    is_screentop6 = BooleanProperty(False)
    is_screentop8 = BooleanProperty(False)
    is_screentop10 = BooleanProperty(False)
    is_screentop12 = BooleanProperty(False)
    try:
        if data_params['screen_top'] == 0:
            is_screentopzero = BooleanProperty(True)
        elif data_params['screen_top'] == -2:
            is_screentop2 = BooleanProperty(True)
        elif data_params['screen_top'] == -4:
            is_screentop4 = BooleanProperty(True)
        elif data_params['screen_top'] == -6:
            is_screentop6 = BooleanProperty(True)
        elif data_params['screen_top'] == -8:
            is_screentop8 = BooleanProperty(True)
        elif data_params['screen_top'] == -10:
            is_screentop10 = BooleanProperty(True)
        elif data_params['screen_top'] == -12:
            is_screentop12 = BooleanProperty(True)
    except:
        pass
        
    # raise screen bottom by
    try:
        is_screenbot0 = BooleanProperty(False)
        is_screenbot2 = BooleanProperty(False)
        is_screenbot4 = BooleanProperty(False)
        is_screenbot6 = BooleanProperty(False)
        is_screenbot8 = BooleanProperty(False)
        is_screenbot10 = BooleanProperty(False)
        is_screenbot12 = BooleanProperty(False)
        if data_params['screen_bot'] == 0:
            is_screenbot0 = BooleanProperty(True)
        elif data_params['screen_bot'] == 2:
            is_screenbot2 = BooleanProperty(True)
        elif data_params['screen_bot'] == 4:
            is_screenbot4 = BooleanProperty(True)
        elif data_params['screen_bot'] == 6:
            is_screenbot6 = BooleanProperty(True)
        elif data_params['screen_bot'] == 8:
            is_screenbot8 = BooleanProperty(True)
        elif data_params['screen_bot'] == 10:
            is_screenbot10 = BooleanProperty(True)
        elif data_params['screen_bot'] == 12:
            is_screenbot12 = BooleanProperty(True)
    except:
        pass
        
    # time until next target appears
    is_ttntnever = BooleanProperty(False)
    is_ttnt025 = BooleanProperty(False)
    is_ttnt050 = BooleanProperty(False)
    is_ttnt075 = BooleanProperty(False)
    is_ttnt100 = BooleanProperty(False)
    is_ttnt150 = BooleanProperty(False)
    try:
        if data_params['time_to_next_targ'] == False:
            is_ttntnever = BooleanProperty(True)
        elif data_params['time_to_next_targ'] == 0.25:
            is_ttnt025 = BooleanProperty(True)
        elif data_params['time_to_next_targ'] == 0.5:
            is_ttnt050 = BooleanProperty(True)
        elif data_params['time_to_next_targ'] == 0.75:
            is_ttnt075 = BooleanProperty(True)
        elif data_params['time_to_next_targ'] == 1.0:
            is_ttnt100 = BooleanProperty(True)
        elif data_params['time_to_next_targ'] == 1.5:
            is_ttnt150 = BooleanProperty(True)
    except:
        pass
    
    # auto quit after
    is_autoqt10 = BooleanProperty(False)
    is_autoqt25 = BooleanProperty(False)
    is_autoqt50 = BooleanProperty(False)
    is_autoqt100 = BooleanProperty(False)
    is_autoqtnever = BooleanProperty(False)
    try:
        if data_params['max_trials'] == 10:
            is_autoqt10 = BooleanProperty(True)
        elif data_params['max_trials'] == 25:
            is_autoqt25 = BooleanProperty(True)
        elif data_params['max_trials'] == 50:
            is_autoqt50 = BooleanProperty(True)
        elif data_params['max_trials'] == 100:
            is_autoqt100 = BooleanProperty(True)
        elif data_params['max_trials'] == 10**10:
            is_autoqtnever = BooleanProperty(True)
    except:
        pass

class COApp(App):
    def build(self, **kwargs):
        Window.size = (fixed_window_size[0], fixed_window_size[1])
        Window.left = 0
        Window.top = 0
        if platform == 'darwin':
            Window.fullscreen = 'auto'
        
        return Manager()

def cm2pix(pos_cm, fixed_window_size_cm=fixed_window_size_cm):
    pix_per_cm = Window.width/fixed_window_size_cm[0]
    
    # Convert from CM to pixels: 
    pix_pos = pix_per_cm*pos_cm

    if type(pix_pos) is np.ndarray:
        # Translate to coordinate system w/ 0, 0 at bottom left
        pix_pos[0] = pix_pos[0] + (Window.width/2.)
        pix_pos[1] = pix_pos[1] + (Window.height/2.)
        # pix_pos[0] = pix_pos[0] + (fixed_window_size[0]/2.)
        # pix_pos[1] = pix_pos[1] + (fixed_window_size[1]/2.)

    return pix_pos

def pix2cm(pos_pix, fixed_window_size_cm=fixed_window_size_cm):
    pix_per_cm = Window.width/fixed_window_size_cm[0]
    
    # First shift coordinate system: 
    pos_pix[0] = pos_pix[0] - (Window.width/2.)
    pos_pix[1] = pos_pix[1] - (Window.height/2.)

    pos_cm = pos_pix*(1./pix_per_cm)
    return pos_cm

if __name__ == '__main__':
    COApp().run()
