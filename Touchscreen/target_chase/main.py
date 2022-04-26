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
import scipy.io as io


Config.set('graphics', 'resizable', False)

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

if platform == 'darwin': # we are on a Mac
    # This probably means that we are testing on a personal laptop
    
    # settings for MBP 16" 2021
    fixed_window_size = (3072, 1920) # we get this automatically now but here it is anyway
    fixed_window_size_cm = (34.5, 21.5) # this is the important part
    pix_per_cm = 104. # we get this automatically now but here it is anyway
elif platform == 'win32':
        # see if there is an external monitor plugged in
    if user_id == 'BasalGangulia':
        fixed_window_size = (2160, 1440) # we get this automatically now but here it is anyway
        fixed_window_size_cm = (47.6, 26.8)
#        fixed_window_size_cm = (22.8, 15.2) # this is the important part
        pix_per_cm = 95. # we get this automatically now but here it is anyway
    else:
        from screeninfo import get_monitors
        mon = get_monitors()
        if len(get_monitors()) > 1 or get_monitors()[0].height == 1080:
            # must be a1n external monitor plugged in
            i_td2230 = False
            for i in range(len(mon)):
                if mon[i].height_mm == 268 and mon[i].width_mm == 477:
                    # assume it is viewsonic TD2230
                    i_td2230 = i
            if not i_td2230:
                i_mon = i
            else:
                i_mon = i_td2230
        else:
            # must just be the surface pro
            i_mon = 0
        fixed_window_size = (mon[i_mon].width, mon[i_mon].height) # we get this automatically now but here it is anyway
        fixed_window_size_cm = (mon[i_mon].width_mm/10, mon[i_mon].height_mm/10) # this is the important part
        pix_per_cm = np.round(10*np.min([mon[i_mon].width/mon[i_mon].width_mm, mon[i_mon].height/mon[i_mon].height_mm]))
    fixed_window_size = (2160, 1440) # we get this automatically now but here it is anyway
    fixed_window_size_cm = (47.6, 26.8)
#        fixed_window_size_cm = (22.8, 15.2) # this is the important part
    pix_per_cm = 95. # we get this automatically now but here it is anyway
    import winsound

Config.set('graphics', 'width', str(fixed_window_size[0]))
Config.set('graphics', 'height', str(fixed_window_size[1]))

# LOAD THE MOST RECENT PARMS TO USE AS DEFAULTS
if user_id == 'Sandon':
    last_param_path = '/Users/Sandon/Documents/'
elif user_id == 'Ganguly':
    last_param_path = 'C:/Users/Ganguly/Documents/'
elif user_id == 'BasalGangulia':
    last_param_path = 'C:/Users/BasalGangulia/Documents/'

last_param_path = last_param_path+'most_recent_target_chase_params.pkl'
if os.path.exists(last_param_path):
    with open(last_param_path, 'rb') as f:
        data_params = pickle.load(f)
        

class Data(tables.IsDescription):
    state = tables.StringCol(24)   # 24-character String
    cursor = tables.Float32Col(shape=(10, 2))
    cursor_ids = tables.Float32Col(shape = (10, ))
    target_pos = tables.Float32Col(shape=(2, ))
    button_state = tables.Float32Col()
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
    exit_pos_y = (fixed_window_size_cm[1]/2)-2.5
    exit_pos = np.array([exit_pos_x, exit_pos_y])
    
    
    pd1_ind_pos_x = (fixed_window_size_cm[0]/2)-0.3
    pd1_ind_pos_y = (fixed_window_size_cm[1]/2)-0.5
    pd1_indicator_pos = np.array([pd1_ind_pos_x, pd1_ind_pos_y])
    pd2_ind_pos_x = (fixed_window_size_cm[0]/2)-1.8
    pd2_ind_pos_y = (fixed_window_size_cm[1]/2)-0.5
    pd2_indicator_pos = np.array([pd2_ind_pos_x, pd2_ind_pos_y])
    
    vid_ind_pos_x = (fixed_window_size_cm[0]/2)-0.5
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
    target3 = ObjectProperty(None)
    target4 = ObjectProperty(None)
    target5 = ObjectProperty(None)
    
    target1_out = ObjectProperty(None)
    target1_in = ObjectProperty(None)
    target2_out = ObjectProperty(None)
    target2_in = ObjectProperty(None)
    target3_out = ObjectProperty(None)
    target3_in = ObjectProperty(None)
    target4_out = ObjectProperty(None)
    target4_in = ObjectProperty(None)
    target5_out = ObjectProperty(None)
    target5_in = ObjectProperty(None)
    target6_out = ObjectProperty(None)
    target6_in = ObjectProperty(None)
    target7_out = ObjectProperty(None)
    target7_in = ObjectProperty(None)
    target8_out = ObjectProperty(None)
    target8_in = ObjectProperty(None)
    target9_out = ObjectProperty(None)
    target9_in = ObjectProperty(None)

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
    
    # Load possible sequences
    seq_poss = io.loadmat('seq_poss.mat')['seq_poss']
        
    
    def on_touch_down(self, touch):
        if self.testing:
            self.tic_on_touch_down_start = time.time()
        
        #handle many touchs:
        ud = touch.ud

        # Add new touch to ids: 
        self.cursor_ids.append(touch.uid)

        # Add cursor
        curs = pix2cm(np.array([touch.x, touch.y]))
        
        # ignore touching around the photodiode
        if len(curs.shape) == 1:
            if np.linalg.norm(np.array(curs) - np.array([self.pd1_ind_pos_x, self.pd1_ind_pos_y])) < 2*self.exit_rad:
                curs = np.array([np.nan, np.nan])
            elif np.linalg.norm(np.array(curs) - np.array([self.pd2_ind_pos_x, self.pd2_ind_pos_y])) < 2*self.exit_rad:
                curs = np.array([np.nan, np.nan])
        elif len(curs.shape) > 1:
            for i in range(curs.shape[1]):
                if np.linalg.norm(np.array(curs[i,:]) - np.array([self.pd1_ind_pos_x, self.pd1_ind_pos_y])) < 2*self.exit_rad:
                    curs[i,:] = np.array([np.nan, np.nan])
                elif np.linalg.norm(np.array(curs[i,:]) - np.array([self.pd2_ind_pos_x, self.pd2_ind_pos_y])) < 2*self.exit_rad:
                    curs[i,:] = np.array([np.nan, np.nan])
        self.cursor[touch.uid] =  curs.copy()
        self.cursor_start[touch.uid] = curs.copy()

        # set self.touch to True
        self.touch = True
        
        if self.testing:
            self.tic_on_touch_down_end = time.time()
            
        

    def on_touch_move(self, touch):
        curs = pix2cm(np.array([touch.x, touch.y]))
        # ignore touching around the photodiode
        if len(curs.shape) == 1:
            if np.linalg.norm(np.array(curs) - np.array([self.pd1_ind_pos_x, self.pd1_ind_pos_y])) < 2*self.exit_rad:
                curs = np.array([np.nan, np.nan])
            elif np.linalg.norm(np.array(curs) - np.array([self.pd2_ind_pos_x, self.pd2_ind_pos_y])) < 2*self.exit_rad:
                curs = np.array([np.nan, np.nan])
        elif len(curs.shape) > 1:
            for i in range(curs.shape[1]):
                if np.linalg.norm(np.array(curs[i,:]) - np.array([self.pd1_ind_pos_x, self.pd1_ind_pos_y])) < 2*self.exit_rad:
                    curs[i,:] = np.array([np.nan, np.nan])
                elif np.linalg.norm(np.array(curs[i,:]) - np.array([self.pd2_ind_pos_x, self.pd2_ind_pos_y])) < 2*self.exit_rad:
                    curs[i,:] = np.array([np.nan, np.nan])
        
        self.cursor[touch.uid] =  curs.copy()
        
        self.touch = True

    def on_touch_up(self, touch):
        try:
            self.cursor_ids.remove(touch.uid)
            _ = self.cursor.pop(touch.uid)
        except:
            print('removing touch from pre-game screen')
            
    def init(self, animal_names_dict=None, rew_in=None, task_in=None,
        hold=None, autoquit=None, # rew_var=None, 
        targ_timeout = None, drag=None, button=None, # nudge_x=None, 
        screen_size=None, juicer=None, taskbreak=None):
        
        self.rew_cnt = 0

        # JUICER VERSION
        juicer_opts = ['yellow', 'red']
        for i, val in enumerate(juicer['juicer']): 
            if val: 
                self.juicer = juicer_opts[i]


        # TARGET TIMEOUT
        targ1_timeout_opts = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 10.0]
        self.target1_timeout_time = 10000
        for i, val in enumerate(targ_timeout['t1tt']):
            if val:
                self.target1_timeout_time = targ1_timeout_opts[i]
        
        targ_timeout_opts = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        self.target_timeout_time = 10000
        for i, val in enumerate(targ_timeout['tt']):
            if val:
                self.target_timeout_time = targ_timeout_opts[i]

        # NUDGE X
        # nudge_x_opts = [-6, -4, -2, 0, 2, 4, 6]    
        self.nudge_x_t1 = 0
        # for i, val in enumerate(nudge_x['nudge_x_t1']):
        #     if val:
        #         self.nudge_x_t1 = nudge_x_opts[i]
                
        self.nudge_x_t2 = 0
        # for i, val in enumerate(nudge_x['nudge_x_t2']):
        #     if val:
        #         self.nudge_x_t2 = nudge_x_opts[i]
        
        self.nudge_x_t3 = 0
        # for i, val in enumerate(nudge_x['nudge_x_t3']):
        #     if val:
        #         self.nudge_x_t3 = nudge_x_opts[i]
                
        self.nudge_x_t4 = 0       
        # for i, val in enumerate(nudge_x['nudge_x_t4']):
        #     if val:
        #         self.nudge_x_t4 = nudge_x_opts[i]
        
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
        self.center_position = np.array([0., 0.])
        # lower the center position by half of the total amount the screen height has been shrunk by
        self.center_position[1] = self.center_position[1] + self.screen_top/2 + self.screen_bot/2
        
        d_center2top = (fixed_window_size_cm[1]/2)-((self.screen_top/2)+(self.screen_bot/2))
        d_center2bot = (fixed_window_size_cm[1]/2)+((self.screen_top/2)+(self.screen_bot/2))
        self.max_y_from_center = (fixed_window_size_cm[1]+self.screen_top-self.screen_bot)/2-self.target_rad
        
        
        # seq_opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'center out', 'button out']
        seq_opts = ['Y', 'rand5', 'repeat', 'randevery', 'rand5-randevery', '2seq-repeat', 'center out', 'button out']
        self.seq = False
        for i, val in enumerate(task_in['seq']):
            if val:
                self.seq = seq_opts[i]
        
        if self.seq == 'A':
            seq_preselect = True
            self.target1_pos_str = 'lower_right'
            self.target2_pos_str = 'upper_right'
            self.target3_pos_str = 'upper_left'
            self.target4_pos_str = 'lower_left'
            self.target5_pos_str = 'none'
        elif self.seq == 'B':
            seq_preselect = True
            self.target1_pos_str = 'center'
            self.target2_pos_str = 'middle_left'
            self.target3_pos_str = 'middle_right'
            self.target4_pos_str = 'upper_left'
            self.target5_pos_str = 'none'
        elif self.seq == 'C':
            seq_preselect = True
            self.target1_pos_str = 'upper_middle'
            self.target2_pos_str = 'lower_left'
            self.target3_pos_str = 'middle_right'
            self.target4_pos_str = 'upper_left'
            self.target5_pos_str = 'none'
        elif self.seq == 'D':
            seq_preselect = True
            self.target1_pos_str = 'middle_left'
            self.target2_pos_str = 'upper_right'
            self.target3_pos_str = 'lower_middle'
            self.target4_pos_str = 'upper_left'
            self.target5_pos_str = 'none'
        elif self.seq == 'E':
            seq_preselect = True
            self.target1_pos_str = 'upper_middle'
            self.target2_pos_str = 'middle_left'
            self.target3_pos_str = 'middle_right'
            self.target4_pos_str = 'lower_middle'
            self.target5_pos_str = 'upper_middle'
            
        elif self.seq == 'F':
            seq_preselect = True
            self.target1_pos_str = 'lower_middle'
            self.target2_pos_str = 'upper_middle'
            self.target3_pos_str = 'middle_right'
            self.target4_pos_str = 'middle_left'
            self.target5_pos_str = 'lower_middle'
        
        elif self.seq == 'G':
            seq_preselect = True
            self.target1_pos_str = 'middle_left'
            self.target2_pos_str = 'middle_right'
            self.target3_pos_str = 'lower_middle'
            self.target4_pos_str = 'upper_middle'
            self.target5_pos_str = 'middle_right'
            
        elif self.seq == 'H':
            seq_preselect = True
            self.target1_pos_str = 'middle_right'
            self.target2_pos_str = 'middle_left'
            self.target3_pos_str = 'lower_middle'
            self.target4_pos_str = 'upper_middle'
            self.target5_pos_str = 'middle_left'
            
        elif self.seq == 'I':
            seq_preselect = True
            self.target1_pos_str = 'lower_left'
            self.target2_pos_str = 'upper_middle'
            self.target3_pos_str = 'lower_right'
            self.target4_pos_str = 'upper_right'
            self.target5_pos_str = 'center'
            
        elif self.seq == 'J':
            seq_preselect = True
            self.target1_pos_str = 'lower_middle'
            self.target2_pos_str = 'upper_right'
            self.target3_pos_str = 'lower_right'
            self.target4_pos_str = 'lower_left'
            self.target5_pos_str = 'upper_middle'
            
        elif self.seq == 'K':
            seq_preselect = True
            self.target1_pos_str = 'upper_middle'
            self.target2_pos_str = 'middle_right'
            self.target3_pos_str = 'lower_left'
            self.target4_pos_str = 'upper_left'
            self.target5_pos_str = 'upper_right'
            
        elif self.seq == 'L':
            seq_preselect = True
            self.target1_pos_str = 'middle_left'
            self.target2_pos_str = 'upper_right'
            self.target3_pos_str = 'center'
            self.target4_pos_str = 'lower_right'
            self.target5_pos_str = 'upper_left'
            
        elif self.seq == 'M':
            seq_preselect = True
            self.target1_pos_str = 'center'
            self.target2_pos_str = 'upper_left'
            self.target3_pos_str = 'middle_right'
            self.target4_pos_str = 'lower_middle'
            self.target5_pos_str = 'upper_right'
            
        elif self.seq == 'N':
            seq_preselect = True
            self.target1_pos_str = 'lower_left'
            self.target2_pos_str = 'center'
            self.target3_pos_str = 'upper_left'
            self.target4_pos_str = 'upper_right'
            self.target5_pos_str = 'lower_right'
            
        elif self.seq == 'O':
            seq_preselect = True
            self.target1_pos_str = 'middle_right'
            self.target2_pos_str = 'upper_left'
            self.target3_pos_str = 'lower_middle'
            self.target4_pos_str = 'center'
            self.target5_pos_str = 'upper_right'
            
        elif self.seq == 'P':
            seq_preselect = True
            self.target1_pos_str = 'upper_right'
            self.target2_pos_str = 'center'
            self.target3_pos_str = 'lower_left'
            self.target4_pos_str = 'upper_middle'
            self.target5_pos_str = 'middle_right'
            
        elif self.seq == 'Q':
            seq_preselect = True
            self.target1_pos_str = 'upper_left'
            self.target2_pos_str = 'lower_left'
            self.target3_pos_str = 'middle_right'
            self.target4_pos_str = 'center'
            self.target5_pos_str = 'lower_middle'
        
        elif self.seq == 'R':
            seq_preselect = True
            self.target1_pos_str = 'middle_right'
            self.target2_pos_str = 'upper_middle'
            self.target3_pos_str = 'middle_left'
            self.target4_pos_str = 'lower_middle'
            self.target5_pos_str = 'upper_left'
            
        elif self.seq == 'S':
            seq_preselect = True
            self.target1_pos_str = 'center'
            self.target2_pos_str = 'lower_right'
            self.target3_pos_str = 'upper_middle'
            self.target4_pos_str = 'middle_left'
            self.target5_pos_str = 'lower_left'
        
        elif self.seq == 'T':
            seq_preselect = True
            self.target1_pos_str = 'upper_left'
            self.target2_pos_str = 'center'
            self.target3_pos_str = 'lower_left'
            self.target4_pos_str = 'middle_right'
            self.target5_pos_str = 'upper_middle'
            
        elif self.seq == 'U':
            seq_preselect = True
            self.target1_pos_str = 'middle_left'
            self.target2_pos_str = 'upper_right'
            self.target3_pos_str = 'upper_left'
            self.target4_pos_str = 'lower_middle'
            self.target5_pos_str = 'center'
            
        elif self.seq == 'V':
            seq_preselect = True
            self.target1_pos_str = 'middle_right'
            self.target2_pos_str = 'upper_middle'
            self.target3_pos_str = 'lower_left'
            self.target4_pos_str = 'center'
            self.target5_pos_str = 'upper_left'
            
        elif self.seq == 'W':
            seq_preselect = True
            self.target1_pos_str = 'upper_right'
            self.target2_pos_str = 'middle_left'
            self.target3_pos_str = 'upper_left'
            self.target4_pos_str = 'lower_middle'
            self.target5_pos_str = 'middle_right'
            
        elif self.seq == 'X':
            seq_preselect = True
            self.target1_pos_str = 'center'
            self.target2_pos_str = 'lower_right'
            self.target3_pos_str = 'upper_right'
            self.target4_pos_str = 'upper_left'
            self.target5_pos_str = 'middle_right'
            
        elif self.seq == 'Y':
            seq_preselect = True
            self.target1_pos_str = 'upper_middle'
            self.target2_pos_str = 'lower_right'
            self.target3_pos_str = 'middle_left'
            self.target4_pos_str = 'upper_left'
            self.target5_pos_str = 'lower_middle'
            
        elif self.seq == 'rand5':
            seq_preselect = True
            self.target1_pos_str = 'random'
            self.target2_pos_str = 'random'
            self.target3_pos_str = 'random'
            self.target4_pos_str = 'random'
            self.target5_pos_str = 'random'
            self.make_random_sequence(True)
            # pos_str_opts = ['upper_left', 'upper_middle', 'upper_right', 'middle_left', 'center', 'middle_right', 'lower_left', 'lower_middle', 'lower_right']
            # pos_order = np.random.permutation(9)
            # self.target1_pos_str = pos_str_opts[pos_order[0]]
            # self.target2_pos_str = pos_str_opts[pos_order[1]]
            # self.target3_pos_str = pos_str_opts[pos_order[2]]
            # self.target4_pos_str = pos_str_opts[pos_order[3]]
            # self.target5_pos_str = pos_str_opts[pos_order[4]]
            
        elif self.seq == 'repeat' or self.seq == '2seq-repeat':
            seq_preselect = True
            self.target1_pos_str = data_params['target1_pos_str']
            self.target2_pos_str = data_params['target2_pos_str']
            self.target3_pos_str = data_params['target3_pos_str']
            self.target4_pos_str = data_params['target4_pos_str']
            self.target5_pos_str = data_params['target5_pos_str']
            
        elif self.seq == 'randevery':
            seq_preselect = True
            self.target1_pos_str = 'random'
            self.target2_pos_str = 'random'
            self.target3_pos_str = 'random'
            self.target4_pos_str = 'random'
            self.target5_pos_str = 'random'
        
        elif self.seq == 'center out':
            seq_preselect = True
            self.target1_pos_str = 'center'
            self.target2_pos_str = 'random'
            self.target3_pos_str = 'none'
            self.target4_pos_str = 'none'
            self.target5_pos_str = 'none'
            self.trial_order = self.gen_trials(self.seq) 
            self.num_targets = 1
        
        elif self.seq == 'button out':
            seq_preselect = True
            self.target1_pos_str = 'random'
            self.target2_pos_str = 'none'
            self.target3_pos_str = 'none'
            self.target4_pos_str = 'none'
            self.target5_pos_str = 'none'
            self.trial_order = self.gen_trials(self.seq)
            self.num_targets = 1
        
        else:
            seq_preselect = False
        
        if self.seq == 'rand5-randevery' \
        or (self.seq == 'repeat' and data_params['seq'] == 'rand5-randevery') \
        or self.seq == '2seq-repeat':
            seq_preselect = True
            if self.seq == 'repeat' and data_params['seq'] == 'rand5-randevery':
                self.seq = 'rand5-randevery'
            elif self.seq == '2seq-repeat':
                self.seq = '2seq-repeat'
                self.seq2generated = False
            else:
                self.target1_pos_str = 'random'
                self.target2_pos_str = 'random'
                self.target3_pos_str = 'random'
                self.target4_pos_str = 'random'
                self.target5_pos_str = 'random'
                self.make_random_sequence(True)
            self.target1_pos_str_og = self.target1_pos_str
            self.target2_pos_str_og = self.target2_pos_str
            self.target3_pos_str_og = self.target3_pos_str
            self.target4_pos_str_og = self.target4_pos_str
            self.target5_pos_str_og = self.target5_pos_str
        
        # target 1
        if not seq_preselect:
            target_pos_opts = ['random', 'center', 'upper_left', 'middle_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'middle_right', 'lower_right']
            for i, val in enumerate(task_in['targ1_pos']):
                if val:
                    self.target1_pos_str = target_pos_opts[i]
            
        self.target1_position = self.get_targpos_from_str(self.target1_pos_str, self.nudge_x_t1)
        
        # target 2
        if not seq_preselect:
            target_pos_opts = ['none', 'random', 'center', 'upper_left', 'middle_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'middle_right', 'lower_right']
            for i, val in enumerate(task_in['targ2_pos']):
                if val:
                    self.target2_pos_str = target_pos_opts[i]

        if not self.target2_pos_str == 'none': 
            self.target2_position = self.get_targpos_from_str(self.target2_pos_str, self.nudge_x_t2)
        else:
            self.target2_position = False
            self.num_targets = 1
        
        # target 3
        if not seq_preselect:
            target_pos_opts = ['none', 'center', 'upper_left', 'middle_left', 'lower_left', 'upper_middle', 'lower_middle', 'upper_right', 'middle_right', 'lower_right']
            for i, val in enumerate(task_in['targ3_pos']):
                if val:
                    self.target3_pos_str = target_pos_opts[i]
        
        if not self.target3_pos_str == 'none': 
            self.target3_position = self.get_targpos_from_str(self.target3_pos_str, self.nudge_x_t3)
        else:
            self.target3_position = False
            if not self.target2_position is False:
                 self.num_targets = 2
            
        # target 4
        if not seq_preselect:
            for i, val in enumerate(task_in['targ4_pos']):
                if val:
                    self.target4_pos_str = target_pos_opts[i]
        
        if not self.target4_pos_str == 'none': 
            self.target4_position = self.get_targpos_from_str(self.target4_pos_str, self.nudge_x_t4)
        else:
            self.target4_position = False
            if not self.target3_position is False:
                 self.num_targets = 3
            
        # target 5
        if not seq_preselect:
            for i, val in enumerate(task_in['targ5_pos']):
                if val:
                    self.target5_pos_str = target_pos_opts[i]
        
        if not self.target5_pos_str == 'none': 
            self.target5_position = self.get_targpos_from_str(self.target5_pos_str, self.nudge_x_t4)
            self.num_targets = 5
        else:
            self.target5_position = False
            if not self.target4_position is False:
                 self.num_targets = 4
        
        if self.seq == 'rand5-randevery' or self.seq == '2seq-repeat':
            self.target1_position_og = self.target1_position
            self.target2_position_og = self.target2_position
            self.target3_position_og = self.target3_position
            self.target4_position_og = self.target4_position
            self.target5_position_og = self.target5_position

        self.active_target_position = self.target1_position
        self.target_index = 1
        
        # ANIMAL NAME
        for i, (nm, val) in enumerate(animal_names_dict.items()):
            if val:
                animal_name = nm
                
        if animal_name == 'testing':
            self.testing = True
            self.tic_touch_target = 0
            self.tic_on_touch_down_start = 0
            self.tic_on_touch_down_end = 0
            self.tic_write_to_h5file_start = 0
            self.tic_write_to_h5file_end = 0
            self.tic_touch_targ_nohold_start = 0
            self.tic_touch_targ_nohold_end = 0
            self.tic_update_start = 0
        else:
            self.testing = False
        
        # HOW MUCH TIME TO WAIT UNTIL THE NEXT TARGET APPEARS
        time_to_next_targ_opts = [False, 0.25, 0.5, 0.75, 1.0, 1.5]
        for i, val in enumerate(task_in['time_to_next_targ']):
            if val:
                self.time_to_next_targ = time_to_next_targ_opts[i]
                
        # INTER TARGET DELAY TIME
        intertarg_delay_opts = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        for i, val in enumerate(task_in['intertarg_delay']):
            if val:
                self.intertarg_delay = intertarg_delay_opts[i]
               
        # BUTTON VERSION        
        button_opts = ['fsr', 'ir']
        for i, val in enumerate(button['button']): 
            if val: 
                self.button_version = button_opts[i]


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
        holdz = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, '.1-.3', '.4-.6']
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
        
        
        # JUICE REWARD SETTINGS
        button_rew_opts = [0., .1, .3, .5]
        for i, val in enumerate(rew_in['button_rew']):
            if val:
                button_rew = button_rew_opts[i]
        
        min_rew_opts = ['No Reward Scaling', 0., .1, .2, .3, 0.4, 0.5, 0.6, 0.7]
        for i, val in enumerate(rew_in['min_rew']):
            if val:
                min_rew = min_rew_opts[i]
        
        big_rew_opts = [.3, .5, .7, 0.9, 1.1]
        for i, val in enumerate(rew_in['big_rew']):
            if val:
                big_rew = big_rew_opts[i]
        
        if button_rew > 0.0:
            self.button_rew = [True, button_rew]
        else:
            self.button_rew = [False, 0]
        
        if min_rew == 'No Reward Scaling':
            self.min_targ_reward = [False, False]
        else:
            self.min_targ_reward = [True, min_rew]

        if big_rew > 0.0:
            self.last_targ_reward = [True, big_rew]
        else:
            self.last_targ_reward = [False, 0]
        
        if animal_name is 'butters':
            targ1on2touch_fast = 0.9
            targon2touch_fast = 0.5
            targon2touch_slow = 0.6
        elif animal_name is 'fifi':
            targ1on2touch_fast = 0.65
            targon2touch_fast = 0.45
            targon2touch_slow = 0.55
        else:
            targ1on2touch_fast = 0.7
            targon2touch_fast = 0.45
            targon2touch_slow = 0.6
        self.time_thresh_for_max_rew = targ1on2touch_fast+targon2touch_fast*(self.num_targets-1)+self.intertarg_delay*self.num_targets
        self.time_thresh_for_min_rew = targ1on2touch_fast+targon2touch_slow*(self.num_targets-1)+self.intertarg_delay*self.num_targets
        
        # reward_delay_opts = [0., .4, .8, 1.2]
        # for i, val in enumerate(rew_del['rew_del']):
        #     if val:
        self.reward_delay_time = 0.0

        self.percent_of_trials_rewarded = 1.0
        self.percent_of_trials_doubled = 0.0
        # reward_var_opt = [1.0, .5, .33]
        # for i, val in enumerate(rew_var['rew_var']):
        #     if val:
        #         self.percent_of_trials_rewarded = reward_var_opt[i]
        #         if self.percent_of_trials_rewarded == 0.33:
        #             self.percent_of_trials_doubled = 0.1
        #         else:
        #             self.percent_of_trials_doubled = 0.0
        
        self.reward_generator = self.gen_rewards(self.percent_of_trials_rewarded, self.percent_of_trials_doubled,
            self.last_targ_reward)
        
        self.trial_completion_time = 0


        # white_screen_opts = [True, False]
        # for i, val in enumerate(white_screen['white_screen']):
        #     if val:
        self.use_white_screen = False

        autoquit_trls = [10, 25, 50, 60, 90, 100, 10**10]
        for i, val in enumerate(autoquit['autoquit']):
            if val: 
                self.max_trials = autoquit_trls[i]
                
        # TASK BREAKS
        break_trls = [0, 10, 15, 20, 25]
        for i, val in enumerate(taskbreak['breaktrl']):
            if val: 
                self.break_trl = break_trls[i]
                
        self.next_breaktrl = self.break_trl
        self.block_ix = 1
                
        break_durs = [30, 60, 90, 120, 150]
        for i, val in enumerate(taskbreak['breakdur']):
            if val: 
                self.break_dur = break_durs[i]
        
        # OK to drag into the targets?
        self.drag_ok = False;
        drag_opts = [True, False]
        for i, val in enumerate(drag['drag']):
            if val:
                self.drag_ok = drag_opts[i]
                
        # display target outlines?
        self.display_outlines = False
        outline_opts = [True, False]
        for i, val in enumerate(task_in['outlines']):
            if val:
                self.display_outlines = outline_opts[i]
        

        # nudge_9am_dist = [0., .5, 1.]
        # for i, val in enumerate(nudge['nudge']):
        #     if val:
        self.nudge_dist = 0.

        # targ_pos = ['corners', None]
        # for i, val in enumerate(targ_pos['targ_pos']):
        #     if val:
        self.generator_kwarg = 'corners'


        # Preload sounds: 
        self.target1_start_sound = SoundLoader.load('C.wav')
        tmp = time.time()
        self.reward1 = SoundLoader.load('reward1.wav')
        print('Time for big reward sound to load: ', time.time()-tmp)
        tmp = time.time()
        self.reward2 = SoundLoader.load('reward2.wav')
        print('Time for small reward sound to load: ', time.time()-tmp)
        

        self.state = 'ITI'
        self.state_start = time.time()
        self.ITI = self.ITI_std + self.ITI_mean
        
        # Initialize the background targets
        self.target1_out.color = (1., 1., 0., 0.)
        self.target1_out.set_size(2*self.target_rad)
        self.target1_in.color = (0., 0., 0., 0.)
        self.target1_in.set_size(2*self.target_rad-0.1)
        self.target2_out.color = (1., 1., 0., 0.)
        self.target2_out.set_size(2*self.target_rad)
        self.target2_in.color = (0., 0., 0., 0.)
        self.target2_in.set_size(2*self.target_rad-0.1)
        self.target3_out.color = (1., 1., 0., 0.)
        self.target3_out.set_size(2*self.target_rad)
        self.target3_in.color = (0., 0., 0., 0.)
        self.target3_in.set_size(2*self.target_rad-0.1)
        self.target4_out.color = (1., 1., 0., 0.)
        self.target4_out.set_size(2*self.target_rad)
        self.target4_in.color = (0., 0., 0., 0.)
        self.target4_in.set_size(2*self.target_rad-0.1)
        self.target5_out.color = (1., 1., 0., 0.)
        self.target5_out.set_size(2*self.target_rad)
        self.target5_in.color = (0., 0., 0., 0.)
        self.target5_in.set_size(2*self.target_rad-0.1)
        self.target6_out.color = (1., 1., 0., 0.)
        self.target6_out.set_size(2*self.target_rad)
        self.target6_in.color = (0., 0., 0., 0.)
        self.target6_in.set_size(2*self.target_rad-0.1)
        self.target7_out.color = (1., 1., 0., 0.)
        self.target7_out.set_size(2*self.target_rad)
        self.target7_in.color = (0., 0., 0., 0.)
        self.target7_in.set_size(2*self.target_rad-0.1)
        self.target8_out.color = (1., 1., 0., 0.)
        self.target8_out.set_size(2*self.target_rad)
        self.target8_in.color = (0., 0., 0., 0.)
        self.target8_in.set_size(2*self.target_rad-0.1)
        self.target9_out.color = (1., 1., 0., 0.)
        self.target9_out.set_size(2*self.target_rad)
        self.target9_in.color = (0., 0., 0., 0.)
        self.target9_in.set_size(2*self.target_rad-0.1)

        # Initialize targets: 
        self.target1.set_size(2*self.target_rad)
        self.target2.set_size(2*self.target_rad)
        self.target3.set_size(2*self.target_rad)
        self.target4.set_size(2*self.target_rad)
        self.target5.set_size(2*self.target_rad)
        self.target1.color = (0., 0., 0., 0.)
        self.target2.color = (0., 0., 0., 0.)
        self.target3.color = (0., 0., 0., 0.)
        self.target4.color = (0., 0., 0., 0.)
        self.target5.color = (0., 0., 0., 0.)
       

        self.exit_target1.set_size(2*self.exit_rad)
        self.exit_target2.set_size(2*self.exit_rad)
        self.exit_target1.color = (.15, .15, .15, 1)
        self.exit_target2.color = (.15, .15, .15, 1)
        self.pd1_indicator_targ.set_size(self.exit_rad)
        self.pd1_indicator_targ.move(self.pd1_indicator_pos)
        self.pd1_indicator_targ.color = (.75, .75, .75, 1.)
        # self.pd2_indicator_targ.set_size(self.exit_rad)
        # self.pd2_indicator_targ.move(self.pd2_indicator_pos)
        # self.pd2_indicator_targ.color = (0., 0., 0., 1.)
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
        self.FSM['ITI'] = dict(end_ITI='taskbreak', stop=None)
        self.FSM['taskbreak'] = dict(end_taskbreak='vid_trig', stop=None)
        self.FSM['vid_trig'] = dict(end_vid_trig='button', stop=None)
        self.FSM['button'] = dict(button_pressed='button_hold', stop=None)
        self.FSM['button_hold'] = dict(finish_button_hold='target', early_leave_button_hold='button', stop=None)

        self.FSM['target'] = dict(touch_target_nohold = 'target', touch_target = 'targ_hold', target_timeout='timeout_error', stop=None)
        self.FSM['targ_hold'] = dict(finish_last_targ_hold='reward', finish_targ_hold='target', early_leave_target_hold = 'hold_error',
         targ_drag_out = 'drag_error', stop=None)
        self.FSM['reward'] = dict(end_reward = 'ITI', stop=None)
        
        self.FSM['timeout_error'] = dict(end_timeout_error='ITI', stop=None)
        self.FSM['hold_error'] = dict(end_hold_error='target', stop=None)
        self.FSM['drag_error'] = dict(end_drag_error='target', stop=None)
        self.FSM['idle_exit'] = dict(stop=None)
        
        # OPEN PORTS
        try:
            if self.juicer == 'yellow':
                if user_id == 'Ganguly':
                    self.reward_port = serial.Serial(port='COM4',
                        baudrate=115200)
                elif user_id == 'BasalGangulia':
                    self.reward_port = serial.Serial(port='COM3',
                        baudrate=115200)
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

                ### setup the flow rate
                time.sleep(.5) 

                ### set volume value and units and rate units
                self.reward_port.write(b"VOL 0.5\r")
                self.reward_port.write(b"VOL ML\r")
                self.reward_port.write(b"RAT 50MM\r") # 50 ml / min

        except:
            pass
        
        # DIO
        try:
            if user_id == 'Ganguly':
                self.dio_port = serial.Serial(port='COM5', baudrate=115200)
            elif user_id == 'BasalGangulia':
                self.dio_port = serial.Serial(port='COM13', baudrate=115200)
            
            time.sleep(4.)
        except:
            pass
        
        # Camera Triggers
        try:
            if user_id == 'Ganguly':
                self.cam_trig_port = serial.Serial(port='COM6', baudrate=9600)
            elif user_id == 'BasalGangulia':
                self.cam_trig_port = serial.Serial(port='COM6', baudrate=9600)
            
            time.sleep(3.)
            # Say hello: 
            self.cam_trig_port.write('a'.encode())

            # Start cams @ 50 Hz
            self.cam_trig_port.write('1'.encode())
        except:
            pass
        
        # Eyetracker Triggers
        try:
            if user_id == 'BasalGangulia':
                self.iscan_trig_port = serial.Serial(port='COM7', baudrate=115200)
            
            time.sleep(4.)
        except:
            pass
        
        
        
        # Send Eyetracker Start Recording Trigger
        try:
            ### write to arduino: 
            word_str = b's'
            self.iscan_trig_port.write(word_str)
        except:
            pass
        
        # External button
        import serial
        try:
            self.is_button_ard = True
            if platform == 'darwin':
                self.button_ard = serial.Serial(port='/dev/cu.usbserial-141130', baudrate=9600)
            else:
                if user_id == 'Ganguly':
                    self.button_ard = serial.Serial(port='COM3', baudrate=9600) 
                elif user_id == 'BasalGangulia':
                    if self.button_version is 'fsr':
                        self.button_ard = serial.Serial(port='COM9', baudrate=9600)
                    elif self.button_version is 'ir':
                        self.button_ard = serial.Serial(port='COM16', baudrate=9600)
        
        except:
            self.is_button_ard = False
        
        if self.is_button_ard and self.button_version is 'fsr':
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
            # self.fsr_baseline = 100+1.5*np.max(baseline_data, axis=0)
            if user_id == 'Ganguly':
                self.fsr_baseline = 20+1.15*np.max(baseline_data, axis=0)
            elif user_id == 'BasalGangulia':
                self.fsr_baseline = 100+np.max(baseline_data, axis=0)
        else: 
            self.fsr_baseline = np.array([200, 200])
        
        
            

        # save parameters: 
        d = dict(animal_name=animal_name,
            juicer = self.juicer,
            user_id = user_id,
            break_trl = self.break_trl,
            break_dur = self.break_dur,
            max_trials = self.max_trials,
            target1_timeout_time = self.target1_timeout_time,
            target_timeout_time = self.target_timeout_time,
            button_rew = button_rew,
            min_targ_reward = self.min_targ_reward[1],
            last_targ_reward = self.last_targ_reward[1],
            nudge_x_t1 = self.nudge_x_t1,
            nudge_x_t2 = self.nudge_x_t2,
            nudge_x_t3 = self.nudge_x_t3,
            nudge_x_t4 = self.nudge_x_t4,
            screen_top = self.screen_top,
            screen_bot = self.screen_bot,
            target_rad=self.target_rad,
            effective_target_rad=self.eff_target_rad,
            center_position = self.center_position,
            seq = self.seq,
            target1_pos_str = self.target1_pos_str,
            target2_pos_str = self.target2_pos_str,
            target3_pos_str = self.target3_pos_str,
            target4_pos_str = self.target4_pos_str,
            target5_pos_str = self.target5_pos_str,
            target1_position = self.target1_position, 
            target2_position = self.target2_position, 
            target3_position = self.target3_position, 
            target4_position = self.target4_position, 
            target5_position = self.target5_position,
            time_to_next_targ = self.time_to_next_targ,
            intertarg_delay = self.intertarg_delay,
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
            fsr_baseline = self.fsr_baseline,
            button_version = self.button_version,
            display_outlines = self.display_outlines
            )

        # if self.testing:
        #     pass

        # else:
        # Try saving to Box
        if user_id == 'Sandon':
            box_path = '/Users/Sandon/Box/Data/NHP_BehavioralData/target_chase/'
            last_param_path = '/Users/Sandon/Documents/'
        elif user_id == 'Ganguly':
            box_path = 'C:/Users/Ganguly/Box/Data/NHP_BehavioralData/target_chase/'
            last_param_path = 'C:/Users/Ganguly/Documents/'
        elif user_id == 'BasalGangulia':
            box_path = 'C:/Users/BasalGangulia/Box/Data/NHP_BehavioralData/target_chase/'
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
        pickle.dump(d, open(last_param_path+'most_recent_target_chase_params.pkl', 'wb'))

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
        
        # Send Eyetracker Stop Recording Trigger
        try:
            ### write to arduino: 
            word_str = b'e'
            self.iscan_trig_port.write(word_str)
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
        
        if self.testing:
            self.tic_update_start = time.time()
            
        ## Change the luminance of the photodiode indicator
        
        if not np.any(self.cursor_ids):
            self.vid_indicator_targ.color = (.25, .25, .25, 1.)
            self.istouch = False
        # cursor = np.zeros((10, 2))
        # cursor[:] = np.nan
        # for ic, curs_id in enumerate(self.cursor_ids):
        #     cursor[ic, :] = self.cursor[curs_id]
        # if np.isnan(cursor).all():
        #     self.vid_indicator_targ.color = (.25, .25, .25, 1.)
        #     # self.pd2_indicator_targ.color = (0., 0., 0., 1.)
        else:
            self.istouch = True
            self.vid_indicator_targ.color = (.5, .5, .5, 1.)
            # if self.state == 'target' and self.touch_target():
            #     self.pd2_indicator_targ.color = (1., 1., 1., 1.)
            # else:
            #     self.pd2_indicator_targ.color = (.75, .75, .75, 1.)
        
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
                    
        
             
        # if self.testing:
        #     pass
        # else:
        if self.state == 'idle_exit':
            pass
        else:
            if self.testing:
                self.tic_write_to_h5file_start = time.time()
            self.write_to_h5file()
            if self.testing:
                self.tic_write_to_h5file_end = time.time()

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

        self.h5_table_row['target_pos'] = self.active_target_position
        if self.is_button_ard and self.button_pressed() \
            and not self.state == 'target' and not self.state == 'targ_hold':
            self.h5_table_row['button_state'] = 1
        else:
            self.h5_table_row['button_state'] = 0
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
        
    # def write_iscan_trig(self):
    #     ### write to arduino: 
    #     word_str = b't'
    #     self.iscan_trig_port.write(word_str)

    def stop(self, **kwargs):
        # If past number of max trials then auto-quit: 
        if np.logical_and(self.trial_counter >= self.max_trials, self.state == 'ITI'):
            self.idle = True
            self.pd1_indicator_targ.color = (.75, .75, .75, 1.)
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
                self.pd1_indicator_targ.color = (.75, .75, .75, 1.)
                return True

            else:
                return False

    def _start_ITI(self, **kwargs):
        try:
            self.cam_trig_port.write('0'.encode())
        except:
            pass
        Window.clearcolor = (0., 0., 0., 1.)
        
        if self.display_outlines:
            self.set_alloutlinetargs_color(0., 0., 0., 0.)

        # Set ITI, CHT, THT
        self.ITI = np.random.random()*self.ITI_std + self.ITI_mean

        if type(self.tht_type) is str:
            tht_min, tht_max = self.tht_type.split('-')
            self.tht = ((float(tht_max) - float(tht_min)) * np.random.random()) + float(tht_min)      
        
        if type(self.button_hold_time_type) is str:
            bht_min, bht_max = self.button_hold_time_type.split('-')
            self.button_hold_time = ((float(bht_max) - float(bht_min)) * np.random.random()) + float(bht_min)     
            
        # Reset the target colors
        self.target1.color = (0., 0., 0., 0.)
        self.target2.color = (0., 0., 0., 0.)
        self.target3.color = (0., 0., 0., 0.)
        self.target4.color = (0., 0., 0., 0.)
        self.target5.color = (0., 0., 0., 0.)
        self.exit_target1.color = (.15, .15, .15, 1)
        self.exit_target2.color = (.15, .15, .15, 1)
        self.pd1_indicator_targ.color = (.75, .75, .75, 1.)
        
        
        
    def end_ITI(self, **kwargs):
        return kwargs['ts'] > self.ITI
    
    def _start_taskbreak(self, **kwargs):
        if self.break_trl == 0:
            self.this_breakdur = 0
        else:
            if self.trial_counter == self.next_breaktrl:
                sound = SoundLoader.load('DoorBell.wav')
                sound.play()
                self.this_breakdur = self.break_dur
                self.next_breaktrl = self.next_breaktrl + self.break_trl
                self.block_ix += 1
            else:
                self.this_breakdur = 0
                
        # do stuff that should be done during the ITI but we put it here because _start_ITI does not run for the first trial
        
        # try:
        #     self.write_iscan_trig()
        # except:
        #     pass
        self.trials_started += 1
        

        self.first_target_attempt = True
        self.first_time_for_this_targ = True
        
        # Reset target index back to 1
        self.target_index = 1
        
        # Get the position of random targets
        # i_pos_rand = np.random.permutation(9)
        if self.seq == 'randevery' or self.seq == 'center out' or self.seq == 'button out':
            self.target1_pos_str = 'random'
            self.target2_pos_str = 'random'
            self.target3_pos_str = 'random'
            self.target4_pos_str = 'random'
            self.target5_pos_str = 'random'
            self.make_random_sequence(False)
        elif self.seq == 'rand5-randevery':
            if np.remainder(self.block_ix, 2) == 1 and self.block_ix > 2:
                self.target1_pos_str = 'random'
                self.target2_pos_str = 'random'
                self.target3_pos_str = 'random'
                self.target4_pos_str = 'random'
                self.target5_pos_str = 'random'
                self.make_random_sequence(False)
            else:
                self.target1_position = self.target1_position_og
                self.target2_position = self.target2_position_og
                self.target3_position = self.target3_position_og
                self.target4_position = self.target4_position_og
                self.target5_position = self.target5_position_og
        elif self.seq == '2seq-repeat':
            if np.remainder(self.block_ix, 2) == 0:
                if self.seq2generated:
                    self.target1_position = self.seq2_target1_pos
                    self.target2_position = self.seq2_target2_pos
                    self.target3_position = self.seq2_target3_pos
                    self.target4_position = self.seq2_target4_pos
                    self.target5_position = self.seq2_target5_pos
                else:
                    self.target1_pos_str = 'random'
                    self.target2_pos_str = 'random'
                    self.target3_pos_str = 'random'
                    self.target4_pos_str = 'random'
                    self.target5_pos_str = 'random'
                    self.make_random_sequence(False)
                    
                    self.seq2_target1_pos = self.target1_position
                    self.seq2_target2_pos = self.target2_position
                    self.seq2_target3_pos = self.target3_position
                    self.seq2_target4_pos = self.target4_position
                    self.seq2_target5_pos = self.target5_position
                    self.seq2generated = True
            else:
                self.target1_position = self.target1_position_og
                self.target2_position = self.target2_position_og
                self.target3_position = self.target3_position_og
                self.target4_position = self.target4_position_og
                self.target5_position = self.target5_position_og
                
        # move the targets to their respective positions
        self.target1.move(self.target1_position)
        self.target2.move(self.target2_position)
        self.target3.move(self.target3_position)
        self.target4.move(self.target4_position)
        self.target5.move(self.target5_position)
    
    def end_taskbreak(self, **kwargs):
        if self.this_breakdur > 0 and kwargs['ts'] > self.this_breakdur:
            sound = SoundLoader.load('DoorBell.wav')
            sound.play()
        return kwargs['ts'] > self.this_breakdur

    def _start_vid_trig(self, **kwargs):
        if self.trial_counter == 0:
            time.sleep(1.)
        try:    
            self.cam_trig_port.write('1'.encode())
        except:
            pass

    def end_vid_trig(self, **kwargs):
        return kwargs['ts'] > self.pre_start_vid_ts
    
    def _start_button(self, **kwargs):
        Window.clearcolor = (0., 0., 0., 1.)
        # self.target1.color = (0., 0., 0., 0.)
        # self.target2.color = (0., 0., 0., 0.)
        # self.exit_target1.color = (.15, .15, .15, 1)
        # self.exit_target2.color = (.15, .15, .15, 1)
        self.pd1_indicator_targ.color = (0., 0., 0., 0.)
        # try:
        #     self.write_iscan_trig()
        # except:
        #     pass
        self.button_pressed_prev = False
        
        if self.display_outlines:
            self.set_targoutline_color(1., 1., 0., 1.)
        
    def button_pressed(self, **kwargs):
        if not self.use_button or not self.is_button_ard:
            return True
        else:
            # Get the button values
            ser = self.button_ard.flushInput()
            _ = self.button_ard.readline()
            port_read = self.button_ard.readline()
            port_read = port_read.decode('ascii')
            
            if self.button_version is 'fsr':
                i_slash = port_read.find('/')
                fsr1 = int(port_read[0:i_slash])
                fsr2 = int(port_read[i_slash+1:])
            
                # Determine if the button was pressed or not
                if fsr1 > self.fsr_baseline[0] or fsr2 > self.fsr_baseline[1]:
                    return True
                    # print('Button Pressed')
                else:
                    return False
                    # print('Button NOT Pressed')
            elif self.button_version is 'ir':
                if int(port_read) == 1:
                    return True
                else:
                    return False
    
    def _start_button_hold(self, **kwargs):
        self.t_button_hold_start = time.time()
        self.pd1_indicator_targ.color = (1., 1., 1., 1.)
        # try:
        #     self.write_iscan_trig()
        # except:
        #     pass
        
    def finish_button_hold(self, **kwargs):
        if self.use_button is False or self.is_button_ard is False:
            return True
        else:
            if time.time() - self.t_button_hold_start > self.button_hold_time:
                # Play the button reward sound
                self.target1_start_sound.play()
                # if the button has been held down long enough
                if self.button_rew[0]:
                    self.run_button_rew()
                return True
            else:
                return False

    def early_leave_button_hold(self, **kwargs):
        if self.use_button is False or self.is_button_ard is False:
            return False
        else:
            if self.button_pressed():
                return False
            else:
                return True
    
    def _start_targ_hold(self, **kwargs):
        if self.tht > 0:
            if self.target_index == 1:
                self.target1.color = (0., 1., 0., 1.)
            elif self.target_index == 2:
                self.target2.color = (0., 1., 0., 1.)
            elif self.target_index == 3:
                self.target3.color = (0., 1., 0., 1.)
            elif self.target_index == 4:
                self.target4.color = (0., 1., 0., 1.)
            elif self.target_index == 5:
                self.target5.color = (0., 1., 0., 1.)

    def _start_timeout_error(self, **kwargs):
        self.target1.color = (0., 0., 0., 0.)
        self.target2.color = (0., 0., 0., 0.)
        self.target3.color = (0., 0., 0., 0.)
        self.target4.color = (0., 0., 0., 0.)
        self.target5.color = (0., 0., 0., 0.)
        #self.repeat = True

    def _start_hold_error(self, **kwargs):
        self.target1.color = (0., 0., 0., 0.)
        self.target2.color = (0., 0., 0., 0.)
        self.target3.color = (0., 0., 0., 0.)
        self.target4.color = (0., 0., 0., 0.)
        self.target5.color = (0., 0., 0., 0.)
        self.repeat = True

    def _start_drag_error(self, **kwargs):
        self.target1.color = (0., 0., 0., 0.)
        self.target2.color = (0., 0., 0., 0.)
        self.target3.color = (0., 0., 0., 0.)
        self.target4.color = (0., 0., 0., 0.)
        self.target5.color = (0., 0., 0., 0.)
        self.repeat = True
                
    def _start_target(self, **kwargs):
        if self.testing:
            toc_start_target = time.time()
            print('Time from touch_target touch to start_target: ', np.round(1000*(toc_start_target-self.tic_touch_target), 3), ' ms')
            print('Time from start of on_touch_down to start_target: ', np.round(1000*(toc_start_target-self.tic_on_touch_down_start), 3), ' ms')
            print('Time from end of on_touch_down to start_target: ', np.round(1000*(toc_start_target-self.tic_on_touch_down_end), 3), ' ms')
            print('Time from start of write_to_h5file to start_target: ', np.round(1000*(toc_start_target-self.tic_write_to_h5file_start), 3), ' ms')
            print('Time from end of write_to_h5file to start_target: ', np.round(1000*(toc_start_target-self.tic_write_to_h5file_end), 3), ' ms')
            print('Time from start of update to start_target: ', np.round(1000*(toc_start_target-self.tic_update_start), 3), ' ms')
            print('Time from start of touch_target_nohold to start_target: ', np.round(1000*(toc_start_target-self.tic_touch_targ_nohold_start), 3), ' ms')
            print('Time from end of touch_target_nohold to start_target: ', np.round(1000*(toc_start_target-self.tic_touch_targ_nohold_end), 3), ' ms')
            
            
            
        Window.clearcolor = (0., 0., 0., 1.)
        
        if self.first_time_for_this_targ:
            self.first_time_for_this_targ_t0 = time.time()
            self.first_time_for_this_targ = False
        
        if self.target_index == 1:
            self.active_target_position = self.target1_position
            self.next_target_position = self.target2_position
            if self.intertarg_delay == 0:
                self.target1.color = (1., 1., 0., 1.)
                self.pd1_indicator_targ.color = (0., 0., 0., 0.)
        elif self.target_index == 2:
            self.active_target_position = self.target2_position
            self.next_target_position = self.target3_position
            if self.intertarg_delay == 0:
                self.target2.color = (1., 1., 0., 1.)
                self.pd1_indicator_targ.color = (1., 1., 1., 1.)
        elif self.target_index == 3:
            self.active_target_position = self.target3_position
            self.next_target_position = self.target4_position
            if self.intertarg_delay == 0:
                self.target3.color = (1., 1., 0., 1.)
                self.pd1_indicator_targ.color = (0., 0., 0., 0.)
        elif self.target_index == 4:
            self.active_target_position = self.target4_position
            self.next_target_position = self.target5_position
            if self.intertarg_delay == 0:
                self.target4.color = (1., 1., 0., 1.)
                self.pd1_indicator_targ.color = (1., 1., 1., 1.)
        elif self.target_index == 5:
            self.active_target_position = self.target5_position
            self.next_target_position = False
            if self.intertarg_delay == 0:
                self.target5.color = (1., 1., 0., 1.)
                self.pd1_indicator_targ.color = (0., 0., 0., 0.)
            
        if self.target_index == 1:
            self.target1_on_time = time.time()

        # self.target1.move(self.active_target_position)
        
        # if self.intertarg_delay == 0:
        #     self.target1.color = (1., 1., 0., 1.)
        #     if np.remainder(self.target_index, 2) == 1:
        #         self.pd1_indicator_targ.color = (0., 0., 0., 0.)
        #     elif np.remainder(self.target_index, 2) == 0:
        #         self.pd1_indicator_targ.color = (1., 1., 1., 1.)
        
        # self.exit_target1.color = (.15, .15, .15, 1)
        # self.exit_target2.color = (.15, .15, .15, 1)
            
        # try:
        #     self.write_iscan_trig()
        # except:
        #     pass
            
            
        if self.first_target_attempt:
            self.first_target_attempt_t0 = time.time();
            self.first_target_attempt = False
            
        self.repeat = False
    
    def _while_target(self, **kwargs):
        # check and see if it is time for the next target to appear
        if self.time_to_next_targ is not False:
            if time.time() - self.first_time_for_this_targ_t0 > self.time_to_next_targ and self.target_index < self.num_targets:
                # illuminate the next target
                if self.target_index == 1:
                    self.target2.color = (1., 1., 0., 1.)
                elif self.target_index == 2:
                    self.target3.color = (1., 1., 0., 1.)
                elif self.target_index == 3:
                    self.target4.color = (1., 1., 0., 1.)
                elif self.target_index == 4:
                    self.target5.color = (1., 1., 0., 1.)
                # self.target2.move(self.next_target_position)
                # self.target2.color = (1., 1., 0., 1.)
                
        if not self.intertarg_delay == 0:
            if self.target_index == 1 or time.time() - self.first_time_for_this_targ_t0 >= self.intertarg_delay:
                if self.target_index == 1:
                    self.target1.color = (1., 1., 0., 1.)
                elif self.target_index == 2:
                    self.target2.color = (1., 1., 0., 1.)
                elif self.target_index == 3:
                    self.target3.color = (1., 1., 0., 1.)
                elif self.target_index == 4:
                    self.target4.color = (1., 1., 0., 1.)
                elif self.target_index == 5:
                    self.target4.color = (1., 1., 0., 1.)
                self.pd1_indicator_targ.color = (0., 0., 0., 0.)
                

    def _start_reward(self, **kwargs):
        self.trial_counter += 1
        Window.clearcolor = (1., 1., 1., 1.)
        if self.display_outlines:
            self.set_alloutlinetargs_color(1., 1., 1., 1.)
        self.target1.color = (1., 1., 1., 1.)
        self.target2.color = (1., 1., 1., 1.)
        self.target3.color = (1., 1., 1., 1.)
        self.target4.color = (1., 1., 1., 1.)
        self.target5.color = (1., 1., 1., 1.)
        self.exit_target1.color = (1., 1., 1., 1.)
        self.exit_target2.color = (1., 1., 1., 1.)
        self.rew_cnt = 0
        self.cnts_in_rew = 0
        if np.remainder(self.num_targets, 2) == 1 or not self.intertarg_delay == 0:
            self.pd1_indicator_targ.color = (1., 1., 1., 1.)
        elif np.remainder(self.num_targets, 2) == 0:
            self.pd1_indicator_targ.color = (0., 0., 0., 1.)
        # try:
        #     self.write_iscan_trig()
        # except:
        #     pass
        self.repeat = False

    def _while_reward(self, **kwargs):
        if self.rew_cnt == 1:
            self.run_big_rew()
            self.rew_cnt += 1

    def write_juice_reward(self, rew_time): 
        if self.juicer == 'yellow': 
            rew_str = [ord(r) for r in 'inf 50 ml/min '+str(rew_time)+' sec\n']
            
            try: # commented out comport open/close -- was giving errors in spinning pal
                self.reward_port.open()
                self.reward_port.write(rew_str)
                time.sleep(.25)
                run_str = [ord(r) for r in 'run\n']
                self.reward_port.write(run_str)
                self.reward_port.close()
            except:
                pass            
        

        elif self.juicer == 'red':
            volume2dispense = rew_time * 50 / 60 #mL/min x 1 min / 60 sec --> sec x mL/sec 
            self.reward_port.write(b"VOL %.1f \r"%volume2dispense)
            time.sleep(.25)
            self.reward_port.write(b"RUN\r")

    def run_big_rew(self, **kwargs):
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
                self.reward1.play()
                if self.reward_generator[self.trial_counter] > 0:
                    #self.reward_port.open()
                    if self.min_targ_reward[0]:
                        if self.trial_completion_time < self.time_thresh_for_max_rew:
                            this_rew = self.last_targ_reward[1]
                        else:
                            this_rew = self.min_targ_reward[1]
                        # else:
                            # rel_time = (self.trial_completion_time - self.time_thresh_for_max_rew)/(self.time_thresh_for_min_rew - self.time_thresh_for_max_rew)
                            # rel_time = 1 - rel_time
                            # this_rew = round(self.min_targ_reward[1] + rel_time*(self.last_targ_reward[1]-self.min_targ_reward[1]), 1)
                        rew_str = [ord(r) for r in 'inf 50 ml/min '+str(this_rew)+' sec\n']
                        print('Scaled reward: ' + str(this_rew) + 'sec')
                        rew_time = this_rew; 
                    else:
                        #rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.last_targ_reward[1])+' sec\n']
                        rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.reward_generator[self.trial_counter])+' sec\n']
                        rew_time = self.reward_generator[self.trial_counter]

                    self.write_juice_reward(rew_time)
                    #self.reward_port.write(rew_str)
                    #time.sleep(.25 + self.reward_delay_time)
                    #run_str = [ord(r) for r in 'run\n']
                    #self.reward_port.write(run_str)
                    #self.reward_port.close()
        except:
            pass
        
    def run_button_rew(self, **kwargs):
        try:
            ### To trigger reward make sure reward is > 0:
            if np.logical_or(self.button_rew[0], self.button_rew[1] > 0):

                self.write_juice_reward(self.button_rew[1])
                # self.reward_port.open()
                # rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.button_rew[1])+' sec\n']
                # self.reward_port.write(rew_str)
                # time.sleep(.25)
                # run_str = [ord(r) for r in 'run\n']
                # self.reward_port.write(run_str)
                # self.reward_port.close()
        except:
            pass

        #self.repeat = True

    def end_reward(self, **kwargs):
        if self.use_white_screen:
            if len(self.cursor_ids)== 0:
                return True
        else:
            if self.cnts_in_rew > 30:
                return True
            else:
                self.cnts_in_rew += 1
                return False

    def end_timeout_error(self, **kwargs):
        return kwargs['ts'] >= self.timeout_error_timeout

    def end_hold_error(self, **kwargs):
        return kwargs['ts'] >= self.hold_error_timeout

    def end_drag_error(self, **kwargs):
        return kwargs['ts'] >= self.drag_error_timeout

    def touch_target(self, **kwargs):
        if self.istouch:
            if self.drag_ok:
                if self.check_if_cursors_in_targ(self.active_target_position, self.eff_target_rad):
                    if self.testing:
                        self.tic_touch_target = time.time()
                    return True
                else:
                    return False
            else:
                if np.logical_and(self.check_if_cursors_in_targ(self.active_target_position, self.eff_target_rad),
                    self.check_if_started_in_targ(self.active_target_position, self.eff_target_rad)):
                    if self.testing:
                        self.tic_touch_target = time.time()
                    return True
                else:
                    return False
        else:
            return False
            
    def touch_target_nohold(self, **kwargs):
        if self.tht == 0.0 and not self.target_index == self.num_targets:
            if self.testing:
                self.tic_touch_targ_nohold_start = time.time()
            if self.drag_ok:
                if self.check_if_cursors_in_targ(self.active_target_position, self.eff_target_rad):
                    if self.testing:
                        self.tic_touch_target = time.time()
                    istargtouch = True
                else:
                    istargtouch = False
            else:
                if np.logical_and(self.check_if_cursors_in_targ(self.active_target_position, self.eff_target_rad),
                    self.check_if_started_in_targ(self.active_target_position, self.eff_target_rad)):
                    if self.testing:
                        self.tic_touch_target = time.time()
                    istargtouch = True
                else:
                    istargtouch = False
            
            if istargtouch:
                # run the equivalent of finish_targ_hold without having to go through another loop
                if self.target_index == 1:
                    self.target1.color = (0., 0., 0., 0.)
                elif self.target_index == 2:
                    self.target2.color = (0., 0., 0., 0.)
                elif self.target_index == 3:
                    self.target3.color = (0., 0., 0., 0.)
                elif self.target_index == 4:
                    self.target4.color = (0., 0., 0., 0.)
                elif self.target_index == 5:
                    self.target5.color = (0., 0., 0., 0.)
                if not self.intertarg_delay == 0:
                    self.pd1_indicator_targ.color = (1., 1., 1., 1.)
                
                # Play a small reward tone
                self.reward2.play()
                self.target_index += 1
                
                # Need to reset this for the next target
                self.first_time_for_this_targ = True
                
                if self.testing:
                    self.tic_touch_targ_nohold_end = time.time()
                
            return istargtouch
        else:
            return False

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
                if self.target_index == 1:
                    self.target1.color = (0., 0., 0., 0.)
                elif self.target_index == 2:
                    self.target2.color = (0., 0., 0., 0.)
                elif self.target_index == 3:
                    self.target3.color = (0., 0., 0., 0.)
                elif self.target_index == 4:
                    self.target4.color = (0., 0., 0., 0.)
                elif self.target_index == 5:
                    self.target5.color = (0., 0., 0., 0.)
                if not self.intertarg_delay == 0:
                    self.pd1_indicator_targ.color = (1., 1., 1., 1.)
                
                # Play a small reward tone
                # sound = SoundLoader.load('reward2.wav')
                # sound.play()
                self.reward2.play()
                self.target_index += 1
                
                # Need to reset this for the next target
                self.first_time_for_this_targ = True
                return True
            else:
                return False
        else:
            return False
        
    def finish_last_targ_hold(self, **kwargs):
        if self.target_index == self.num_targets:
            if self.tht <= kwargs['ts']:
                if self.target_index == 1:
                    self.target1.color = (0., 0., 0., 0.)
                elif self.target_index == 2:
                    self.target2.color = (0., 0., 0., 0.)
                elif self.target_index == 3:
                    self.target3.color = (0., 0., 0., 0.)
                elif self.target_index == 4:
                    self.target4.color = (0., 0., 0., 0.)
                elif self.target_index == 5:
                    self.target5.color = (0., 0., 0., 0.)
                self.trial_completion_time = time.time() - self.target1_on_time
                return True
            else:
                return False
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
        
    def set_alloutlinetargs_color(self, r, g, b, a):
        self.target1_out.color = (r, g, b, a)
        self.target1_in.color = (r, g, b, a)
        self.target2_out.color = (r, g, b, a)
        self.target2_in.color = (r, g, b, a)
        self.target3_out.color = (r, g, b, a)
        self.target3_in.color = (r, g, b, a)
        self.target4_out.color = (r, g, b, a)
        self.target4_in.color = (r, g, b, a)
        self.target5_out.color = (r, g, b, a)
        self.target5_in.color = (r, g, b, a)
        self.target6_out.color = (r, g, b, a)
        self.target6_in.color = (r, g, b, a)
        self.target7_out.color = (r, g, b, a)
        self.target7_in.color = (r, g, b, a)
        self.target8_out.color = (r, g, b, a)
        self.target8_in.color = (r, g, b, a)
        self.target9_out.color = (r, g, b, a)
        self.target9_in.color = (r, g, b, a)
        
    def set_targoutline_color(self, r, g, b, a):
        self.target1_out.move(np.array([self.center_position[0], self.center_position[1]]))
        self.target1_in.move(np.array([self.center_position[0], self.center_position[1]]))
        self.target2_out.move(np.array([self.center_position[0], self.center_position[1] + self.max_y_from_center]))
        self.target2_in.move(np.array([self.center_position[0], self.center_position[1] + self.max_y_from_center]))
        self.target3_out.move(np.array([self.center_position[0], self.center_position[1] - self.max_y_from_center]))
        self.target3_in.move(np.array([self.center_position[0], self.center_position[1] - self.max_y_from_center]))
        self.target4_out.move(np.array([self.max_y_from_center, self.center_position[1] + self.max_y_from_center]))
        self.target4_in.move(np.array([self.max_y_from_center, self.center_position[1] + self.max_y_from_center]))
        self.target5_out.move(np.array([self.max_y_from_center, self.center_position[1]]))
        self.target5_in.move(np.array([self.max_y_from_center, self.center_position[1]]))
        self.target6_out.move(np.array([self.max_y_from_center, self.center_position[1] - self.max_y_from_center]))
        self.target6_in.move(np.array([self.max_y_from_center, self.center_position[1] - self.max_y_from_center]))
        self.target7_out.move(np.array([-self.max_y_from_center, self.center_position[1] + self.max_y_from_center]))
        self.target7_in.move(np.array([-self.max_y_from_center, self.center_position[1] + self.max_y_from_center]))
        self.target8_out.move(np.array([-self.max_y_from_center, self.center_position[1]]))
        self.target8_in.move(np.array([-self.max_y_from_center, self.center_position[1]]))
        self.target9_out.move(np.array([-self.max_y_from_center, self.center_position[1] - self.max_y_from_center]))
        self.target9_in.move(np.array([-self.max_y_from_center, self.center_position[1] - self.max_y_from_center]))
        
        
        self.target1_out.color = (r, g, b, a)
        self.target1_in.color = (0., 0., 0., a)
        self.target2_out.color = (r, g, b, a)
        self.target2_in.color = (0., 0., 0., a)
        self.target3_out.color = (r, g, b, a)
        self.target3_in.color = (0., 0., 0., a)
        self.target4_out.color = (r, g, b, a)
        self.target4_in.color = (0., 0., 0., a)
        self.target5_out.color = (r, g, b, a)
        self.target5_in.color = (0., 0., 0., a)
        self.target6_out.color = (r, g, b, a)
        self.target6_in.color = (0., 0., 0., a)
        self.target7_out.color = (r, g, b, a)
        self.target7_in.color = (0., 0., 0., a)
        self.target8_out.color = (r, g, b, a)
        self.target8_in.color = (0., 0., 0., a)
        self.target9_out.color = (r, g, b, a)
        self.target9_in.color = (0., 0., 0., a)
    
    def ind2sub(array_shape, ind):
        rows = (ind / array_shape[1])
        cols = (ind % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
        return (rows, cols)
    
    def get_targpos_from_str(self, pos_str, nudge_x):
        if pos_str == 'random': # set for now, will get overriden later
            targ_x = self.center_position[0]
            targ_y = self.center_position[1]
        elif pos_str == 'center':
            targ_x = self.center_position[0]+nudge_x
            targ_y = self.center_position[1]
        elif pos_str == 'upper_middle':
            targ_x = self.center_position[0]+nudge_x
            targ_y = self.center_position[1] + self.max_y_from_center
        elif pos_str == 'lower_middle':
            targ_x = self.center_position[0]+nudge_x
            targ_y = self.center_position[1] - self.max_y_from_center
        elif pos_str == 'upper_right':
            targ_x = self.max_y_from_center+nudge_x
            targ_y = self.center_position[1] + self.max_y_from_center
        elif pos_str == 'middle_right':
            targ_x = self.max_y_from_center+nudge_x
            targ_y = self.center_position[1]
        elif pos_str == 'lower_right':
            targ_x = self.max_y_from_center+nudge_x
            targ_y = self.center_position[1] - self.max_y_from_center
        elif pos_str == 'lower_left':
            targ_x = -self.max_y_from_center+nudge_x
            targ_y = self.center_position[1] - self.max_y_from_center
        elif pos_str == 'middle_left':
            targ_x = -self.max_y_from_center+nudge_x
            targ_y = self.center_position[1]
        elif pos_str == 'upper_left':
            targ_x = -self.max_y_from_center+nudge_x
            targ_y = self.center_position[1] + self.max_y_from_center
            
        return np.array([targ_x, targ_y])
        
    def make_random_sequence(self, change_str):
        # Get the position of random targets
        pos_str_opts = ['upper_left', 'upper_middle', 'upper_right', 'middle_left', 'center', 'middle_right', 'lower_left', 'lower_middle', 'lower_right']
        
        seq_poss_ix = np.random.randint(0, np.shape(self.seq_poss)[0])
        
        if self.target1_pos_str == 'random': 
            i_pos = self.seq_poss[seq_poss_ix, 0]-1
            targ1_pos = self.get_targpos_from_str(pos_str_opts[i_pos], self.nudge_x_t1)
    
            self.target1_position = targ1_pos 
            if change_str:
                self.target1_pos_str = pos_str_opts[i_pos]
            
        if self.target2_pos_str == 'random': 
            i_pos = self.seq_poss[seq_poss_ix, 1]-1
            targ2_pos = self.get_targpos_from_str(pos_str_opts[i_pos], self.nudge_x_t2)
    
            self.target2_position = targ2_pos 
            if change_str:
                self.target2_pos_str = pos_str_opts[i_pos]
            
        if self.target3_pos_str == 'random': 
            i_pos = self.seq_poss[seq_poss_ix, 2]-1
            targ3_pos = self.get_targpos_from_str(pos_str_opts[i_pos], self.nudge_x_t3)
    
            self.target3_position = targ3_pos 
            if change_str:
                self.target3_pos_str = pos_str_opts[i_pos]
            
        if self.target4_pos_str == 'random': 
            i_pos = self.seq_poss[seq_poss_ix, 3]-1
            targ4_pos = self.get_targpos_from_str(pos_str_opts[i_pos], self.nudge_x_t4)
    
            self.target4_position = targ4_pos 
            if change_str:
                self.target4_pos_str = pos_str_opts[i_pos]
            
        if self.target5_pos_str == 'random': 
            i_pos = self.seq_poss[seq_poss_ix, 4]-1
            targ5_pos = self.get_targpos_from_str(pos_str_opts[i_pos], self.nudge_x_t4)
    
            self.target5_position = targ5_pos 
            if change_str:
                self.target5_pos_str = pos_str_opts[i_pos]
        
    

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
    # juicer version
    try:
        juicer_r = BooleanProperty(data_params['juicer'] == 'red')
        juicer_y = BooleanProperty(data_params['juicer'] == 'yellow')
    except: 
        juicer_r = BooleanProperty(False)
        juicer_y = BooleanProperty(False)
        
    # button version
    try:
        is_button_fsr = BooleanProperty(data_params['button_version'] == 'fsr')
        is_button_ir = BooleanProperty(data_params['button_version'] == 'ir')
    except: 
        is_button_fsr = BooleanProperty(False)
        is_button_ir = BooleanProperty(False)
        
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
    is_t1tt0pt8 = BooleanProperty(False)
    is_t1tt1pt0 = BooleanProperty(False)
    is_t1tt1pt5 = BooleanProperty(False)
    is_t1tt2pt0 = BooleanProperty(False)
    is_t1tt2pt5 = BooleanProperty(False)
    is_t1tt3pt0 = BooleanProperty(False)
    is_t1tt3pt5 = BooleanProperty(False)
    is_t1tt4pt0 = BooleanProperty(False)
    is_t1tt10pt0 = BooleanProperty(False)
    try:
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
    is_tt0pt9 = BooleanProperty(False)
    is_tt1pt0 = BooleanProperty(False)
    is_tt1pt1 = BooleanProperty(False)
    is_tt1pt2 = BooleanProperty(False)
    is_tt1pt3 = BooleanProperty(False)
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
        elif data_params['target_timeout_time'] == 0.9:
            is_tt0pt9 = BooleanProperty(True)
        elif data_params['target_timeout_time'] == 1.0:
            is_tt1pt0 = BooleanProperty(True)
        elif data_params['target_timeout_time'] == 1.1:
            is_tt1pt1 = BooleanProperty(True)
        elif data_params['target_timeout_time'] == 1.2:
            is_tt1pt2 = BooleanProperty(True)
        elif data_params['target_timeout_time'] == 1.3:
            is_tt1pt3 = BooleanProperty(True)
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
        
        
    # minimum final target reward
    is_minthrewnone = BooleanProperty(False)
    is_minthrew000 = BooleanProperty(False)
    is_minthrew100 = BooleanProperty(False)
    is_minthrew200 = BooleanProperty(False)
    is_minthrew300 = BooleanProperty(False)
    is_minthrew400 = BooleanProperty(False)
    is_minthrew500 = BooleanProperty(False)
    is_minthrew600 = BooleanProperty(False)
    is_minthrew700 = BooleanProperty(False)
    try:
        if data_params['min_targ_reward'] == 'No Reward Scaling':
            is_minthrewnone = BooleanProperty(True)
        elif data_params['min_targ_reward'] == 0.0:
            is_minthrew000 = BooleanProperty(True)
        elif data_params['min_targ_reward'] == 0.1:
            is_minthrew100 = BooleanProperty(True)
        elif data_params['min_targ_reward'] == 0.2:
            is_minthrew200 = BooleanProperty(True)
        elif data_params['min_targ_reward'] == 0.3:
            is_minthrew300 = BooleanProperty(True)
        elif data_params['min_targ_reward'] == 0.4:
            is_minthrew400 = BooleanProperty(True)
        elif data_params['min_targ_reward'] == 0.5:
            is_minthrew500 = BooleanProperty(True)
        elif data_params['min_targ_reward'] == 0.6:
            is_minthrew600 = BooleanProperty(True)
        elif data_params['min_targ_reward'] == 0.7:
            is_minthrew700 = BooleanProperty(True)
    except:
        pass
        
        
    # max final target reward
    is_threw300 = BooleanProperty(False)
    is_threw500 = BooleanProperty(False)
    is_threw700 = BooleanProperty(False)
    is_threw900 = BooleanProperty(False)
    is_threw1100 = BooleanProperty(False)
    try:
        if data_params['last_targ_reward'] == 0.3:
            is_threw300 = BooleanProperty(True)
        elif data_params['last_targ_reward'] == 0.5:
            is_threw500 = BooleanProperty(True)
        elif data_params['last_targ_reward'] == 0.7:
            is_threw700 = BooleanProperty(True)
        elif data_params['last_targ_reward'] == 0.9:
            is_threw900 = BooleanProperty(True)
        elif data_params['last_targ_reward'] == 1.1:
            is_threw1100 = BooleanProperty(True)
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
    
    # display outlines
    is_outlines = BooleanProperty(False)
    is_no_outlines = BooleanProperty(False)

    try:
        if data_params['display_outlines'] == True:
            is_outlines     = BooleanProperty(True)
        elif data_params['display_outlines'] == False:
            is_no_outlines  = BooleanProperty(True)
    except: 
        pass
        
    # sequence preselect
    is_seqA = BooleanProperty(False)
    is_seqB = BooleanProperty(False)
    is_seqC = BooleanProperty(False)
    is_seqD = BooleanProperty(False)
    is_seqE = BooleanProperty(False)
    is_seqF = BooleanProperty(False)
    is_seqG = BooleanProperty(False)
    is_seqH = BooleanProperty(False)
    is_seqI = BooleanProperty(False)
    is_seqJ = BooleanProperty(False)
    is_seqK = BooleanProperty(False)
    is_seqL = BooleanProperty(False)
    is_seqM = BooleanProperty(False)
    is_seqN = BooleanProperty(False)
    is_seqO = BooleanProperty(False)
    is_seqP = BooleanProperty(False)
    is_seqQ = BooleanProperty(False)
    is_seqR = BooleanProperty(False)
    is_seqS = BooleanProperty(False)
    is_seqT = BooleanProperty(False)
    is_seqU = BooleanProperty(False)
    is_seqV = BooleanProperty(False)
    is_seqW = BooleanProperty(False)
    is_seqX = BooleanProperty(False)
    is_seqY = BooleanProperty(False)
    is_seqRand5 = BooleanProperty(False)
    is_seqRepeat = BooleanProperty(False)
    is_seqRandomEvery = BooleanProperty(False)
    is_seqRand5RandEvery = BooleanProperty(False)
    is_seq2repeat = BooleanProperty(False)
    is_CO = BooleanProperty(False)
    is_BO = BooleanProperty(False)
    try:
        if data_params['seq'] == 'A':
            is_seqA = BooleanProperty(True)
        elif data_params['seq'] == 'B':
            is_seqB = BooleanProperty(True)
        elif data_params['seq'] == 'C':
            is_seqC = BooleanProperty(True) 
        elif data_params['seq'] == 'D':
            is_seqD = BooleanProperty(True) 
        elif data_params['seq'] == 'E':
            is_seqE = BooleanProperty(True) 
        elif data_params['seq'] == 'F':
            is_seqF = BooleanProperty(True) 
        elif data_params['seq'] == 'G':
            is_seqG = BooleanProperty(True) 
        elif data_params['seq'] == 'H':
            is_seqH = BooleanProperty(True) 
        elif data_params['seq'] == 'I':
            is_seqI = BooleanProperty(True) 
        elif data_params['seq'] == 'J':
            is_seqJ = BooleanProperty(True) 
        elif data_params['seq'] == 'K':
            is_seqK = BooleanProperty(True) 
        elif data_params['seq'] == 'L':
            is_seqL = BooleanProperty(True) 
        elif data_params['seq'] == 'M':
            is_seqM = BooleanProperty(True) 
        elif data_params['seq'] == 'N':
            is_seqN = BooleanProperty(True) 
        elif data_params['seq'] == 'O':
            is_seqO = BooleanProperty(True) 
        elif data_params['seq'] == 'P':
            is_seqP = BooleanProperty(True) 
        elif data_params['seq'] == 'Q':
            is_seqQ = BooleanProperty(True) 
        elif data_params['seq'] == 'R':
            is_seqR = BooleanProperty(True) 
        elif data_params['seq'] == 'S':
            is_seqS = BooleanProperty(True) 
        elif data_params['seq'] == 'T':
            is_seqT = BooleanProperty(True) 
        elif data_params['seq'] == 'U':
            is_seqU = BooleanProperty(True) 
        elif data_params['seq'] == 'V':
            is_seqV = BooleanProperty(True) 
        elif data_params['seq'] == 'W':
            is_seqW = BooleanProperty(True) 
        elif data_params['seq'] == 'X':
            is_seqX = BooleanProperty(True) 
        elif data_params['seq'] == 'Y':
            is_seqY = BooleanProperty(True) 
        elif data_params['seq'] == 'rand5':
            is_seqRand5 = BooleanProperty(True) 
        elif data_params['seq'] == 'repeat':
            is_seqRepeat = BooleanProperty(True) 
        elif data_params['seq'] == 'randevery':
            is_seqRandomEvery = BooleanProperty(True) 
        elif data_params['seq'] == 'rand5-randevery':
            is_seqRand5RandEvery = BooleanProperty(True) 
        elif data_params['seq'] == 'center out':
            is_CO = BooleanProperty(True) 
        elif data_params['seq'] == 'button out': 
            is_BO = BooleanProperty(True) 
        elif data_params['seq'] == '2seq-repeat': 
            is_seq2repeat = BooleanProperty(True) 
    except: 
        pass

    # target 1 position
    is_t1cent = BooleanProperty(False)
    is_t1um = BooleanProperty(False)
    is_t1lm = BooleanProperty(False)
    is_t1ul = BooleanProperty(False)
    is_t1ml = BooleanProperty(False)
    is_t1ll = BooleanProperty(False)
    is_t1ur = BooleanProperty(False)
    is_t1mr = BooleanProperty(False)
    is_t1lr = BooleanProperty(False)
    is_t1rand = BooleanProperty(False) 
    try:
        if data_params['target1_pos_str'] == 'center':
            is_t1cent = BooleanProperty(True)
        elif data_params['target1_pos_str'] == 'upper_middle':
            is_t1um = BooleanProperty(True)
        elif data_params['target1_pos_str'] == 'lower_middle':
            is_t1lm = BooleanProperty(True) 
        elif data_params['target1_pos_str'] == 'upper_left':
            is_t1ul = BooleanProperty(True)
        elif data_params['target1_pos_str'] == 'middle_left':
            is_t1ml = BooleanProperty(True)
        elif data_params['target1_pos_str'] == 'lower_left':
            is_t1ll = BooleanProperty(True)
        elif data_params['target1_pos_str'] == 'upper_right':
            is_t1ur = BooleanProperty(True)
        elif data_params['target1_pos_str'] == 'middle_right':
            is_t1mr = BooleanProperty(True)
        elif data_params['target1_pos_str'] == 'lower_right':
            is_t1lr = BooleanProperty(True)
        elif data_params['target1_pos_str'] == 'random': 
            is_t1rand = BooleanProperty(True)
    except:
        pass
        
    # target 1 nudge
    is_t1nudgeneg6 = BooleanProperty(False)
    is_t1nudgeneg4 = BooleanProperty(False)
    is_t1nudgeneg2 = BooleanProperty(False)
    is_t1nudgezero = BooleanProperty(False)
    is_t1nudgepos2 = BooleanProperty(False)
    is_t1nudgepos4 = BooleanProperty(False)
    is_t1nudgepos6 = BooleanProperty(False)
    try:
        if data_params['nudge_x_t1'] == -6:
            is_t1nudgeneg6 = BooleanProperty(True)
        elif data_params['nudge_x_t1'] == -4:
            is_t1nudgeneg4 = BooleanProperty(True)
        elif data_params['nudge_x_t1'] == -2:
            is_t1nudgeneg2 = BooleanProperty(True)
        elif data_params['nudge_x_t1'] == 0:
            is_t1nudgezero = BooleanProperty(True)
        elif data_params['nudge_x_t1'] == 2:
            is_t1nudgepos2 = BooleanProperty(True)
        elif data_params['nudge_x_t1'] == 4:
            is_t1nudgepos4 = BooleanProperty(True)
        elif data_params['nudge_x_t1'] == 6:
            is_t1nudgepos6 = BooleanProperty(True)
    except:
        pass
        
    # target 2 position
    is_t2cent = BooleanProperty(False)
    is_t2um = BooleanProperty(False)
    is_t2lm = BooleanProperty(False)
    is_t2ul = BooleanProperty(False)
    is_t2ml = BooleanProperty(False)
    is_t2ll = BooleanProperty(False)
    is_t2ur = BooleanProperty(False)
    is_t2mr = BooleanProperty(False)
    is_t2lr = BooleanProperty(False)
    is_t2rand = BooleanProperty(False)
    is_t2none = BooleanProperty(False)
    try:
        if data_params['target2_pos_str'] == 'center':
            is_t2cent = BooleanProperty(True)
        elif data_params['target2_pos_str'] == 'upper_middle':
            is_t2um = BooleanProperty(True)
        elif data_params['target2_pos_str'] == 'lower_middle':
            is_t2lm = BooleanProperty(True) 
        elif data_params['target2_pos_str'] == 'upper_left':
            is_t2ul = BooleanProperty(True)
        elif data_params['target2_pos_str'] == 'middle_left':
            is_t2ml = BooleanProperty(True)
        elif data_params['target2_pos_str'] == 'lower_left':
            is_t2ll = BooleanProperty(True)
        elif data_params['target2_pos_str'] == 'upper_right':
            is_t2ur = BooleanProperty(True)
        elif data_params['target2_pos_str'] == 'middle_right':
            is_t2mr = BooleanProperty(True)
        elif data_params['target2_pos_str'] == 'lower_right':
            is_t2lr = BooleanProperty(True)
        elif data_params['target2_pos_str'] == 'random':
            is_t2rand = BooleanProperty(True)
        elif data_params['target2_pos_str'] == 'none': 
            is_t2none = BooleanProperty(True)
    except:
        pass
        
    # target 2 nudge
    is_t2nudgeneg6 = BooleanProperty(False)
    is_t2nudgeneg4 = BooleanProperty(False)
    is_t2nudgeneg2 = BooleanProperty(False)
    is_t2nudgezero = BooleanProperty(False)
    is_t2nudgepos2 = BooleanProperty(False)
    is_t2nudgepos4 = BooleanProperty(False)
    is_t2nudgepos6 = BooleanProperty(False)
    try:
        if data_params['nudge_x_t2'] == -6:
            is_t2nudgeneg6 = BooleanProperty(True)
        elif data_params['nudge_x_t2'] == -4:
            is_t2nudgeneg4 = BooleanProperty(True)
        elif data_params['nudge_x_t2'] == -2:
            is_t2nudgeneg2 = BooleanProperty(True)
        elif data_params['nudge_x_t2'] == 0:
            is_t2nudgezero = BooleanProperty(True)
        elif data_params['nudge_x_t2'] == 2:
            is_t2nudgepos2 = BooleanProperty(True)
        elif data_params['nudge_x_t2'] == 4:
            is_t2nudgepos4 = BooleanProperty(True)
        elif data_params['nudge_x_t2'] == 6:
            is_t2nudgepos6 = BooleanProperty(True)
    except:
        pass
        
    # target 3 position
    is_t3cent = BooleanProperty(False)
    is_t3um = BooleanProperty(False)
    is_t3lm = BooleanProperty(False)
    is_t3ul = BooleanProperty(False)
    is_t3ml = BooleanProperty(False)
    is_t3ll = BooleanProperty(False)
    is_t3ur = BooleanProperty(False)
    is_t3mr = BooleanProperty(False)
    is_t3lr = BooleanProperty(False)
    is_t3none = BooleanProperty(False)
    try:
        if data_params['target3_pos_str'] == 'center':
            is_t3cent = BooleanProperty(True)
        elif data_params['target3_pos_str'] == 'upper_middle':
            is_t3um = BooleanProperty(True)
        elif data_params['target3_pos_str'] == 'lower_middle':
            is_t3lm = BooleanProperty(True) 
        elif data_params['target3_pos_str'] == 'upper_left':
            is_t3ul = BooleanProperty(True)
        elif data_params['target3_pos_str'] == 'middle_left':
            is_t3ml = BooleanProperty(True)
        elif data_params['target3_pos_str'] == 'lower_left':
            is_t3ll = BooleanProperty(True)
        elif data_params['target3_pos_str'] == 'upper_right':
            is_t3ur = BooleanProperty(True)
        elif data_params['target3_pos_str'] == 'middle_right':
            is_t3mr = BooleanProperty(True)
        elif data_params['target3_pos_str'] == 'lower_right':
            is_t3lr = BooleanProperty(True)
        elif data_params['target3_pos_str'] == 'none':
            is_t3none = BooleanProperty(True)
    except:
        pass
        
    # target 3 nudge
    is_t3nudgeneg6 = BooleanProperty(False)
    is_t3nudgeneg4 = BooleanProperty(False)
    is_t3nudgeneg2 = BooleanProperty(False)
    is_t3nudgezero = BooleanProperty(False)
    is_t3nudgepos2 = BooleanProperty(False)
    is_t3nudgepos4 = BooleanProperty(False)
    is_t3nudgepos6 = BooleanProperty(False)
    try:
        if data_params['nudge_x_t3'] == -6:
            is_t3nudgeneg6 = BooleanProperty(True)
        elif data_params['nudge_x_t3'] == -4:
            is_t3nudgeneg4 = BooleanProperty(True)
        elif data_params['nudge_x_t3'] == -2:
            is_t3nudgeneg2 = BooleanProperty(True)
        elif data_params['nudge_x_t3'] == 0:
            is_t3nudgezero = BooleanProperty(True)
        elif data_params['nudge_x_t3'] == 2:
            is_t3nudgepos2 = BooleanProperty(True)
        elif data_params['nudge_x_t3'] == 4:
            is_t3nudgepos4 = BooleanProperty(True)
        elif data_params['nudge_x_t3'] == 6:
            is_t3nudgepos6 = BooleanProperty(True)
    except:
        pass
    
    
    # target 4 position
    is_t4cent = BooleanProperty(False)
    is_t4um = BooleanProperty(False)
    is_t4lm = BooleanProperty(False)
    is_t4ul = BooleanProperty(False)
    is_t4ml = BooleanProperty(False)
    is_t4ll = BooleanProperty(False)
    is_t4ur = BooleanProperty(False)
    is_t4mr = BooleanProperty(False)
    is_t4lr = BooleanProperty(False)
    is_t4none = BooleanProperty(False)
    try:
        if data_params['target4_pos_str'] == 'center':
            is_t4cent = BooleanProperty(True)
        elif data_params['target4_pos_str'] == 'upper_middle':
            is_t4um = BooleanProperty(True)
        elif data_params['target4_pos_str'] == 'lower_middle':
            is_t4lm = BooleanProperty(True) 
        elif data_params['target4_pos_str'] == 'upper_left':
            is_t4ul = BooleanProperty(True)
        elif data_params['target4_pos_str'] == 'middle_left':
            is_t4ml = BooleanProperty(True)
        elif data_params['target4_pos_str'] == 'lower_left':
            is_t4ll = BooleanProperty(True)
        elif data_params['target4_pos_str'] == 'upper_right':
            is_t4ur = BooleanProperty(True)
        elif data_params['target4_pos_str'] == 'middle_right':
            is_t4mr = BooleanProperty(True)
        elif data_params['target4_pos_str'] == 'lower_right':
            is_t4lr = BooleanProperty(True)
        elif data_params['target4_pos_str'] == 'none':
            is_t4none = BooleanProperty(True)
    except:
        pass
        
    # target 4 nudge
    is_t4nudgeneg6 = BooleanProperty(False)
    is_t4nudgeneg4 = BooleanProperty(False)
    is_t4nudgeneg2 = BooleanProperty(False)
    is_t4nudgezero = BooleanProperty(False)
    is_t4nudgepos2 = BooleanProperty(False)
    is_t4nudgepos4 = BooleanProperty(False)
    is_t4nudgepos6 = BooleanProperty(False)
    try:
        if data_params['nudge_x_t4'] == -6:
            is_t4nudgeneg6 = BooleanProperty(True)
        elif data_params['nudge_x_t4'] == -4:
            is_t4nudgeneg4 = BooleanProperty(True)
        elif data_params['nudge_x_t4'] == -2:
            is_t4nudgeneg2 = BooleanProperty(True)
        elif data_params['nudge_x_t4'] == 0:
            is_t4nudgezero = BooleanProperty(True)
        elif data_params['nudge_x_t4'] == 2:
            is_t4nudgepos2 = BooleanProperty(True)
        elif data_params['nudge_x_t4'] == 4:
            is_t4nudgepos4 = BooleanProperty(True)
        elif data_params['nudge_x_t4'] == 6:
            is_t4nudgepos6 = BooleanProperty(True)
    except:
        pass
    
    # target 5 position
    is_t5cent = BooleanProperty(False)
    is_t5um = BooleanProperty(False)
    is_t5lm = BooleanProperty(False)
    is_t5ul = BooleanProperty(False)
    is_t5ml = BooleanProperty(False)
    is_t5ll = BooleanProperty(False)
    is_t5ur = BooleanProperty(False)
    is_t5mr = BooleanProperty(False)
    is_t5lr = BooleanProperty(False)
    is_t5none = BooleanProperty(False)
    try:
        if data_params['target5_pos_str'] == 'center':
            is_t5cent = BooleanProperty(True)
        elif data_params['target5_pos_str'] == 'upper_middle':
            is_t5um = BooleanProperty(True)
        elif data_params['target5_pos_str'] == 'lower_middle':
            is_t5lm = BooleanProperty(True) 
        elif data_params['target5_pos_str'] == 'upper_left':
            is_t5ul = BooleanProperty(True)
        elif data_params['target5_pos_str'] == 'middle_left':
            is_t5ml = BooleanProperty(True)
        elif data_params['target5_pos_str'] == 'lower_left':
            is_t5ll = BooleanProperty(True)
        elif data_params['target5_pos_str'] == 'upper_right':
            is_t5ur = BooleanProperty(True)
        elif data_params['target5_pos_str'] == 'middle_right':
            is_t5mr = BooleanProperty(True)
        elif data_params['target5_pos_str'] == 'lower_right':
            is_t5lr = BooleanProperty(True)
        elif data_params['target5_pos_str'] == 'none':
            is_t5none = BooleanProperty(True)
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
    is_screenbot0 = BooleanProperty(False)
    is_screenbot2 = BooleanProperty(False)
    is_screenbot4 = BooleanProperty(False)
    is_screenbot6 = BooleanProperty(False)
    is_screenbot8 = BooleanProperty(False)
    is_screenbot10 = BooleanProperty(False)
    is_screenbot12 = BooleanProperty(False)
    try:
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
    
    # intertarg delay
    is_inttargdelay0 = BooleanProperty(False)
    is_inttargdelay100 = BooleanProperty(False)
    is_inttargdelay150 = BooleanProperty(False)
    is_inttargdelay200 = BooleanProperty(False)
    is_inttargdelay250 = BooleanProperty(False)
    is_inttargdelay300 = BooleanProperty(False)
    is_inttargdelay400 = BooleanProperty(False)
    is_inttargdelay500 = BooleanProperty(False)
    try:
        if data_params['intertarg_delay'] == 0:
            is_inttargdelay0 = BooleanProperty(True)
        elif data_params['intertarg_delay'] == 0.1:
            is_inttargdelay100 = BooleanProperty(True)
        elif data_params['intertarg_delay'] == 0.15:
            is_inttargdelay150 = BooleanProperty(True)
        elif data_params['intertarg_delay'] == 0.2:
            is_inttargdelay200 = BooleanProperty(True)
        elif data_params['intertarg_delay'] == 0.25:
            is_inttargdelay250 = BooleanProperty(True)
        elif data_params['intertarg_delay'] == 0.3:
            is_inttargdelay300 = BooleanProperty(True)
        elif data_params['intertarg_delay'] == 0.4:
            is_inttargdelay400 = BooleanProperty(True)
        elif data_params['intertarg_delay'] == 0.5:
            is_inttargdelay500 = BooleanProperty(True)
    except:
        pass
    
    # break after x trials
    is_nobreak = BooleanProperty(False)
    is_break10 = BooleanProperty(False)
    is_break15 = BooleanProperty(False)
    is_break20 = BooleanProperty(False)
    is_break25 = BooleanProperty(False)
    try:
        if data_params['break_trl'] == 0:
            is_nobreak = BooleanProperty(True)
        elif data_params['break_trl'] == 10:
            is_break10 = BooleanProperty(True)
        elif data_params['break_trl'] == 15:
            is_break15 = BooleanProperty(True)
        elif data_params['break_trl'] == 20:
            is_break20 = BooleanProperty(True)
        elif data_params['break_trl'] == 25:
            is_break25 = BooleanProperty(True)
    except:
        pass
    
    # break duration
    is_breakdur30 = BooleanProperty(False)
    is_breakdur60 = BooleanProperty(False)
    is_breakdur90 = BooleanProperty(False)
    is_breakdur120 = BooleanProperty(False)
    is_breakdur150 = BooleanProperty(False)
    try:
        if data_params['break_dur'] == 30:
            is_breakdur30 = BooleanProperty(True)
        elif data_params['break_dur'] == 60:
            is_breakdur60 = BooleanProperty(True)
        elif data_params['break_dur'] == 90:
            is_breakdur90 = BooleanProperty(True)
        elif data_params['break_dur'] == 120:
            is_breakdur120 = BooleanProperty(True)
        elif data_params['break_dur'] == 150:
            is_breakdur150 = BooleanProperty(True)
    except:
        pass
    
    # auto quit after
    is_autoqt10 = BooleanProperty(False)
    is_autoqt25 = BooleanProperty(False)
    is_autoqt50 = BooleanProperty(False)
    is_autoqt60 = BooleanProperty(False)
    is_autoqt90 = BooleanProperty(False)
    is_autoqt100 = BooleanProperty(False)
    is_autoqtnever = BooleanProperty(False)
    try:
        if data_params['max_trials'] == 10:
            is_autoqt10 = BooleanProperty(True)
        elif data_params['max_trials'] == 25:
            is_autoqt25 = BooleanProperty(True)
        elif data_params['max_trials'] == 50:
            is_autoqt50 = BooleanProperty(True)
        elif data_params['max_trials'] == 60:
            is_autoqt60 = BooleanProperty(True)
        elif data_params['max_trials'] == 90:
            is_autoqt90 = BooleanProperty(True)
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
