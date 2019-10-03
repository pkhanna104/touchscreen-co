"""
190930
1. Move on to the phase 3
2. Add time difference between two targets
3. Some minor changes
"""

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
import serial, time, pickle, datetime, winsound
from numpy import binary_repr
import struct
import scipy.io


Config.set('graphics', 'resizable', False)
fixed_window_size = (1800, 1000)
pix_per_cm = 85.
Config.set('graphics', 'width', str(fixed_window_size[0]))
Config.set('graphics', 'height', str(fixed_window_size[1]))

import time
import numpy as np
import tables


class Data(tables.IsDescription):
    state = tables.StringCol(24)   # 24-character String
    cursor = tables.Float32Col(shape=(10, 2))
    cursor_ids = tables.Float32Col(shape = (10, ))
    target1_pos = tables.Float32Col(shape=(2, ))
    target2_pos = tables.Float32Col(shape=(2, ))
    cap_touch = tables.Float32Col()
    time = tables.Float32Col()

class COGame(Widget):
    center = ObjectProperty(None)
    target = ObjectProperty(None)
    # first_target_attempt_t0 = time.time();

    # Time to wait after starting the video before getting to the center target display. 
    pre_start_vid_ts = 0.1

    ITIs_mean = 1.
    ITIf_mean = 2.
    ITI_std = .2
    center_target_rad = 1.5
    periph_target_rad = 1.5

    exit_pos = np.array([7, 4])
    indicator_pos = np.array([8, 5])
    exit_rad = 1.
    exit_hold = 2 #seconds

    ch_timeout = 10. # ch timeout
    cht = .001 # center hold time

    target_timeout_time = 5.
    tht = .001

    cursor = {}
    cursor_start = {}
    cursor_ids = []

    anytouch_prev = False
    touch_error_timeout = 3.
    timeout_error_timeout = 3.
    hold_error_timeout = 3.
    drag_error_timeout = 3.

    ntargets = 2.
    target_distance = 6. 
    touch = False

    center_target = ObjectProperty(None)
    periph_target1 = ObjectProperty(None)
    periph_target2 = ObjectProperty(None)
    stims = ObjectProperty(None)

    done_init = False
    prev_exit_ts = np.array([0,0])

    # Number of trials: 
    trial_counter = NumericProperty(0)
    tc_text = StringProperty('')
    total_counter = NumericProperty(0)
    #indicator_txt = StringProperty('o')
    #indicator_txt_color = ListProperty([.5, .5, .5, 1.])

    t0 = time.time()

    cht_text = StringProperty('')
    tht_text = StringProperty('')
    generatorz_text = StringProperty('')
    targ_size_text = StringProperty('')
    big_rew_text = StringProperty('')
    cht_param = StringProperty('')
    tht_param = StringProperty('')
    targ_size_param = StringProperty('')
    big_rew_time_param = StringProperty('')
    # generatorz_param = StringProperty('')
    nudge_text = StringProperty('')
    nudge_param = StringProperty('')

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
            

# dict(stims=[stim_on.active, stim_off.active, stim_rand.active])

# dict(donu=chk_h.active, sand=chk_sand.active,sabo=chk_sabo.active), 

# dict(tdbt=[tdbt_7.active, tdbt_5.active, tdbt_3.active, tdbt_2.active, tdbt_1.active,
#    tdbt_0.active, tdbt_7_4.active, tdbt_3_0.active])

# dict(rew_manual=False, rew_anytouch=False, rew_center_pls_targ=rew_center_pls_targ_chk.active, rew_targ=rew_targ_chk.active, snd_only=False, 
#    small_rew=[small_rew_pt1_sec.active, small_rew_pt3_sec.active, small_rew_pt5_sec.active, small_rew_pt7_sec.active], big_rew=[big_rew_pt3_sec.active, big_rew_pt5_sec.active, big_rew_pt7_sec.active]), 

# dict(targ_rad=[targ_rad_10.active, targ_rad_12.active, targ_rad_15.active]), 

# dict(test=[False, True, False]), 

# dict(chold=[chold_16_20.active, chold_18_22.active, chold_20_25.active, chold_23_27.active, chold_25_30.active, chold_28_32.active, chold_30_35.active]), 

# dict(iti = [iti_10_20.active, iti_15_25.active, iti_20_30.active]),

## dict(get_targets_rand=False, get_4targets=False, get_targets_co=True),

# dict(autoquit=[ten_trials.active, twenty_five_trials.active, fifty_trials.active, hundred_trials.active, no_trials.active]),

# dict(rew_var=[rew_all.active, rew_50.active, rew_30.active, rew_easy.active, rew_medium.active, rew_hard.active]),

# dict(tt=[tt_15_sec.active, tt_30_sec.active, tt_45_sec.active, tt_60_sec.active]))


    def init(self, stims=None, animal_names_dict=None, tdbt=None, rew_in=None, task_in=None,
        test=None, hold=None, iti=None,
        autoquit=None, rew_var=None, targ_timeout = None, targ_pos=None):

        print('')
        print('')

        self.rew_cnt = 0
        self.small_rew_cnt = 0

        # self.use_cap_sensor = False
        # if self.use_cap_sensor:
        #     self.serial_port_cap = serial.Serial(port='COM5')
        # self.rhtouch_sensor = 0.

        stims_opts = ['stim_on', 'stim_off', 'stim_rand']
        for i, val in enumerate(stims['stims']):
            if val:
                self.stims = stims_opts[i]
        print("----- Stimulation type:", self.stims)
        if self.stims =='stim_rand':
            self.stim_order = np.random.permutation(1000)
            self.stim_order = self.stim_order%2
        elif self.stims == 'stim_on':
            self.stim_order = np.ones((1000,), dtype=int)
        elif self.stims == 'stim_off':
            self.stim_order = np.zeros((1000,), dtype=int)

        # if stims==['stim_on']:
        #     self.stims = 'on'
        # elif stims==['stim_off']:
        #     self.stims = 'off'
        # elif stims==['stim_rand']:
        #     self.stims = 'rand'

        tdbt_opt = [0.7, 0.5, 0.3, 0.2, 0.1, 0.0, '0.7-0.4', '0.3-0.0']
        self.tdbt_type = None; 
        for i, val in enumerate(tdbt['tdbt']):
            if val:
                if type(tdbt_opt[i]) is str:
                    mx, mn = tdbt_opt[i].split('-')
                    self.tdbt_type = tdbt_opt[i]
                    self.tdbt_mn = float(mn)
                    self.tdbt_mx = float(mx)
                    self.tdbt = (float(mn)+float(mx))*.5
                else:
                    self.tdbt = tdbt_opt[i]
        print('TDBT %.2f' %self.tdbt)
        targ_timeout_opts = [15, 30, 45, 60]
        for i, val in enumerate(targ_timeout['tt']):
            if val:
                self.target_timeout_time = targ_timeout_opts[i]

        small_rew_opts = [.1, .3, .5]
        for i, val in enumerate(rew_in['small_rew']):
            if val:
                small_rew = small_rew_opts[i]

        big_rew_opts = [.3, .5, .7]
        for i, val in enumerate(rew_in['big_rew']):
            if val:
                big_rew = big_rew_opts[i]

        if np.logical_or(rew_in['rew_targ'], rew_in['rew_center_pls_targ']):
            self.reward_for_targtouch = [True, big_rew]
        else:
            self.reward_for_targtouch = [False, 0]

        if rew_in['rew_center_pls_targ']:
            self.reward_for_center = [True, small_rew]
        else:
            self.reward_for_center = [False, 0]
        self.skip_juice = False

        target_rad_opts = [1.0, 1.2, 1.5]
        for i, val in enumerate(task_in['targ_rad']):
            if val:
                self.periph_target_rad = target_rad_opts[i]
                self.center_target_rad = target_rad_opts[i]



        for i, (nm, val) in enumerate(animal_names_dict.items()):
            if val:
                animal_name = nm

        self.use_center = True
        # for i, (nm, val) in enumerate(targ_structure.items()):
        #     if val:
        #         generatorz = getattr(self, nm)
        #         self.generatorz_param2 = nm
        #         if 'co' in nm:
        #             self.use_center = True

        # self.testing = False; 
        # self.in_cage = False; 

        choldz = ['1.6-2.0', '1.8-2.2', '2.0-2.5', '2.3-2.7', '2.5-3.0', '2.8-3.2', '3.0-3.5']
        tholdz = .375
        self.tht_type = None
        self.cht_type = None; 

        for i, val in enumerate(hold['chold']):
            if val:
                if type(choldz[i]) is str:
                    mx, mn = choldz[i].split('-')
                    self.cht_type = choldz[i]
                    self.cht = (float(mn)+float(mx))*.5
                else:
                    self.cht = choldz[i]
        self.tht = tholdz

        try:
            pygame.mixer.init()    
        except:
            pass




        ITI_opt = ['1.0-2.0', '1.5-2.5', '2.0-3.0']
        self.ITI_type = None; 
        for i, val in enumerate(iti['iti']):
            if val:
                if type(ITI_opt[i]) is str:
                    mx, mn = ITI_opt[i].split('-')
                    self.ITI_type = ITI_opt[i]
                    self.ITI_s = float(mn)
                    self.ITI_f = float(mx)

        
        self.reward_delay_time = 0.0
        reward_var_opt = [1.0, .5, .33]
        for i, val in enumerate(rew_var['rew_var']):
            if val:
                self.percent_of_trials_rewarded = reward_var_opt[i]
                if self.percent_of_trials_rewarded == 0.33:
                    self.percent_of_trials_doubled = 0.1
                else:
                    self.percent_of_trials_doubled = 0.0
        
        ### Game initiates ####
        try:
            pygame.mixer.init()    
        except:
            pass

        self.reward_generator = self.gen_rewards(self.percent_of_trials_rewarded, self.percent_of_trials_doubled,
            self.reward_for_targtouch)


        self.use_white_screen = True

        test_vals = [True, False, False]
        in_cage_vals = [False, False, True]
        for i, val in enumerate(test['test']):
            if val:
                self.testing = test_vals[i]
                self.in_cage = in_cage_vals[i]

        autoquit_trls = [10, 25, 50, 100, 10**10]
        for i, val in enumerate(autoquit['autoquit']):
            if val: 
                self.max_trials = autoquit_trls[i]


        # targ_timeout_opts = [15, 30, 45, 60]
        # for i, val in enumerate(targ_timeout['tt']):
        #     if val:
        #         self.target_timeout_time = targ_timeout_opts[i]
        #         self.ch_timeout = targ_timeout_opts[i]
        # 
        # self.reward_for_anytouch = False; 
        # 
        # for i, (nm, val) in enumerate(animal_names_dict.items()):
        #     if val:
        #         animal_name = nm



        self.drag_ok = False;
        self.nudge_dist = 0.
        self.generator_kwarg = 'corners'

        # Preload sounds: 
        self.reward1 = SoundLoader.load('reward1.wav')

        self.state = 'ITI'
        self.state_start = time.time()
        self.ITI = self.ITI_std + self.ITIs_mean

        # Initlize center target to be in the lower left corner: 
        self.center_target.set_size(2*self.center_target_rad)
        self.center_target.move(np.array([-4.24264069, -2.5]))

        # Initlize peripheral targets to be in 2 and 4 o'clock from the center target: 
        self.periph_target1.set_size(2*self.periph_target_rad)
        self.periph_target2.set_size(2*self.periph_target_rad)
        self.periph_target1_pos = np.array([-2.51, 1.5]) 
        self.periph_target2_pos = np.array([0, -1.5]) 
        self.periph_target1.move(self.periph_target1_pos)
        self.periph_target2.move(self.periph_target2_pos)

        ## Keep exit targets; 
        self.exit_target1.set_size(2*self.exit_rad)
        self.exit_target2.set_size(2*self.exit_rad)

        ## Keep the photodiode target 
        self.indicator_targ.set_size(self.exit_rad)
        self.indicator_targ.move(self.indicator_pos)
        self.indicator_targ.color = (0., 0., 0., 1.)

        # this scene is appeared at just before the very first trial of the task
        self.exit_target1.move(self.exit_pos)
        self.exit_pos2 = np.array([self.exit_pos[0], -1*self.exit_pos[1]])
        self.exit_target2.move(self.exit_pos2)
        self.exit_target1.color = (.5, .5, .5, 1)
        self.exit_target2.color = (.5, .5, .5, 1)

# what are these?
        # self.target_list = generatorz(self.target_distance, self.nudge_dist, self.generator_kwarg)
        # self.successed_trials = 0
        self.repeat = False

        self.center_target_position = np.array([-4.24264069, -2.5])
        self.periph_target1_position = self.periph_target1_pos
        self.periph_target2_position = self.periph_target2_pos

        self.FSM = dict()
        self.FSM['ITI'] = dict(end_ITI='vid_trig', stop=None)
        
        self.FSM['vid_trig'] = dict(rhtouch='center', stop=None)
        self.FSM['center'] = dict(touch_center='center_hold', center_timeout='timeout_error', non_rhtouch='RH_touch',stop=None)
        self.FSM['center_hold'] = dict(finish_center_hold='target1', early_leave_center_hold='hold_error', non_rhtouch='RH_touch', stop=None)

        self.FSM['target1'] = dict(dummy = 'tdbt', stop=None) # anytouch='rew_anytouch',touch_not_target='touch_error')
        self.FSM['tdbt'] = dict(dummy = 'target2', stop=None)
        self.FSM['target2'] = dict(touch_target = 'targ_hold', target_timeout='timeout_error', stop=None,
            non_rhtouch='RH_touch') # anytouch='rew_anytouch',touch_not_target='touch_error')
        self.FSM['targ_hold'] = dict(finish_targ_hold='reward', early_leave_target_hold = 'hold_error',
         targ_drag_out = 'drag_error', stop=None, non_rhtouch='RH_touch')

        self.FSM['reward'] = dict(end_reward = 'ITI', stop=None, non_rhtouch='RH_touch')

        return_ = 'center'

        self.FSM['touch_error'] = dict(end_touch_error=return_, stop=None, non_rhtouch='RH_touch')
        self.FSM['hold_error'] = dict(end_hold_error='ITI', stop=None, non_rhtouch='RH_touch')
        self.FSM['drag_error'] = dict(end_drag_error=return_, stop=None, non_rhtouch='RH_touch')
        self.FSM['timeout_error'] = dict(end_timeout_error='ITI', stop=None, non_rhtouch='RH_touch')
        self.FSM['idle_exit'] = dict(stop=None)

        ### Initialize the reward 
        try:
            self.reward_port = serial.Serial(port='COM4', baudrate=115200)
            self.reward_port.close()
        except:
            pass

        ### Initialize the DIO
        try:
            self.dio_port = serial.Serial(port='COM5', baudrate=115200)
            time.sleep(4.)
        except:
            pass

        ### Initialize the Camera triggers
        try:
            self.cam_trig_port = serial.Serial(port='COM6', baudrate=9600)
            time.sleep(3.)
            # Say hello: 
            self.cam_trig_port.write('a'.encode())

            # Start cams @ 50 Hz
            self.cam_trig_port.write('1'.encode())
        except:
            pass

        ### Initialize the stim trigger
        try:
            self.stim_port = serial.Serial(port='COM7', baudrate=115200)
            time.sleep(1.)
            # Say hello: 
            # self.stim_port.write('c'.encode())
        except:
            pass

        # save parameters: 
        d = dict(animal_name=animal_name, center_target_rad=self.center_target_rad,
            periph_target_rad=self.periph_target_rad, 
            # target_list = self.target_list, 
            stims = self.stims, stim_order = self.stim_order,
            time_diff_btw_targets = self.tdbt, 
            ITIs_mean=self.ITIs_mean, ITIf_mean=self.ITIf_mean, ITI_std = self.ITI_std, 
            ch_timeout=self.ch_timeout, 
            cht=self.cht, reward_time_small=self.reward_for_center[1],
            reward_time_big=self.reward_for_targtouch[1],
            # reward_for_anytouch=self.reward_for_anytouch[0],
            reward_for_center = self.reward_for_center[0],
            reward_for_targtouch=self.reward_for_targtouch[0], 
            touch_error_timeout = self.touch_error_timeout,
            timeout_error_timeout = self.timeout_error_timeout,
            hold_error_timeout = self.hold_error_timeout,
            drag_error_timeout = self.drag_error_timeout,
            ntargets = self.ntargets,
            target_distance = self.target_distance,
            start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M'),
            testing=self.testing,
            rew_delay = self.reward_delay_time,
            # use_cap_sensor = self.use_cap_sensor,
            drag_ok = self.drag_ok,
            )

        print("----- Reward_for_center: ", self.reward_for_center)
        print("----- Reward_for_target: ", self.reward_for_targtouch)
        # print(self.reward_for_anytouch)

        try:
            if self.testing:
                pass
            else:
                import os
                path = os.getcwd()
                print ('test for cwd: ', path)
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
                self.filename = p+ animal_name+'_ISP_'+datetime.datetime.now().strftime('%Y%m%d_%H%M')
                if self.in_cage:
                    self.filename = self.filename+'_cage'

                pickle.dump(d, open(self.filename+'_params.pkl', 'wb'))
                scipy.io.savemat(self.filename+'_params.mat', mdict={'tpara': d})
                self.h5file = tables.open_file(self.filename + '_data.hdf', mode='w', title = 'NHP data')
                self.h5_table = self.h5file.create_table('/', 'task', Data, '')
                self.h5_table_row = self.h5_table.row
                self.h5_table_row_cnt = 0

                    # Note in python 3 to open pkl files: 
                    #with open('xxxx_params.pkl', 'rb') as f:
                    #    data_params = pickle.load(f)
        except:
            pass


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

    def close_app(self):
        # Save Data Eventually
         #Stop the video: 
        try:
            self.cam_trig_port.write('0'.encode())
        except:
            pass

        # if self.use_cap_sensor:
        #     self.serial_port_cap.close()
        
        if self.idle:
            self.state = 'idle_exit'

            # self.trial_counter = -1
            self.trial_counter = str(self.trial_counter)

            # Set relevant params text: 
            self.cht_text = 'Center Hold Time: '
            self.tht_text = 'Target Hold Time: '
            self.generatorz_text = 'Target Structure: '
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

            self.targ_size_param = str(self.center_target_rad)
            self.big_rew_time_param = str(self.reward_for_targtouch[1])
            # self.generatorz_param = self.generatorz_param2

            self.nudge_text = 'Total trials? '
            self.nudge_param = str(self.total_counter)
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
            else:
                while_state_fn_name = "_while_%s" % self.state
                if hasattr(self, while_state_fn_name):
                    while_state_fn = getattr(self, while_state_fn_name)
                    while_state_fn()
            
        # if self.use_cap_sensor:
        #     try:
        #         self.serial_port_cap.flushInput()
        #         port_read = self.serial_port_cap.read(4)
        #         if str(port_read[:2]) == "b'N1'":
        #             self.rhtouch_sensor = False
        #         elif str(port_read[:2]) == "b'C1'":
        #             self.rhtouch_sensor = True
        #             print(self.rhtouch_sensor)
        #     except:
        #         print('passing state! ')
        #         pass     
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

        cursor_id = np.zeros((10, ))
        cursor_id[:] = np.nan
        cursor_id[:len(self.cursor_ids)] = self.cursor_ids
        self.h5_table_row['cursor_ids'] = cursor_id

        self.h5_table_row['target1_pos'] = self.periph_target1_position
        self.h5_table_row['target2_pos'] = self.periph_target2_position
        self.h5_table_row['time'] = time.time() - self.t0
        # self.h5_table_row['cap_touch'] = self.rhtouch_sensor
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
        bgcolor = (.62, .32, 0.17, 1.)
        Window.clearcolor = bgcolor
        self.exit_target1.color = bgcolor
        self.exit_target2.color = bgcolor

        # Set ITI, CHT, THT
        if self.reward_for_targtouch[0]:
            self.ITI = np.random.random()*self.ITI_std + self.ITI_s
        else:
            self.ITI = np.random.random()*self.ITI_std + self.ITI_f

        if type(self.cht_type) is str:
            cht_min, cht_max = self.cht_type.split('-')
            self.cht = ((float(cht_max) - float(cht_min)) * np.random.random()) + float(cht_min)

        if type(self.tht_type) is str:
            tht_min, tht_max = self.tht_type.split('-')
            self.tht = ((float(tht_max) - float(tht_min)) * np.random.random()) + float(tht_min)            
        
        self.center_target.color = bgcolor
        self.periph_target1.color = bgcolor
        self.periph_target2.color = bgcolor
        self.indicator_targ.color = (0., 0., 0., 0.)
        
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

        # ??
        # if np.logical_and(self.use_cap_sensor, not self.rhtouch_sensor):
        #     self.center_target.color = (1., 0., 0., 1.)
        #     self.periph_target1.color = (.62, .32, 0.17, 1.)
        #     self.periph_target2.color = (.62, .32, 0.17, 1.)
        #     Window.clearcolor = (1., 0., 0., 1.)

        #     # Turn exit buttons redish:
        #     self.exit_target1.color = (.9, 0, 0, 1.)
        #     self.exit_target2.color = (.9, 0, 0, 1.)
         

    def end_vid_trig(self, **kwargs):
        return kwargs['ts'] > self.pre_start_vid_ts
    


    def rhtouch(self, **kwargs):
        return True
    #     if self.use_cap_sensor:
    #         if self.rhtouch_sensor:
    #             return True
    #         else:
    #             return False
    #     else:
    #         return True

    def non_rhtouch(self, **kwargs):
        x = not self.rhtouch()
        # if x:
        #     self.repeat = True
        return x

    def _start_center(self, **kwargs):
        if self.stim_order[self.total_counter] == 0:
            self.stims = 'stim_off'
        elif self.stim_order[self.total_counter] == 1:
            self.stims = 'stim_on'
        print('')            
        print('Trial num: ', self.trial_counter,'/', self.total_counter)
        print(self.stims)
        print("----- Start to push the center button")
        # if self.stims == 'stim_on':
        #     bgcolor = (0., 0., 0., 1.)
        # elif self.stims == 'stim_off':
        #     bgcolor = (1., 1., 1., 1.)
        bgcolor = (.5, .5, .5, 1.)
        Window.clearcolor = bgcolor
        self.center_target.color = (1., 0., 0., 1.)
        self.periph_target1.color = bgcolor
        self.periph_target2.color = bgcolor
        self.exit_target1.color = bgcolor
        self.exit_target2.color = bgcolor
        self.indicator_targ.color = (.25, .25, .25, 1.)

    def _start_center_hold(self, **kwargs):
        print("----- Start to hold the center button")
        if self.stims == 'stim_on':
            self.stim_port.write('1'.encode())
            print("---------- Stim Start !")
        elif self.stims == 'stim_off':
            print("---------- No Stim Start !")
        self.center_target.color = (0.70, 0., 0., 1.)
        self.indicator_targ.color = (0.75, .75, .75, 1.)

    # ??: It passed quickly but not skipped. It exist!
    def _end_center_hold(self, **kwargs):
        if self.stims == 'stim_on':
            self.stim_port.write('0'.encode())
            print("---------- End center hold, Stim Off !")
        elif self.stims == 'stim_off':
            print("---------- End center hold, No Stim Off !")
        Window.clearcolor = (.8, .9, 1., 1.)
        self.center_target.color = (0., 0., 0., 1.)

    def _start_target1(self, **kwargs):
        print("----- Start to push the target button")
        bgcolor = (0.5, 0.5, 0.5, 1.)
        Window.clearcolor = bgcolor
        self.center_target.color = bgcolor
        if self.stims == 'stim_on':
            self.periph_target1.color = (0., 0., 0., 1.)
        elif self.stims == 'stim_off':
            self.periph_target2.color = (1., 1., 1., 1.)

    def _start_tdbt(self, **kwargs):
        self.run_tdbt()

    def _start_target2(self, **kwargs):
        if self.stims == 'stim_on':
            self.periph_target2.color = (1., 1., 1., 1.)
        elif self.stims == 'stim_off':
            self.periph_target1.color = (0., 0., 0., 1.)
        
        if self.repeat is False:
            self.periph_target1_position = self.periph_target1_pos
            self.periph_target2_position = self.periph_target2_pos
            # self.successed_trials += 1
            # print(self.periph_target1_position)
            # print(self.periph_target2_position)
            # print(self.successed_trials)

        #self.periph_target.move(self.periph_target_position)
        #self.periph_target.color = (1., 1., 0., 1.)
        self.repeat = False
        self.exit_target1.color = (.5, .5, .5, 1)
        self.exit_target2.color = (.5, .5, .5, 1)
        self.indicator_targ.color = (.25, .25, .25, 1.)
        if self.first_target_attempt:
            self.first_target_attempt_t0 = time.time();
            self.first_target_attempt = False

    def _start_targ_hold(self, **kwargs):
        print("----- Start to hold the target button")
        bgcolor = (.5, .5, .5, 1.)
        Window.clearcolor = bgcolor
        if self.stims == 'stim_on':
            tgcolor = (0.2, 0.2, 0.2, 1.)
            self.periph_target1.color = tgcolor
            self.periph_target2.color = bgcolor
        elif self.stims == 'stim_off':
            tgcolor = (0.8, 0.8, 0.8, 1.)
            self.periph_target1.color = bgcolor
            self.periph_target2.color = tgcolor
        self.indicator_targ.color = (0.75, .75, .75, 1.)

    # def _end_target_hold(self, **kwargs):
    #     print("---------- Hold Success !")
    #     self.periph_target1.color = (0, 0, .5, 1.)
    #     self.periph_target2.color = (0, 0, .5, 1.)

    # def _start_touch_error(self, **kwargs):
    #     if self.stims == 'stim_on':
    #         self.stim_port.write('er'.encode())
    #         print("---------- Stim Error ! : start touch error")
    #     elif self.stims == 'stim_off':
    #         print("---------- No Stim Error ! : start touch error")
    #     bgcolor = (.62, .32, 0.17, 1.)
    #     Window.clearcolor = bgcolor
    #     self.center_target.color = bgcolor
    #     self.periph_target1.color = bgcolor
    #     self.periph_target2.color = bgcolor
    #     self.exit_target1.color = bgcolor
    #     self.exit_target2.color = bgcolor
    #     self.total_counter += 1
    #     self.repeat = True

    def _start_timeout_error(self, **kwargs):
        print("-------------X- Timeout Error !")
        if self.stims == 'stim_on':
            self.stim_port.write('er'.encode())
        bgcolor = (.62, .32, 0.17, 1.)
        Window.clearcolor = bgcolor
        self.center_target.color = bgcolor
        self.periph_target1.color = bgcolor
        self.periph_target2.color = bgcolor
        self.exit_target1.color = bgcolor
        self.exit_target2.color = bgcolor
        self.total_counter += 1
        self.repeat = True

    def _start_hold_error(self, **kwargs):
        print("-------------X- Hold Error !")
        if self.stims == 'stim_on':
            self.stim_port.write('er'.encode())
        bgcolor = (.62, .32, 0.17, 1.)
        Window.clearcolor = bgcolor
        self.center_target.color = bgcolor
        self.periph_target1.color = bgcolor
        self.periph_target2.color = bgcolor
        self.exit_target1.color = bgcolor
        self.exit_target2.color = bgcolor
        self.total_counter += 1
        self.repeat = True

    def _start_drag_error(self, **kwargs):
        print("-------------X- Drag Error !")
        if self.stims == 'stim_on':
            self.stim_port.write('er'.encode())
        bgcolor = (.62, .32, 0.17, 1.)
        Window.clearcolor = bgcolor
        self.center_target.color = bgcolor
        self.periph_target1.color = bgcolor
        self.periph_target2.color = bgcolor
        self.exit_target1.color = bgcolor
        self.exit_target2.color = bgcolor
        # self.total_counter += 1
        self.repeat = True

    def _start_reward(self, **kwargs):
        self.trial_counter += 1
        self.total_counter += 1
        bgcolor = (.62, .32, 0.17, 1.)
        Window.clearcolor = bgcolor
        self.center_target.color = bgcolor
        self.periph_target1.color = bgcolor
        self.periph_target2.color = bgcolor
        self.exit_target1.color = bgcolor
        self.exit_target2.color = bgcolor
        self.rew_cnt = 0
        self.cnts_in_rew = 0
        self.indicator_targ.color = (1., 1., 1., 1.)
        self.repeat = False

    def _while_reward(self, **kwargs):
        if self.rew_cnt == 1:
            self.run_big_rew()
            self.rew_cnt += 1

    # def _start_rew_anytouch(self, **kwargs):
    #     #if self.small_rew_cnt == 1:
    #     if self.reward_for_anytouch[0]:
    #         self.run_small_rew()
    #     else:
    #         self.repeat = True
    #         #self.small_rew_cnt += 1

    def run_tdbt(self, **kwargs):
        try:
            print('--------------- in tdbt')       
            time.sleep(self.tdbt)
        except:
            pass

    def run_big_rew(self, **kwargs):
        try:
            print('--------------- in big reward:')
            self.repeat = False
            if self.reward_for_targtouch[0]:
                #winsound.PlaySound('beep1.wav', winsound.SND_ASYNC)
                #sound = SoundLoader.load('reward1.wav')
                print('-------------O- reward_for_targtouch')
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
            print('--------------- in small reward:')            
            # if np.logical_or(self.reward_for_anytouch[0], self.reward_for_center[0]):
            if self.reward_for_center[0]:
                #winsound.PlaySound('beep1.wav', winsound.SND_ASYNC)
                print('-------------O- reward_for_centertouch')
                sound = SoundLoader.load('reward2.wav')
                sound.play()
                
                self.reward_port.open()
                # if self.reward_for_anytouch[0]:
                #     rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.reward_for_anytouch[1])+' sec\n']
                if self.reward_for_center[0]:
                    rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.reward_for_center[1])+' sec\n']
                self.reward_port.write(rew_str)
                time.sleep(.25)
                run_str = [ord(r) for r in 'run\n']
                self.reward_port.write(run_str)
                self.reward_port.close()
        except:
            pass

        #self.repeat = True

    def end_reward(self, **kwargs):
        self.indicator_txt_color = (1.,1., 1., 1.)
        if self.use_white_screen:
            if len(self.cursor_ids)== 0:
                return True
        else:
            if self.cnts_in_rew > 30:
                return True
            else:
                self.cnts_in_rew += 1
                return False

    # def end_rewanytouch(self, **kwargs):
    #     if self.small_rew_cnt > 1:
    #         return True
    #     else:
    #         return False

    def end_touch_error(self, **kwargs):
        return kwargs['ts'] >= self.touch_error_timeout

    def end_timeout_error(self, **kwargs):
        return kwargs['ts'] >= self.timeout_error_timeout

    def end_hold_error(self, **kwargs):
        return kwargs['ts'] >= self.hold_error_timeout

    def end_drag_error(self, **kwargs):
        return kwargs['ts'] >= self.drag_error_timeout

    def touch_center(self, **kwargs):
        if self.drag_ok:
            return self.check_if_cursors_in_targ(self.center_target_position, self.center_target_rad)
        else:
            return np.logical_and(self.check_if_cursors_in_targ(self.center_target_position, self.center_target_rad),
                self.check_if_started_in_targ(self.center_target_position, self.center_target_rad))

    def center_timeout(self, **kwargs):
        if self.stims == 'stim_on':
            self.stim_port.write('0'.encode())
        return kwargs['ts'] > self.ch_timeout

    def finish_center_hold(self, **kwargs):
        if self.cht <= kwargs['ts']:
            if self.stims == 'stim_on':
                self.stim_port.write('0'.encode())
            if self.reward_for_targtouch[0]:
                self.run_small_rew()
            return True
        else:
            return False
    def dummy(self, **kwargs):
        return True

    def early_leave_center_hold(self, **kwargs):
        if self.stims == 'stim_on':
            self.stim_port.write('0'.encode())        
        return not self.check_if_cursors_in_targ(self.center_target_position, self.center_target_rad)
        
    # def center_drag_out(self, **kwargs):
    #     touch = self.touch
    #     self.touch = True
    #     stay_in = self.check_if_cursors_in_targ(self.center_target_position, self.center_target_rad)
    #     self.touch = touch
    #     return not stay_in

    def touch_target(self, **kwargs):
        if self.drag_ok:
            # if self.stims == 'stim_on':
            #     return self.check_if_cursors_in_targ(self.periph_target1_position, self.periph_target_rad)
            # elif self.stims == 'stim_off':
            #     return self.check_if_cursors_in_targ(self.periph_target2_position, self.periph_target_rad)
            return np.logical_or(self.check_if_cursors_in_targ(self.periph_target1_position, self.periph_target_rad),self.check_if_cursors_in_targ(self.periph_target2_position, self.periph_target_rad))

        else:
            # if self.stims == 'stim_on':
            #     return np.logical_and(self.check_if_cursors_in_targ(self.periph_target1_position, self.periph_target_rad),
            #         self.check_if_started_in_targ(self.periph_target1_position, self.periph_target_rad))
            # elif self.stims == 'stim_off':
            #     return np.logical_and(self.check_if_cursors_in_targ(self.periph_target2_position, self.periph_target_rad),
            #         self.check_if_started_in_targ(self.periph_target2_position, self.periph_target_rad))
            return np.logical_or(np.logical_and(self.check_if_cursors_in_targ(self.periph_target1_position, self.periph_target_rad),
                self.check_if_started_in_targ(self.periph_target1_position, self.periph_target_rad)),
                np.logical_and(self.check_if_cursors_in_targ(self.periph_target2_position, self.periph_target_rad),
                self.check_if_started_in_targ(self.periph_target2_position, self.periph_target_rad)))

    def target_timeout(self, **kwargs):
        #return kwargs['ts'] > self.target_timeout_time
        if time.time() - self.first_target_attempt_t0 > self.target_timeout_time:
            self.repeat = False
            return True

    def finish_targ_hold(self, **kwargs):
        return self.tht <= kwargs['ts']

    def early_leave_target_hold(self, **kwargs):
        if self.stims == 'stim_on':
            return not self.check_if_cursors_in_targ(self.periph_target1_position, self.periph_target_rad)
        elif self.stims == 'stim_off':
            return not self.check_if_cursors_in_targ(self.periph_target2_position, self.periph_target_rad)

    def targ_drag_out(self, **kwargs):
        touch = self.touch
        self.touch = True
        if self.stims == 'stim_on':
            stay_in = self.check_if_cursors_in_targ(self.periph_target1_position, self.periph_target_rad)
        elif self.stims == 'stim_off':
            stay_in = self.check_if_cursors_in_targ(self.periph_target2_position, self.periph_target_rad)
        self.touch = touch
        return not stay_in

    def anytouch(self, **kwargs):
        if not self.touch_target():
            current_touch = len(self.cursor_ids) > 0
            rew = False
            if current_touch and not self.anytouch_prev:
                rew = True
            self.anytouch_prev = current_touch
            return rew
        else:
            return False

    # def get_4targets(self, target_distance=4, nudge=0., gen_kwarg=None):
    #     return self.get_targets_co(target_distance=target_distance, nudge=0.)

    def get_targets_co(self, target_distance=4, nudge=0., gen_kwarg=None, ntargets=4):
        # Targets in CM: 
        if gen_kwarg ==  'corners':
            angle = np.linspace(0, 2*np.pi, 5)[:-1] + (np.pi/4.)
            target_distance = 6.
        else:
            angle = np.linspace(0, 2*np.pi, ntargets+1)[:-1]
    
        x = np.cos(angle)*target_distance
        y = np.sin(angle)*target_distance
        tmp = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
        nudge_targ = np.array([0, 0, 1., 0])

        tgs = []
        nudges = []
        for blks in range(100):
            ix = np.random.permutation(tmp.shape[0])
            tgs.append(tmp[ix, :])
            nudges.append(nudge_targ[ix])

        tgs = np.vstack((tgs))
        nudges = np.hstack((nudges))
        nudge_ix = np.nonzero(nudges==1)[0]
        print('Nudges: ')
        print(len(nudge_ix))

        to_nudge = np.array([-1., 1.])*nudge
        tgs[nudge_ix, :] = tgs[nudge_ix, :] + to_nudge[np.newaxis, :]

        return tgs

    # def get_targets_rand(self, target_distance=4):
    #     # Targets in CM: 
    #     angle = np.linspace(0, 2*np.pi, 1000)
    #     target_distance = np.linspace(target_distance/4., target_distance, 1000)

    #     ix_ang = np.random.permutation(1000)
    #     ix_dist = np.random.permutation(1000)

    #     x = np.cos(angle[ix_ang])*target_distance[ix_dist]
    #     y = np.sin(angle[ix_ang])*target_distance[ix_dist]
    #     return np.hstack((x[:, np.newaxis], y[:, np.newaxis]))

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
    pass

class COApp(App):
    def build(self, **kwargs):
        from win32api import GetSystemMetrics
        screenx = GetSystemMetrics(0)
        screeny = GetSystemMetrics(1)
        Window.size = (1800, 1000)
        Window.left = (screenx - 1800)/2
        Window.top = (screeny - 1000)/2
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

if __name__ == '__main__':
    COApp().run()
