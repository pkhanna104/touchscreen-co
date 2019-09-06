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
    target_pos = tables.Float32Col(shape=(2, ))
    cap_touch = tables.Float32Col()
    time = tables.Float32Col()

class COGame(Widget):
    center = ObjectProperty(None)
    target = ObjectProperty(None)

    # Time to wait after starting the video before getting to the center target display. 
    pre_start_vid_ts = 0.1

    ITI_mean = 2.
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

    ntargets = 4.
    target_distance = 6. 
    touch = False

    center_target = ObjectProperty(None)
    periph_target = ObjectProperty(None)

    done_init = False
    prev_exit_ts = np.array([0,0])

    # Number of trials: 
    trial_counter = NumericProperty(0)
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
    generatorz_param = StringProperty('')
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
            

# root.current = 'splash_start'
# splash.init(
# dict(donu=chk_h.active, sand=chk_sand.active,sabo=chk_sabo.active), 

# dict(reward=[small_rew_pt1_sec.active, small_rew_pt3_sec.active, small_rew_pt5_sec.active, small_rew_pt7_sec.active]), 

# dict(targ_rad=[targ_rad_5.active, targ_rad_75.active, targ_rad_82.active, targ_rad_91.active, targ_rad_10.active, targ_rad_15.active]), 

# dict(test=[False, True, False]), 

# dict(hold=[hold_4_6.active, hold_4_7.active, hold_5_8.active, hold_6_10.active, hold_7_13.active, hold_8_15.active, hold_9_17.active, hold_10_17.active, hold_12_20.active, hold_15_25.active, hold_20_30.active],

# dict(autoquit=[ten_trials.active, twenty_five_trials.active, fifty_trials.active, hundred_trials.active, no_trials.active]),

# dict(rew_var=[rew_all.active, rew_50.active, rew_30.active]),

# dict(tt=[tt_15_sec.active, tt_30_sec.active, tt_45_sec.active, tt_60_sec.active]))


    def init(self, animal_names_dict=None, rew_in=None, task_in=None,
        hold=None, autoquit=None, rew_var=None, targ_timeout = None):

        self.rew_cnt = 0
        self.small_rew_cnt = 0

        self.use_cap_sensor = False
        if self.use_cap_sensor:
            self.serial_port_cap = serial.Serial(port='COM5')
        self.rhtouch_sensor = 0.


        rew_opts = [.1, .3, .5, .7]
        for i, val in enumerate(rew_in['small_rew']):
            if val:
                small_rew = rew_opts[i]
                self.reward_for_targtouch = [True, small_rew]
        self.reward_for_center = [False, 0.]
        self.reward_for_anytouch = [False, 0.]

        target_rad_opts = [.5, .75, .82, .91, 1.0, 1.5]
        for i, val in enumerate(task_in['targ_rad']):
            if val:
                self.periph_target_rad = target_rad_opts[i]
                self.center_target_rad = target_rad_opts[i]

        self.testing = False; 
        self.in_cage = False; 

        holdz = ['.4-.6', '.4-.7', '.5-.8', '.6-1.0', '.7-1.3', '.8-1.5', '.9-1.7', '1.0-1.7', '1.2-2.0', '1.5-2.5', '2.0-3.0']
        self.tht_type = None
        self.cht_type = None; 

        for i, val in enumerate(hold['hold']):
            if val:
                if type(holdz[i]) is str:

                    mx, mn = holdz[i].split('-')
                    self.tht_type = holdz[i]
                    self.tht =  (float(mn)+float(mx))*.5

                    self.cht_type = holdz[i]
                    self.cht = (float(mn)+float(mx))*.5
                else:
                    self.tht = holdz[i]
                    self.cht = holdz[i]
        
        autoquit_trls = [10, 25, 50, 100, 10**10]
        for i, val in enumerate(autoquit['autoquit']):
            if val: 
                self.max_trials = autoquit_trls[i]


        targ_timeout_opts = [15, 30, 45, 60]
        for i, val in enumerate(targ_timeout['tt']):
            if val:
                self.target_timeout_time = targ_timeout_opts[i]
                self.ch_timeout = targ_timeout_opts[i]

        self.reward_for_anytouch = False; 
        self.skip_juice = False

        for i, (nm, val) in enumerate(animal_names_dict.items()):
            if val:
                animal_name = nm


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
        self.drag_ok = False;
        self.nudge_dist = 0.

        # Preload sounds: 
        self.reward1 = SoundLoader.load('reward1.wav')

        self.state = 'ITI'
        self.state_start = time.time()
        self.ITI = self.ITI_std + self.ITI_mean

        # Initialize targets: 
        self.center_target.set_size(2*self.center_target_rad)
        
        # Initlize center target to be in the lower left corner: 
        self.center_target.move(np.array([-4.24264069, -4.24264069]))

        self.periph_target.set_size(2*self.periph_target_rad)

        ## Keep exit targets; 
        self.exit_target1.set_size(2*self.exit_rad)
        self.exit_target2.set_size(2*self.exit_rad)

        ## Keep the photodiode target 
        self.indicator_targ.set_size(self.exit_rad)
        self.indicator_targ.move(self.indicator_pos)
        self.indicator_targ.color = (0., 0., 0., 1.)

        self.exit_target1.move(self.exit_pos)
        self.exit_pos2 = np.array([self.exit_pos[0], -1*self.exit_pos[1]])
        self.exit_target2.move(self.exit_pos2)
        self.exit_target1.color = (.15, .15, .15, 1)
        self.exit_target2.color = (.15, .15, .15, 1)

        self.center_target_position = np.array([-4.24264069, -4.24264069])

        self.FSM = dict()
        self.FSM['ITI'] = dict(end_ITI='vid_trig', stop=None)
        self.FSM['vid_trig'] = dict(rhtouch='center', stop=None)
        self.FSM['center'] = dict(touch_center='center_hold', center_timeout='timeout_error', non_rhtouch='RH_touch',stop=None)    
        self.FSM['center_hold'] = dict(finish_center_hold='reward', early_leave_center_hold='hold_error', non_rhtouch='RH_touch', stop=None)
        self.FSM['reward'] = dict(end_reward = 'ITI', stop=None, non_rhtouch='RH_touch')
        self.FSM['hold_error'] = dict(end_hold_error='ITI', stop=None, non_rhtouch='RH_touch')
        self.FSM['timeout_error'] = dict(end_timeout_error='ITI', stop=None, non_rhtouch='RH_touch')
        self.FSM['idle_exit'] = dict(stop=None)

        ### Initialize the reward 
        try:
            self.reward_port = serial.Serial(port='COM4',
                baudrate=115200)
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

        # save parameters: 
        d = dict(animal_name=animal_name, 
            center_target_rad=self.center_target_rad,
            ITI_mean=self.ITI_mean, 
            ITI_std = self.ITI_std, 
            ch_timeout=self.ch_timeout, 
            cht=self.cht, 
            reward_time=self.reward_for_targtouch[1],
            timeout_error_timeout = self.timeout_error_timeout,
            hold_error_timeout = self.hold_error_timeout,
            target_distance = self.target_distance,
            start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M'),
            rew_delay = self.reward_delay_time,
            use_cap_sensor = self.use_cap_sensor,
            drag_ok = self.drag_ok,
            )

        print(self.reward_for_center)
        print(self.reward_for_targtouch)
        print(self.reward_for_anytouch)

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
            self.h5file = tables.open_file(self.filename + '_data.hdf', mode='w', title = 'NHP data')
            self.h5_table = self.h5file.create_table('/', 'task', Data, '')
            self.h5_table_row = self.h5_table.row
            self.h5_table_row_cnt = 0

                # Note in python 3 to open pkl files: 
                #with open('xxxx_params.pkl', 'rb') as f:
                #    data_params = pickle.load(f)


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

        if self.use_cap_sensor:
            self.serial_port_cap.close()
        
        if self.idle:
            self.state = 'idle_exit'
            self.trial_counter = -1

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
            self.generatorz_param = self.generatorz_param2

            self.nudge_text = 'Nudge 9oclock targ? '
            self.nudge_param = str(self.nudge_dist)
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

        # self.h5_table_row['target_pos'] = self.periph_target_position
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
        self.cam_trig_port.write('0'.encode())
        Window.clearcolor = (0., 0., 0., 1.)
        self.exit_target1.color = (.15, .15, .15, 1.)
        self.exit_target2.color = (.15, .15, .15, 1.)

        # Set ITI, CHT, THT
        self.ITI = np.random.random()*self.ITI_std + self.ITI_mean

        if type(self.cht_type) is str:
            cht_min, cht_max = self.cht_type.split('-')
            self.cht = ((float(cht_max) - float(cht_min)) * np.random.random()) + float(cht_min)

        if type(self.tht_type) is str:
            tht_min, tht_max = self.tht_type.split('-')
            self.tht = ((float(tht_max) - float(tht_min)) * np.random.random()) + float(tht_min)            
        
        self.center_target.color = (0., 0., 0., 0.)
        self.periph_target.color = (0., 0., 0., 0.)
        self.indicator_targ.color = (0., 0., 0., 0.)
        
    def end_ITI(self, **kwargs):
        return kwargs['ts'] > self.ITI

    def _start_vid_trig(self, **kwargs):
        if self.trial_counter == 0:
            time.sleep(1.)
        self.cam_trig_port.write('1'.encode())
        self.first_target_attempt = True

        if np.logical_and(self.use_cap_sensor, not self.rhtouch_sensor):
            self.periph_target.color = (1., 0., 0., 1.)
            self.center_target.color = (1., 0., 0., 1.)
            Window.clearcolor = (1., 0., 0., 1.)

            # Turn exit buttons redish:
            self.exit_target1.color = (.9, 0, 0, 1.)
            self.exit_target2.color = (.9, 0, 0, 1.)

    def end_vid_trig(self, **kwargs):
        return kwargs['ts'] > self.pre_start_vid_ts


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

    def _start_center(self, **kwargs):
        Window.clearcolor = (.5, .5, .5, 1.)
        self.center_target.color = (1., 0., 0., 1.)
        self.exit_target1.color = (.15, .15, .15, 1)
        self.exit_target2.color = (.15, .15, .15, 1)
        self.periph_target.color = (.5, .5, .5, 1.)
        self.indicator_targ.color = (.25, .25, .25, 1.)

    def _start_center_hold(self, **kwargs):
        self.center_target.color = (0.75, 0., 0., 1.)
        self.indicator_targ.color = (0.75, .75, .75, 1.)

    def _start_targ_hold(self, **kwargs):
        self.periph_target.color = (0., 1., 0., 1.)
        self.indicator_targ.color = (0.75, .75, .75, 1.)

    def _end_center_hold(self, **kwargs):
        self.center_target.color = (0., 0., 0., 1.)

    def _end_target_hold(self, **kwargs):
        self.periph_target.color = (0., 0., 0., 1.)

    def _start_touch_error(self, **kwargs):
        self.center_target.color = (0., 0., 0., 1.)
        self.periph_target.color = (0., 0., 0., 1.)
        self.repeat = True

    def _start_timeout_error(self, **kwargs):
        self.center_target.color = (0., 0., 0., 1.)
        self.periph_target.color = (0., 0., 0., 1.)
        #self.repeat = True

    def _start_hold_error(self, **kwargs):
        self.center_target.color = (0., 0., 0., 1.)
        self.periph_target.color = (0., 0., 0., 1.)
        self.repeat = True

    def _start_drag_error(self, **kwargs):
        self.center_target.color = (0., 0., 0., 1.)
        self.periph_target.color = (0., 0., 0., 1.)
        self.repeat = True

    def _start_target(self, **kwargs):
        Window.clearcolor = (0., 0., 0., 1.)
        self.center_target.color = (0., 0., 0., 0.)

        if self.repeat is False:
            self.periph_target_position = self.target_list[self.target_index, :]
            self.target_index += 1
            print(self.periph_target_position)
            print(self.target_index)

        self.periph_target.move(self.periph_target_position)
        self.periph_target.color = (1., 1., 0., 1.)
        self.repeat = False
        self.exit_target1.color = (.15, .15, .15, 1)
        self.exit_target2.color = (.15, .15, .15, 1)
        self.indicator_targ.color = (.25, .25, .25, 1.)
        if self.first_target_attempt:
            self.first_target_attempt_t0 = time.time();
            self.first_target_attempt = False

    def _start_reward(self, **kwargs):
        self.trial_counter += 1
        Window.clearcolor = (1., 1., 1., 1.)
        self.periph_target.color = (1., 1., 1., 1.)
        self.exit_target1.color = (1., 1., 1., 1.)
        self.exit_target2.color = (1., 1., 1., 1.)
        self.rew_cnt = 0
        self.cnts_in_rew = 0
        self.indicator_targ.color = (1., 1., 1., 1.)
        self.repeat = False

    def _while_reward(self, **kwargs):
        if self.rew_cnt == 1:
            self.run_big_rew()
            self.rew_cnt += 1

    def _start_rew_anytouch(self, **kwargs):
        #if self.small_rew_cnt == 1:
        if self.reward_for_anytouch[0]:
            self.run_small_rew()
        else:
            self.repeat = True
            #self.small_rew_cnt += 1

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

    def end_rewanytouch(self, **kwargs):
        if self.small_rew_cnt > 1:
            return True
        else:
            return False

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
        return kwargs['ts'] > self.ch_timeout

    def finish_center_hold(self, **kwargs):
        if self.cht <= kwargs['ts']:
            if self.reward_for_targtouch[0]:
                self.run_small_rew()
            return True
        else:
            return False

    def early_leave_center_hold(self, **kwargs):
        return not self.check_if_cursors_in_targ(self.center_target_position, self.center_target_rad)
        
    def center_drag_out(self, **kwargs):
        touch = self.touch
        self.touch = True
        stay_in = self.check_if_cursors_in_targ(self.center_target_position, self.center_target_rad)
        self.touch = touch
        return not stay_in

    def touch_target(self, **kwargs):
        if self.drag_ok:
            return self.check_if_cursors_in_targ(self.periph_target_position, self.periph_target_rad)
        else:
            return np.logical_and(self.check_if_cursors_in_targ(self.periph_target_position, self.periph_target_rad),
                self.check_if_started_in_targ(self.periph_target_position, self.periph_target_rad))

    def target_timeout(self, **kwargs):
        #return kwargs['ts'] > self.target_timeout_time
        if time.time() - self.first_target_attempt_t0 > self.target_timeout_time:
            self.repeat = False
            return True

    def finish_targ_hold(self, **kwargs):
        return self.tht <= kwargs['ts']

    def early_leave_target_hold(self, **kwargs):
        return not self.check_if_cursors_in_targ(self.periph_target_position, self.periph_target_rad)

    def targ_drag_out(self, **kwargs):
        touch = self.touch
        self.touch = True
        stay_in = self.check_if_cursors_in_targ(self.periph_target_position, self.periph_target_rad)
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

    def get_4targets(self, target_distance=4, nudge=0., gen_kwarg=None):
        return self.get_targets_co(target_distance=target_distance, nudge=0.)

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

    def get_targets_rand(self, target_distance=4):
        # Targets in CM: 
        angle = np.linspace(0, 2*np.pi, 1000)
        target_distance = np.linspace(target_distance/4., target_distance, 1000)

        ix_ang = np.random.permutation(1000)
        ix_dist = np.random.permutation(1000)

        x = np.cos(angle[ix_ang])*target_distance[ix_dist]
        y = np.sin(angle[ix_ang])*target_distance[ix_dist]
        return np.hstack((x[:, np.newaxis], y[:, np.newaxis]))

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
