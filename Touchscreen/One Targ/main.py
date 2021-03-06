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

    ITI_mean = 1.
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
    touch_error_timeout = 0.
    timeout_error_timeout = 0.
    hold_error_timeout = 0.
    drag_error_timeout = 0.

    ntargets = 4.
    target_distance = 4.
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
            
    def init(self, animal_names_dict=None, rew_in=None, task_in=None,
        test=None, hold=None, targ_structure=None,
        autoquit=None, rew_var=None, targ_timeout = None, targ_pos=None):

        self.rew_cnt = 0
        self.small_rew_cnt = 0


        self.use_cap_sensor = False

        if self.use_cap_sensor:
            self.serial_port_cap = serial.Serial(port='COM5')

        self.rhtouch_sensor = 0.


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


        if rew_in['rew_anytouch']:
            self.reward_for_anytouch = [True, small_rew]
        else:
            self.reward_for_anytouch = [False, 0]

        if np.logical_or(rew_in['rew_targ'], rew_in['rew_center_pls_targ']):
            self.reward_for_targtouch = [True, big_rew]
        else:
            self.reward_for_targtouch = [False, 0]

        if rew_in['rew_center_pls_targ']:
            self.reward_for_center = [True, small_rew]
        else:
            self.reward_for_center = [False, 0]

        if rew_in['snd_only']:
            self.reward_for_targtouch = [True, 0.]
            self.skip_juice = True
        else:
            self.skip_juice = False

        target_rad_opts = [.5, .75, .82, .91, 1.0, 1.5]
        for i, val in enumerate(task_in['targ_rad']):
            if val:
                self.periph_target_rad = target_rad_opts[i]
                self.center_target_rad = target_rad_opts[i]

        for i, (nm, val) in enumerate(animal_names_dict.items()):
            if val:
                animal_name = nm

        self.use_center = False
        for i, (nm, val) in enumerate(targ_structure.items()):
            if val:
                generatorz = getattr(self, nm)
                self.generatorz_param2 = nm
                if 'co' in nm:
                    self.use_center = True

        holdz = [ .375, .5, .575, .6, '.4-.6']
        
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
            self.reward_for_targtouch)


        # white_screen_opts = [True, False]
        # for i, val in enumerate(white_screen['white_screen']):
        #     if val:
        self.use_white_screen = False

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

        # drag_ok = [True, False]
        # for i, val in enumerate(drag['drag']):
        #     if val:
        #         self.drag_ok = drag_ok[i]
        self.drag_ok = False;

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
        self.center_target.set_size(2*self.center_target_rad)
        self.center_target.move(np.array([0., 0.]))
        self.periph_target.set_size(2*self.periph_target_rad)

        self.exit_target1.set_size(2*self.exit_rad)
        self.exit_target2.set_size(2*self.exit_rad)
        self.indicator_targ.set_size(self.exit_rad)
        self.indicator_targ.move(self.indicator_pos)
        self.indicator_targ.color = (0., 0., 0., 1.)

        self.exit_target1.move(self.exit_pos)
        self.exit_pos2 = np.array([self.exit_pos[0], -1*self.exit_pos[1]])
        self.exit_target2.move(self.exit_pos2)
        self.exit_target1.color = (.15, .15, .15, 1)
        self.exit_target2.color = (.15, .15, .15, 1)

        self.target_list = generatorz(self.target_distance, self.nudge_dist, self.generator_kwarg)
        self.target_index = 0
        self.repeat = False
        self.center_target_position = np.array([0., 0.])
        self.periph_target_position = self.target_list[self.target_index, :]

        self.FSM = dict()
        self.FSM['ITI'] = dict(end_ITI='vid_trig', stop=None)
        self.FSM['vid_trig'] = dict(rhtouch='target', stop=None)
        
        if self.use_center:
            self.FSM['vid_trig'] = dict(end_vid_trig='center', stop=None)
            self.FSM['center'] = dict(touch_center='center_hold', center_timeout='timeout_error', non_rhtouch='RH_touch',stop=None)
            self.FSM['center_hold'] = dict(finish_center_hold='target', early_leave_center_hold='hold_error', non_rhtouch='RH_touch', stop=None)

        self.FSM['target'] = dict(touch_target = 'targ_hold', target_timeout='timeout_error', stop=None,
            anytouch='rew_anytouch', non_rhtouch='RH_touch')#,touch_not_target='touch_error')
        self.FSM['targ_hold'] = dict(finish_targ_hold='reward', early_leave_target_hold = 'hold_error',
         targ_drag_out = 'drag_error', stop=None, non_rhtouch='RH_touch')
        self.FSM['reward'] = dict(end_reward = 'ITI', stop=None, non_rhtouch='RH_touch')

        if self.use_center:
            return_ = 'center'
        else:
            return_ = 'target'

        self.FSM['touch_error'] = dict(end_touch_error=return_, stop=None, non_rhtouch='RH_touch')
        self.FSM['timeout_error'] = dict(end_timeout_error='ITI', stop=None, non_rhtouch='RH_touch')
        self.FSM['hold_error'] = dict(end_hold_error=return_, stop=None, non_rhtouch='RH_touch')
        self.FSM['drag_error'] = dict(end_drag_error=return_, stop=None, non_rhtouch='RH_touch')
        self.FSM['rew_anytouch'] = dict(end_rewanytouch='target', stop=None, non_rhtouch='RH_touch')
        self.FSM['idle_exit'] = dict(stop=None)

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
        d = dict(animal_name=animal_name, center_target_rad=self.center_target_rad,
            periph_target_rad=self.periph_target_rad, target_structure = generatorz.__name__, 
            target_list = self.target_list, 
            ITI_mean=self.ITI_mean, ITI_std = self.ITI_std, ch_timeout=self.ch_timeout, 
            cht=self.cht, reward_time_small=self.reward_for_center[1],
            reward_time_big=self.reward_for_targtouch[1],
            reward_for_anytouch=self.reward_for_anytouch[0],
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

        self.h5_table_row['target_pos'] = self.periph_target_position
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
        Window.clearcolor = (0., 0., 0., 1.)
        self.center_target.color = (1., 1., 0., 1.)
        self.exit_target1.color = (.15, .15, .15, 1)
        self.exit_target2.color = (.15, .15, .15, 1)
        self.periph_target.color = (0., 0., 0., 1.)
        self.indicator_targ.color = (.25, .25, .25, 1.)

    def _start_center_hold(self, **kwargs):
        self.center_target.color = (0., 1., 0., 1.)
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
            if self.reward_for_center[0]:
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
