from kivy.app import App
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, ListProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.vector import Vector
from kivy.clock import Clock
from random import randint
from kivy.config import Config
import serial, time, pickle, datetime, winsound
# to install -- pyserial, pygame, 

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

    ITI_mean = 1.
    ITI_std = .2
    center_target_rad = 1.5
    periph_target_rad = 1.5

    exit_pos = np.array([7, 4])
    exit_rad = 1.
    exit_hold = 2 #seconds

    ch_timeout = 10. # ch timeout
    cht = .001 # center hold time

    target_timeout_time = 5000.
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
    target_distance = 5.
    touch = False

    center_target = ObjectProperty(None)
    periph_target = ObjectProperty(None)

    done_init = False
    prev_exit_ts = np.array([0,0])

    # Number of trials: 
    trial_counter = NumericProperty(0)


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
            
    def init(self, animal_names_dict=None, rew_in=None, task_in=None, rew_del=None,
        test=None, cap_on=None, hold=None, targ_structure=None,
        autoquit=None, drag=None):

        holdz = [.25, .5, .625, .75]
        for i, val in enumerate(hold['hold']):
            if val:
                self.cht = holdz[i]
                self.tht = holdz[i]

        cap = [1, 0]
        for i, val in enumerate(cap_on['cap']):
            if val:
                self.use_cap_sensor = cap[i]

        if self.use_cap_sensor:
            self.serial_port_cap = serial.Serial(port='COM3')

        self.rhtouch_sensor = 0.

        small_rew_opts = [.1, .3, .5]
        for i, val in enumerate(rew_in['small_rew']):
            if val:
                small_rew = small_rew_opts[i]

        big_rew_opts = [.3, .5, .7]
        for i, val in enumerate(rew_in['big_rew']):
            if val:
                big_rew = big_rew_opts[i]


        if np.logical_or(rew_in['rew_anytouch'], rew_in['rew_any_pls_targ']):
            self.reward_for_anytouch = [True, small_rew]
        else:
            self.reward_for_anytouch = [False, 0]

        if np.logical_or(rew_in['rew_targ'], rew_in['rew_any_pls_targ']):
            self.reward_for_targtouch = [True, big_rew]
        else:
            self.reward_for_targtouch = [False, 0]

        if rew_in['snd_only']:
            self.reward_for_targtouch = [True, 0.]
            self.skip_juice = True
        else:
            self.skip_juice = False

        target_rad_opts = [.5, 1.0, 1.5, 2.0, 3.0]
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
                if 'co' in nm:
                    self.use_center = True
        try:
            pygame.mixer.init()    
        except:
            pass

        reward_delay_opts = [0., .4, .8, 1.2]
        for i, val in enumerate(rew_del['rew_del']):
            if val:
                self.reward_delay_time = reward_delay_opts[i]

        test_vals = [True, False, False]
        in_cage_vals = [False, False, True]
        for i, val in enumerate(test['test']):
            if val:
                self.testing = test_vals[i]
                self.in_cage = in_cage_vals[i]
        
        autoquit_trls = [25, 50, 10**10]
        for i, val in enumerate(autoquit['autoquit']):
            if val: 
                self.max_trials = autoquit_trls[i]

        drag_ok = [True, False]
        for i, val in enumerate(drag['drag']):
            if val:
                self.drag_ok = drag_ok[i]

        self.state = 'ITI'
        self.state_start = time.time()
        self.ITI = np.random.random()*self.ITI_std + self.ITI_mean

        # Initialize targets: 
        self.center_target.set_size(2*self.center_target_rad)
        self.center_target.move(np.array([0., 0.]))
        self.periph_target.set_size(2*self.periph_target_rad)

        self.exit_target1.set_size(2*self.exit_rad)
        self.exit_target2.set_size(2*self.exit_rad)
        self.exit_target1.move(self.exit_pos)
        self.exit_pos2 = np.array([self.exit_pos[0], -1*self.exit_pos[1]])
        self.exit_target2.move(self.exit_pos2)
        self.exit_target1.color = (.15, .15, .15, 1)
        self.exit_target2.color = (.15, .15, .15, 1)

        self.target_list = generatorz()
        self.target_index = 0
        self.repeat = False
        self.center_target_position = np.array([0., 0.])
        self.periph_target_position = self.target_list[self.target_index, :]

        self.FSM = dict()
        self.FSM['ITI'] = dict(end_ITI='RH_touch', stop=None)
        self.FSM['RH_touch'] = dict(rhtouch='target', stop=None)
        
        if self.use_center:
            self.FSM['RH_touch'] = dict(rhtouch='center', stop=None)
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
        self.FSM['timeout_error'] = dict(end_timeout_error=return_, stop=None, non_rhtouch='RH_touch')
        self.FSM['hold_error'] = dict(end_hold_error=return_, stop=None, non_rhtouch='RH_touch')
        self.FSM['drag_error'] = dict(end_drag_error=return_, stop=None, non_rhtouch='RH_touch')
        self.FSM['rew_anytouch'] = dict(end_rewanytouch='target', stop=None, non_rhtouch='RH_touch')
        self.FSM['idle_exit'] = dict(stop=None)

        try:
            self.reward_port = serial.Serial(port='COM6',
                baudrate=115200)
            self.reward_port.close()
        except:
            pass

        # save parameters: 
        d = dict(animal_name=animal_name, center_target_rad=self.center_target_rad,
            periph_target_rad=self.periph_target_rad, target_structure = generatorz.__name__, 
            target_list = self.target_list, 
            ITI_mean=self.ITI_mean, ITI_std = self.ITI_std, ch_timeout=self.ch_timeout, 
            cht=self.cht, reward_time_small=self.reward_for_anytouch[1],
            reward_time_big=self.reward_for_targtouch[1],
            reward_for_anytouch=self.reward_for_anytouch[0],
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

        try:
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
                self.filename = p+ animal_name+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M')
                if self.in_cage:
                    self.filename = self.filename+'_cage'

                pickle.dump(d, open(self.filename+'_params.pkl', 'wb'))
                self.h5file = tables.open_file(self.filename + '_data.hdf', mode='w', title = 'NHP data')
                self.h5_table = self.h5file.create_table('/', 'task', Data, '')
                self.h5_table_row = self.h5_table.row

                # Note in python 3 to open pkl files: 
                #with open('xxxx_params.pkl', 'rb') as f:
                #    data_params = pickle.load(f)

        except:
            pass

    def close_app(self):
        # Save Data Eventually
        if self.use_cap_sensor:
            self.serial_port_cap.close()
        if self.idle:
            self.state = 'idle_exit'
            self.trial_counter = -1
        else:
            App.get_running_app().stop()
            Window.close()

    def update(self, ts):
        self.state_length = time.time() - self.state_start
        
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
        self.h5_table_row['time'] = time.time()
        self.h5_table_row['cap_touch'] = self.rhtouch_sensor
        self.h5_table_row.append()

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
        Window.clearcolor = (0., 0., 0., 1.)
        self.exit_target1.color = (.15, .15, .15, 1.)
        self.exit_target2.color = (.15, .15, .15, 1.)

        self.ITI = np.random.random()*self.ITI_std + self.ITI_mean
        self.center_target.color = (0., 0., 0., 0.)
        self.periph_target.color = (0., 0., 0., 0.)

    def end_ITI(self, **kwargs):
        return kwargs['ts'] > self.ITI

    def _start_RH_touch(self, **kwargs):
        if np.logical_and(self.use_cap_sensor, not self.rhtouch_sensor):
            self.periph_target.color = (1., 0., 0., 1.)
            self.center_target.color = (1., 0., 0., 1.)
            Window.clearcolor = (1., 0., 0., 1.)

            # Turn exit buttons redish:
            self.exit_target1.color = (.9, 0, 0, 1.)
            self.exit_target2.color = (.9, 0, 0, 1.)


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
        if x:
            self.repeat = True
        return x

    def _start_center(self, **kwargs):
        Window.clearcolor = (0., 0., 0., 1.)
        self.center_target.color = (1., 1., 0., 1.)
        self.exit_target1.color = (.15, .15, .15, 1)
        self.exit_target2.color = (.15, .15, .15, 1)
        self.periph_target.color = (0., 0., 0., 1.)

    def _start_center_hold(self, **kwargs):
        self.center_target.color = (0., 1., 0., 1.)

    def _start_targ_hold(self, **kwargs):
        self.periph_target.color = (0., 1., 0., 1.)

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
        self.repeat = True

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

        self.periph_target.move(self.periph_target_position)
        self.periph_target.color = (1., 1., 0., 1.)
        self.repeat = False
        self.exit_target1.color = (.15, .15, .15, 1)
        self.exit_target2.color = (.15, .15, .15, 1)

    def _start_reward(self, **kwargs):
        self.trial_counter += 1
        Window.clearcolor = (1., 1., 1., 1.)
        self.periph_target.color = (1., 1., 1., 1.)
        self.exit_target1.color = (1., 1., 1., 1.)
        self.exit_target2.color = (1., 1., 1., 1.)
        
        try:
            if self.reward_for_targtouch[0]:
                winsound.PlaySound('beep1.wav', winsound.SND_ASYNC)

                if not self.skip_juice:
                    self.reward_port.open()
                    rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.reward_for_targtouch[1])+' sec\n']
                    self.reward_port.write(rew_str)
                    time.sleep(.5 + self.reward_delay_time)
                    run_str = [ord(r) for r in 'run\n']
                    self.reward_port.write(run_str)
                    self.reward_port.close()
        except:
            pass
        
    def _start_rew_anytouch(self, **kwargs):
        try:
            if self.reward_for_anytouch[0]:
                winsound.PlaySound('beep1.wav', winsound.SND_ASYNC)
                self.reward_port.open()
                rew_str = [ord(r) for r in 'inf 50 ml/min '+str(self.reward_for_anytouch[1])+' sec\n']
                self.reward_port.write(rew_str)
                time.sleep(.5)
                run_str = [ord(r) for r in 'run\n']
                self.reward_port.write(run_str)
                self.reward_port.close()
        except:
            pass

        self.repeat = True

    def end_reward(self, **kwargs):
        if len(self.cursor_ids)== 0:
            return True

    def end_rewanytouch(self, **kwargs):
        return True

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
        return self.cht <= kwargs['ts']

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
        return kwargs['ts'] > self.target_timeout_time

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

    def get_4targets(self):
        return self.get_targets_co()

    def get_targets_co(self, ntargets=4, target_distance=4):
        # Targets in CM: 
        angle = np.linspace(0, 2*np.pi, ntargets+1)[:-1]
        x = np.cos(angle)*target_distance
        y = np.sin(angle)*target_distance
        tmp = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))

        tgs = []
        for blks in range(100):
            ix = np.random.permutation(tmp.shape[0])
            tgs.append(tmp[ix, :])
        return np.vstack((tgs))

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
