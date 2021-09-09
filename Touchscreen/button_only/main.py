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
import serial, time, pickle, datetime
from numpy import binary_repr
import struct
from sys import platform


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
    
    # SET THE POSITION OF THE EXIT BUTTONS AND THE PHOTODIODE INDICATOR LIGHT
    # positions are in CM measured from the center of the screen
    # if platform == 'darwin':
    
    exit_pos_x = (fixed_window_size_cm[0]/2)-1.5
    exit_pos_y = (fixed_window_size_cm[1]/2)-1.5
                 
    exit_pos = np.array([exit_pos_x, exit_pos_y])
    ind_pos_x = (fixed_window_size_cm[0]/2)-0.5
    ind_pos_y = (fixed_window_size_cm[1]/2)-0.5
                 
    indicator_pos = np.array([ind_pos_x, ind_pos_y])
    # elif platform == 'win32':
    #     exit_pos = np.array([7, 4])
    #     indicator_pos = np.array([8, 5])
    exit_rad = 1.
    exit_hold = 2 #seconds
    
    # SET THE HOLD AND TIMEOUT TIMES
    cursor = {}
    cursor_start = {}
    cursor_ids = []

    touch = False

    done_init = False
    prev_exit_ts = np.array([0,0])

    t0 = time.time()
    
    trial_counter = NumericProperty(0)
    big_rew_text = StringProperty('')
    big_rew_time_param = StringProperty('')
    
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
            
    def init(self, animal_names_dict=None, iti_in=None, rew_in=None, 
        hold=None, autoquit=None, rew_var=None):
        
        self.rew_cnt = 0
        
        # JUICE REWARD SETTINGS
        button_rew_opts = [.1, .3, .5, .7]
        for i, val in enumerate(rew_in['button_rew']):
            if val:
                button_rew = button_rew_opts[i]
        self.button_rew = [True, button_rew]
        
        
        # ANIMAL NAME
        for i, (nm, val) in enumerate(animal_names_dict.items()):
            if val:
                animal_name = nm
            
        
        # ITI DURATION
        iti_opts = [01.0, 1.5, 2.0, 2.5, 3.0]
        for i, val in enumerate(iti_in['iti']):
            if val:
                self.ITI_mean = iti_opts[i]

        # BUTTON HOLD TIMES
        holdz = [0.0, 0.1, 0.2, 0.3, 0.4, .5, .6, '.4-.6']
        self.button_hold_time_type = None
        for i, val in enumerate(hold['button_hold']):
            if val:
                if type(holdz[i]) is str:
                    mx, mn = holdz[i].split('-')
                    self.button_hold_time_type = holdz[i]
                    self.button_hold_time =  (float(mn)+float(mx))*.5
                else:
                    self.button_hold_time = holdz[i]
        
        try:
            pygame.mixer.init()    
        except:
            pass

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
            self.button_rew)


        self.use_white_screen = False

        self.testing = False

        autoquit_trls = [10, 25, 50, 100, 10**10]
        for i, val in enumerate(autoquit['autoquit']):
            if val: 
                self.max_trials = autoquit_trls[i]

        # Preload sounds: 
        self.reward1 = SoundLoader.load('C.wav')

        self.state = 'ITI'
        self.state_start = time.time()
        self.ITI = self.ITI_std + self.ITI_mean

        # Initialize targets: 
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

        self.repeat = False

        self.FSM = dict()
        self.FSM['ITI'] = dict(end_ITI='vid_trig', stop=None)
        self.FSM['vid_trig'] = dict(end_vid_trig='button', stop=None)
        self.FSM['button'] = dict(button_held='reward', stop=None)
        self.FSM['reward'] = dict(end_reward = 'ITI', stop=None)
        self.FSM['idle_exit'] = dict(stop=None)
        
        # OPEN PORTS
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
        
        # External button
        try:
            self.is_button_ard = True
            if platform == 'darwin':
                self.button_ard = serial.Serial(port='/dev/cu.usbmodem1421301', baudrate=9600)
            else:
                self.button_ard = serial.Serial(port='COM3', baudrate=9600)
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
            self.fsr_baseline = 10+1.5*np.max(baseline_data, axis=0)
        else: 
            self.fsr_baseline = np.array([200, 200])

        
        # save parameters: 
        d = dict(animal_name=animal_name,
            ITI_mean=self.ITI_mean, ITI_std = self.ITI_std,
            start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M'),
            testing=self.testing,
            rew_delay = self.reward_delay_time,
            fsr_baseline = self.fsr_baseline
            )

        if self.testing or platform == 'darwin':
            pass

        else:
            import os
            path = os.getcwd()
            path = path.split('\\')
            path_data = [p for p in path if np.logical_and('Touch' not in p, 'Targ' not in p)]
            path_root = ''
            for ip in path_data:
                path_root += ip+'/'
            p = path_root + 'data/'
            print('Auto path : %s'%p)
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
        

            pickle.dump(d, open(self.filename+'_params.pkl', 'wb'))
            self.h5file = tables.open_file(self.filename + '_data.hdf', mode='w', title = 'NHP data')
            self.h5_table = self.h5file.create_table('/', 'task', Data, '')
            self.h5_table_row = self.h5_table.row
            self.h5_table_row_cnt = 0

            # Note in python 3 to open pkl files: 
            #with open('xxxx_params.pkl', 'rb') as f:
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

    def close_app(self):
        # Save Data Eventually
         #Stop the video: 
        try:
            self.cam_trig_port.write('0'.encode())
        except:
            pass
        
        if self.idle:
            self.state = 'idle_exit'
            self.trial_counter = -1

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
             
        if self.testing or platform == 'darwin':
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
            
            self.exit_target1.color = (.15, .15, .15, 1)
            self.exit_target2.color = (.15, .15, .15, 1)
            
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
        Window.clearcolor = (0., 0., 1., 1.)
        self.exit_target1.color = (0., 0., 1., 1.)
        self.exit_target2.color = (0., 0., 1., 1.)

        # Set ITI, CHT, THT
        self.ITI = np.random.random()*self.ITI_std + self.ITI_mean
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

    def end_vid_trig(self, **kwargs):
        return kwargs['ts'] > self.pre_start_vid_ts
    
    def _start_button(self, **kwargs):
        Window.clearcolor = (0., 0., 0., 1.)
        sound = SoundLoader.load('E.wav')
        sound.play()
        
        self.exit_target1.color = (.15, .15, .15, 1)
        self.exit_target2.color = (.15, .15, .15, 1)
        self.indicator_targ.color = (.25, .25, .25, 1.)
        self.button_pressed_prev = False
        
    def button_held(self, **kwargs):
        button_pressed_prev = self.button_pressed_prev
        self.button_pressed_prev = self.button_pressed
        if self.button_pressed:
            if button_pressed_prev:
                if time.time() - self.t_button_hold_start > self.button_hold_time:
                    # if the button has been held down long enough
                    return True
                else:
                    return False
            else:
                # this is the first cycle that the button has been pressed for
                self.t_button_hold_start = time.time()
                return False
        else:
            return False

    def _start_reward(self, **kwargs):
        self.trial_counter += 1
        Window.clearcolor = (1., 1., 1., 1.)
        self.exit_target1.color = (1., 1., 1., 1.)
        self.exit_target2.color = (1., 1., 1., 1.)
        self.rew_cnt = 0
        self.cnts_in_rew = 0
        self.indicator_targ.color = (1., 1., 1., 1.)
        self.repeat = False

    def _while_reward(self, **kwargs):
        if self.rew_cnt == 1:
            self.run_button_rew()
            self.rew_cnt += 1
        
    def run_button_rew(self, **kwargs):
        try:
            #winsound.PlaySound('beep1.wav', winsound.SND_ASYNC)
            sound = SoundLoader.load('C.wav')
            sound.play()

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
