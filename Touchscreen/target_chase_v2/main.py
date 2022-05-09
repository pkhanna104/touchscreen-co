__version__ = '1.0'

import kivy
kivy.require('1.0.6')

from kivy.uix.widget import Widget
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.graphics import Color, Rectangle, Ellipse, Point, GraphicException
from random import random
from math import sqrt
from kivy.config import Config
from sys import platform
import os
import numpy as np


# Config.set('graphics', 'resizable', False)

# if platform == 'darwin': # we are on a Mac
#     # This probably means that we are testing on a personal laptop
    
#     # settings for MBP 16" 2021
#     fixed_window_size = (3072, 1920) # we get this automatically now but here it is anyway
#     fixed_window_size_cm = (34.5, 21.5) # this is the important part
#     pix_per_cm = 104. # we get this automatically now but here it is anyway
# elif platform == 'win32':
#     # see if there is an external monitor plugged in
#     from screeninfo import get_monitors
#     mon = get_monitors()
# #    if len(get_monitors()) > 1 or get_monitors()[0].height == 1080:
# #        # must be an external monitor plugged in
# #        # assume that it is the ViewSonic TD2230
# #        fixed_window_size = (1920, 1080) # we get this automatically now but here it is anyway
# #        fixed_window_size_cm = (47.6, 26.8) # this is the important part
# #        pix_per_cm = 40. # we get this automatically now but here it is anyway
# #    else:
#         # must just be the Surface Pro
#         # These are surface pro settings
#     fixed_window_size = (2160, 1440) # we get this automatically now but here it is anyway
#     fixed_window_size_cm = (47.6, 26.8)
# #        fixed_window_size_cm = (22.8, 15.2) # this is the important part
#     pix_per_cm = 95. # we get this automatically now but here it is anyway
#     import winsound

# Config.set('graphics', 'width', str(fixed_window_size[0]))
# Config.set('graphics', 'height', str(fixed_window_size[1]))

def get_targpos_from_str(pos_str, center_position, max_y_from_center, nudge_x):
    if pos_str == 'random': # set for now, will get overriden later
        targ_x = center_position[0]
        targ_y = center_position[1]
    elif pos_str == 'center':
        targ_x = center_position[0]+nudge_x
        targ_y = center_position[1]
    elif pos_str == 'upper_middle':
        targ_x = center_position[0]+nudge_x
        targ_y = center_position[1] + max_y_from_center
    elif pos_str == 'lower_middle':
        targ_x = center_position[0]+nudge_x
        targ_y = center_position[1] - max_y_from_center
    elif pos_str == 'upper_right':
        targ_x = max_y_from_center+nudge_x
        targ_y = center_position[1] + max_y_from_center
    elif pos_str == 'middle_right':
        targ_x = max_y_from_center+nudge_x
        targ_y = center_position[1]
    elif pos_str == 'lower_right':
        targ_x = max_y_from_center+nudge_x
        targ_y = center_position[1] - max_y_from_center
    elif pos_str == 'lower_left':
        targ_x = -max_y_from_center+nudge_x
        targ_y = center_position[1] - max_y_from_center
    elif pos_str == 'middle_left':
        targ_x = -max_y_from_center+nudge_x
        targ_y = center_position[1]
    elif pos_str == 'upper_left':
        targ_x = -max_y_from_center+nudge_x
        targ_y = center_position[1] + max_y_from_center
        
    return np.array([targ_x, targ_y])



class Touchtracer(Widget):
    target_rad = 300
    center_position = np.array([0., 0.])
    # max_y_from_center = fixed_window_size_cm[1]/2-target_rad
    
    target1_pos_str = 'middle_right'
    target2_pos_str = 'upper_middle'
    target3_pos_str = 'lower_left'
    target4_pos_str = 'center'
    target5_pos_str = 'upper_left'
    
    target1_position = (800, 450) #get_targpos_from_str(target1_pos_str, center_position, max_y_from_center, 0)
    target2_position = (450, 800) #get_targpos_from_str(target2_pos_str, center_position, max_y_from_center, 0)
    target3_position = (100, 100) #get_targpos_from_str(target3_pos_str, center_position, max_y_from_center, 0)
    target4_position = (450, 450) #get_targpos_from_str(target4_pos_str, center_position, max_y_from_center, 0)
    target5_position = (100, 800) #get_targpos_from_str(target5_pos_str, center_position, max_y_from_center, 0)
    
    target_index = 1
    target1_drawn = False
    cursor = [1, 1]
    cursor_ids = [1]
    active_targpos = []
    
    def check_if_cursors_in_targ(self, targ_center, targ_rad):
        inTarg = False
        for id_ in self.cursor_ids:
            if np.linalg.norm(np.array(self.cursor[id_]) - targ_center) < targ_rad:
                inTarg = True

        return inTarg
    
    def on_touch_down(self, touch):
        # import pdb; pdb.set_trace()
        win = self.get_parent_window()
        ud = touch.ud
        
        # Add new touch to ids: 
        self.cursor_ids.append(touch.uid)

        # Add cursor
        curs = np.array([touch.x, touch.y])
        self.cursor.append(curs.copy())
        
        ud['group'] = g = str(touch.uid)

        with self.canvas:
            Color(1., 1., 0)
            ud['lines'] = [
                Ellipse(pos=(touch.x, touch.y), size=(50, 50), group=g)]
        
        if not self.target1_drawn:
            with self.canvas:
                ud['targ'] = [
                    Ellipse(pos=(self.target1_position), size=(self.target_rad, self.target_rad), group='targ1')]
            self.active_targpos = self.target1_position
            self.target1_drawn = True
        
        if self.check_if_cursors_in_targ(self.active_targpos, self.target_rad):
            self.target_index += 1
            
            if self.target_index == 1:
                self.active_targpos = self.target1_position
            elif self.target_index == 2:
                self.active_targpos = self.target2_position
            elif self.target_index == 3:
                self.active_targpos = self.target3_position
            elif self.target_index == 4:
                self.active_targpos = self.target4_position
            elif self.target_index == 5:
                self.active_targpos = self.target5_position
        
            with self.canvas:
                ud['targ'] = [
                    Ellipse(pos=(self.active_targpos), size=(self.target_rad, self.target_rad), group='targ1')]


        touch.grab(self)
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is not self:
            return
        ud = touch.ud
        curs = np.array([touch.x, touch.y])
        self.cursor[touch.uid] =  curs.copy()
        ud['lines'][0].pos = touch.x, touch.y
        ud['targ'][0].pos = self.active_targpos

        import time
        t = int(time.time())
        if t not in ud:
            ud[t] = 1
        else:
            ud[t] += 1

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return
        touch.ungrab(self)
        ud = touch.ud
        self.canvas.remove_group(ud['group'])
        
    

class TouchtracerApp(App):
    title = 'Touchtracer'
    icon = 'icon.png'

    def build(self, **kwargs):
        
        return Touchtracer()

    def on_pause(self):
        return True


# def cm2pix(pos_cm, fixed_window_size_cm=fixed_window_size_cm):
#     # pix_per_cm = Window.width/fixed_window_size_cm[0]
    
#     # Convert from CM to pixels: 
#     pix_pos = pix_per_cm*pos_cm

#     if type(pix_pos) is np.ndarray:
#         # Translate to coordinate system w/ 0, 0 at bottom left
#         pix_pos[0] = pix_pos[0] + (fixed_window_size[0]/2.)
#         pix_pos[1] = pix_pos[1] + (fixed_window_size[1]/2.)
#         # pix_pos[0] = pix_pos[0] + (fixed_window_size[0]/2.)
#         # pix_pos[1] = pix_pos[1] + (fixed_window_size[1]/2.)

#     return pix_pos

# def pix2cm(pos_pix, fixed_window_size_cm=fixed_window_size_cm):
#     pix_per_cm = Window.width/fixed_window_size_cm[0]
    
#     # First shift coordinate system: 
#     pos_pix[0] = pos_pix[0] - (fixed_window_size[0]/2.)
#     pos_pix[1] = pos_pix[1] - (fixed_window_size[1]/2.)

#     pos_cm = pos_pix*(1./pix_per_cm)
#     return pos_cm

if __name__ == '__main__':
    TouchtracerApp().run()