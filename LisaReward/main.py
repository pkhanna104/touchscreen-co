from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.widget import Widget
import serial, time
from kivy.core.window import Window

class Rew_Buttons(Widget):
    def init(self):
        try:
            self.reward_port = serial.Serial(port='COM6',
                baudrate=115200)
            self.reward_port.close()
        except:
            pass    

    def big_reward(self):
        self.deliver_juice(.5)

    def small_reward(self):
        self.deliver_juice(.1)

    def close(self):
        App.get_running_app().stop()
        Window.close()

    def deliver_juice(self, secs):
        try:
            self.reward_port.open()
            rew_str = [ord(r) for r in 'inf 50 ml/min '+str(secs)+' sec\n']
            self.reward_port.write(rew_str)
            time.sleep(.5 + self.reward_delay_time)
            run_str = [ord(r) for r in 'run\n']
            self.reward_port.write(run_str)
            self.reward_port.close()
        except:
            print('in delivery: ', str(secs))


class MainScreen(Screen):
    pass

class AnotherScreen(Screen):
    pass

class ScreenManagement(ScreenManager):
    pass

presentation = Builder.load_file("main.kv")

class MainApp(App):
    def build(self):
        return presentation
        
MainApp().run()


