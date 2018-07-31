''' Script to be continuously running on Rasperry Pi for NHP project: 
	- normally in off/idle state
	- when a button is pressed, script moves into active state 
		- open serial port connections
		- open file to write
		- start task

'''
import serial, time, struct

ntarg = 2
LED_on = [0b00000001, 0b00000010, 0b00000100]
LED_off = 0b00000000

file_directory = ''

class Task(object):
	def __init__(self):
		self.state = 'idle'
		self.timeout = 5.
		self.timeout_penalty = 5.

		# Similar to BMI3D -- keys are states, entries are dict with 
		# keys equal to next state, and entry equal to assessment fcn
		self.FSM = dict()
		self.FSM['idle'] = dict('init_task' = self.on_button_tog)
		self.FSM['init_task'] = dict('task' = self.done_init)
		self.FSM['task'] = dict('stop_task' = self.on_button_tog, 'stop_task' = self.end_of_targs)
		self.FSM['stop_task'] = dict('idle' = self.cleanup_done)

		self.arduino = serial.Serial('/dev/tty.usbmodem14441', baudrate=9600)
		self.run()

	def run(self):

		# Functions to attempt to change state:
		for i, (key, fcn) in enumerate(self.FSM.items()):
			outcome = fcn()

			if outcome:
				self.state = key
				break

		# What to do while in a state: 
		if self.state == 'init_task':
			self.init_task()

		elif self.state == 'stop_task':
			
			# Close reward port and file
			self.reward_port.close()
			self.file.close()

			#Turn off lights
			self.arduino.write('w'+struct.pack('<H', LED_off))

		elif self.state == 'task':

			if self.task_state == 'init':

				if self.prev_task_outcome == 'reward':
					# New target:
					self.target = self.target_order.pop()

				# Turn on LED: 
				self.arduino.write('w'+struct.pack('<H', LED_on[self.target]))

				# Start timer
				self.target_start = time.time()

				# Move to target state: 
				self.task_state = 'target'

			elif self.task_state == 'target':

				if time.time() - self.target_start < self.timeout:

					# Check if button has been pressed; 
					self.arduino.write('q')

					# Read the return
					buttons = self.arduino.read(ntarg+1+2)

					# Process the return -- on/off toggle
					if int(buttons[self.target]) == 1:
						self.task_state = 'reward'

					for but in range(ntarg):
						if but != self.target:
							if int(buttons[but]) == 1:
								self.task_state = 'wrong_button'

				else:
					self.task_state = 'timeout'

				if self.task_state != 'target':
					# Deal with the state changes: 
					t_end = time.strftime('%H%M%S')

					# Write down the a) trial outcome, b) time, c) target number
					self.file.write(self.task_state+','+t_end+','+str(self.target))
					
					# Turn off the LED: 
					self.arduino.write('w'+struct.pack('<H', LED_off))

					# Write soemthing to the file: 
					self.file.write()

					if self.task_state in ['timeout', 'wrong_button']:
						time.sleep(self.timeout_penalty)

					elif self.task_state == 'reward':
						self.deliver_reward(self.reward_port)
						
					self.task_state = 'init'

	def on_button_tog(self):
		# Check if there's anything available on the arduino
		# Query the arduino
		self.arduino.write('q')

		# Read the return
		buttons = self.arduino.read(ntarg+1+2)

		# Process the return -- on/off toggle
		if int(buttons[ntarg]) == 0:
			return True
		else:
			return False

	def init_task(self):
		self.reward_port = self.init_reward()
		self.target_order = self.return_target_order()
		self.init_file()

	def done_init(self):
		self.task_state = 'init'
		self.prev_task_outcome = 'reward'
		return True

	def return_target_order(self, ntargs=1000):

		# Blocks of 2*ntarg
		x = np.array(range(ntarg)+range(ntarg))
		nblks = ntargs/len(x)

		y = []
		for b in range(nblks):
			np.random.permutation(x)
			y.append(x)
		return list(np.hstack((x)))

	def cleanup_done(self):
		return True

	def end_of_targs(self):
		if len(self.target_order) == 1:
			return True
		else:
			return False

	''' File writting related fcns here '''

	def init_file(self):
		self.file = open(file_directory+time.strftime('%Y%m%d_%H%M'), 'w')


	''' Reward related functions here '''

	def init_reward(self):
		# Macbook pro name -- change for Rasperrry Pi
		port_name = '/dev/tty.usbserial-A8008WJh'
		port = serial.Serial(port=port_name, baudrate=115200)
		port.close()
		return port


	def deliver_reward(self, port):
		port.open()
		port.write('inf 25 ml/min .8 sec\n')
		time.sleep(.5)
		port.write('run\n')
		port.close()

