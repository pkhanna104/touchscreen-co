import serial, struct, time; 

pinch_rest = 'd'+struct.pack('<H', 0)
power = 'd'+struct.pack('<H', 45)
power_rest = 'd'+struct.pack('<H', 89)
tripod = 'd'+struct.pack('<H', 133)
tripod_rest = 'd'+struct.pack('<H', 178)
pinch = 'd'+struct.pack('<H', 222)

pinch_rest = 'd'+struct.pack('<H', 0)
power = 'd'+struct.pack('<H', 2)
power_rest = 'd'+struct.pack('<H', 4)
tripod = 'd'+struct.pack('<H', 6)
tripod_rest = 'd'+struct.pack('<H', 8)
pinch = 'd'+struct.pack('<H', 10)

for p in [tripod, tripod_rest]: 
	port.write(p)
	time.sleep(10)




port = serial.Serial('/dev/cu.usbmodem145301', baudrate=115200)



port = serial.Serial('/dev/cu.usbmodem144301', baudrate=115200)


t=0; 
pos = []
slot = [tripod, pinch, power, tripod_rest, pinch_rest, power_rest, pinch]
while t < 200*7: 
	if t % 200 == 0: 
		port.write(slot[t/200])
	port.flushInput()
	x = port.readline(); 
	x1 = x.split('     \t     ')
	while len(x1)!= 8: 
		x = port.readline(); 
		x1 = x.split('     \t     ')
	
	pos.append(int(x1[3]))
	t += 1; 

##### New acquisition ######
t=0; pos=[]
port.flushInput()
while t < 20000:
	if np.mod(t, 1000) == 0: 
		print(t)
	port.flushInput()
	x = port.readline();
	x1 = x.split('\t')
	while len(x1) != 3: 
		x = port.readline()
		x1 = x.split('\t') 

	if len(x1) == 3:
		if len(x1[0])>0:
			if len(x1[1])>0: 
				pos.append([int(x1[0]), int(x1[1])])
				t += 1; 

pos = np.vstack((pos))

### Rising edge events ###
encoder_A = pos[:, 0]
encoder_B = pos[:, 1]

### OLD_COUNT ###
count = np.zeros((len(encoder_A)))
count_OLD = np.zeros((len(encoder_A)))
for i in range(1, len(encoder_A)): 

	### Rising edge: 
	if encoder_B[i-1] == 0 and encoder_B[i] == 1: 
		if encoder_A[i] == 1: 
			count[i] = count[i-1] + 1; 
			count_OLD[i] = count_OLD[i-1] + 1; 
		else: 
			count_OLD[i] = count_OLD[i-1] - 1; 
	else: 
		count[i] = count[i-1]
		count_OLD[i] = count_OLD[i-1]

	if encoder_A[i-1] == 0 and encoder_A[i] == 1: 
		if encoder_B[i] == 1: 
			count[i] = count[i-1] - 1; 
	else: 
		count[i] = count[i-1]
		count_OLD[i] = count_OLD[i-1]




