import serial
import time
port = serial.Serial(port='COM5')

x = []
t0 = time.time()
while time.time() - t0 < 60.:
    ser = port.flushInput()
    _ =port.readline()
    port_read = port.readline()
    port_splits = port_read.decode('ascii').split('/t')
    #print(port_splits)
    #print(self.state)
    if len(port_splits) != 4:
        ser = port.flushInput()
        _ = port.readline()
        port_read = port.readline()
        port_splits = port_read.decode('ascii').split('/t')            
    force = int(port_splits[0])
    x.append([force, time.time()-t0])


### Plot calibration curves: 
#VD, R = 100 kOhm
g = [31, 10, 11, 16, 20, 25, 39, 41, 55, 56, 63, 46, 24, 14, 7, 5, 0, 17, 22, 36, 47, 36, 
    54, 52, 64, 66, 112, 131, 34, 0, 173, 97, 156, 46, 230, 355, 392, 358, 231, 257, 294,  ]
ard = [18, 3, 4,   7, 17, 12, 22, 30, 33, 38, 44, 37, 16, 7,  10, 0, 0, 7, 15,  26, 35, 22, 
    37, 37, 43, 42,  79,  91, 34, 0, 105, 59, 101, 26, 185, 200, 195, 147, 155, 171, 141, ]

g_680 =   [0, 0, 7, 18, 18, 25, 38, 41, 51, 53, 46, 51,   39, 43 , 33, 41 , 59, 69, 81, 88, 230, 250, 113, 185, 204, 184, 148, 159, 3, 0, 3, 12, 15, 19, 22, 29, 32, 32,    24, 18, 12,10, 2,0 ]
ard_680 = [3, 0, 19, 79,81,152,163, 167,177,227,230,214, 182, 178, 165,196,209,260,280,285, 460, 449, 287, 416, 430, 423, 393, 401, 22,12,33,42, 46, 85, 100, 99, 134, 140, 77, 73, 51,81, 25, 18]

g_340 =   [2, 0, 7, 11, 19, 21, 32, 40, 29, 36, 43, 53,  54, 69, 70, 81, 59,  66, 102, 210, 110, 125, 131, 224, 104, 200, 63 , 54, 51 , 48, 43,  32, 5, 8, 14, 18,32, 35, 248, 270, 281,301,309,311, 330, 325, 354, 320,]
ard_340 = [3, 4, 28,28, 52, 46, 57, 89, 73, 75, 119,119,120,138,137,155, 121,124, 182, 288, 213, 217, 252, 312, 185, 310, 135, 120,109, 104, 100,90,12,33, 29, 48,78, 100,328, 361, 372,387,387,392,395,392, 432,406,]


# OpAmp, R_feedback = 2.1 MOhm 
g_opamp_2 =     [0,  43, 72, 144, 168, 439, 206, 235, 427, 185, 166, 270, 241, 199, 249, 150, 321, 358, 388, 391, 417, 144, 188 , 160, 175, 187, 75,  92,   89,  140,  150, 157, 110, 83, 75, 241,  ]
ard_opamp_2 = [68, 72, 84, 126, 140, 335, 188, 197, 304, 197, 167, 245, 228, 217, 235, 184, 270, 297, 313, 319, 339, 135, 167, 157, 177, 175, 124, 120, 107, 160, 157, 158, 145, 135, 125, 209]

# OpAmp, R_feedback = 3.1 MOhm
g_opamp_3 = [0, 62, 0,   60, 0,   61, 100, 103, 106, 116, 126, 132, 144, 151, 165, 181, 137, 136,   66,   81,   94,   99, 112, 126, 200, 206, 226, 240, 93, 86,   26, 30, 35, 46, 51, 59, 44, 39, 47, 50, 68, 71, 85, 75, 10, 12, 14, 16,0]
ard_opamp_3 = [64, 79, 64, 80, 68, 80, 119, 132, 130, 141, 150, 154, 180, 193, 182, 209, 165, 170, 109, 133, 132, 147, 151, 160, 230, 244, 237,
254, 140, 130, 71, 65, 75, 78, 86, 99, 78, 72, 84, 91, 100, 110, 134, 129, 67,67,68,68 ,65]

f, ax  =plt.subplots()
plt.plot(g, ard, '.', label = 'voltage divider, 100 kOhm')
plt.plot(g_680, ard_680, '.', label = 'voltage divider, 680 kOhm')
plt.plot(g_340, ard_340, '.', label= 'voltage divider, 340 kOhm')
plt.plot(g_opamp_2, ard_opamp_2, '.', label='opamp rfb = 2.1MOhm')
plt.plot(g_opamp_3, ard_opamp_3, '.', label='opamp rfb = 3.1MOhm')
plt.legend()
plt.xlabel('Grams')
plt.ylabel('Arduino Units ([0-1023] --> [0-5V]')
plt.show()