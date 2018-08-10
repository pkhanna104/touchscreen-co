import numpy as np
import tables
import matplotlib.pyplot as plt

data_path = 'C:/Users/scientist/Documents/Preeya/data/'
day_to_analyze = '0802'
monk = 'haun'

import glob
files = glob.glob(data_path+monk+'*'+day_to_analyze+'*'+'_data.hdf')


def run_analysis():
	for fl in files: 
		hdf = tables.open_file(fl)
		state = [str(st) for st in hdf.root.task[:]['state']]

		ix = np.nonzero(np.array(state)=="b'reward'")[0]
		ix2 = []
		for i, j in enumerate(ix):
			if j-ix[i-1] > 1:
				ix2.append(j)
		print(fl+'........')
		print('number of reward: '+ str(len(ix2)))
