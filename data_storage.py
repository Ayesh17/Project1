#imports
import numpy as np

def build_nparray(data):
	header = np.array(data[0])
	samples_arr = np.empty([len(data)-1,len(data[0])-1])
	for i in range(1,len(data)):
		for j in range(len(data[i])-1):
			samples_arr[i-1][j]=data[i][j]

	labels_arr= np.empty(len(data)-1)
	for i in range(len(labels_arr)):
		labels_arr[i] = int(data[i+1][len(data[0]) - 1])

	return samples_arr,labels_arr


def build_list(data):
	header = data[0]
	samples =[]
	for i in range (1,len(data)):
		sample = []
		for j in range(len(data[i])-1):
			sample.append(float(data[i][j]))
		samples.append(sample)
	labels = []
	for i in range (1,len(data)):
		labels.append(int(data[i][len(data[i])-1]))
	return samples,labels

def build_dict(data):
	header = data[0]

	Dict = {0: {header[0]: [], header[1]: [], header[2]: [], header[3]: [], header[4]: []},
			1: {header[0]: [], header[1]: [], header[2]: [], header[3]: [], header[4]: []},
			2: {header[0]: [], header[1]: [], header[2]: [], header[3]: [], header[4]: []},
			3: {header[0]: [], header[1]: [], header[2]: [], header[3]: [], header[4]: []},
			4: {header[0]: [], header[1]: [], header[2]: [], header[3]: [], header[4]: []},
			5: {header[0]: [], header[1]: [], header[2]: [], header[3]: [], header[4]: []},}

	for i in range(1,len(data)):
		sub_dict = {header[0]: [], header[1]: [], header[2]: [], header[3]: [], header[4]: []}
		for j in range(len(data[i])-1):
			sub_dict[header[j]] = float(data[i][j])
		Dict.update({i-1:sub_dict})

	label_dict = {}

	for i in range(len(data)-1):
		label_dict.update({ i : data[i+1][len(data[0])-1]})

	return Dict, label_dict
