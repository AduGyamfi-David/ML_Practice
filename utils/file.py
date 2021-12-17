import csv
import numpy as np

def cleanData(instr):
	return ''.join([c for c in instr if c.isdigit()])

def openCSV(filename, x=[], y=[]):
	with open('data\\' + filename + '.csv', mode='r') as data_file:
		line = csv.reader(data_file, delimiter="\n")
		count = 0
		for row in line:
			if not (count == 0):
				data = row[0].split(",")
				x.append(int(cleanData(data[6])))
				y.append(int(cleanData(data[7])))
			else:
				count += 1
	return (np.array(x), np.array(y))