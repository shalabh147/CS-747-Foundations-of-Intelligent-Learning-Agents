import numpy as np 
#from matplotlib import pyplot as plt
import argparse
import random
#from pulp import *

parser = argparse.ArgumentParser(description='encode')
parser.add_argument('--grid', type=str, default=3,help='grid')
parser.add_argument('--value_policy', type=str, default=3, help='val')

args = parser.parse_args()
grid_file = args.grid
val_file = args.value_policy

new_arr = []
with open(grid_file,'r') as f:
	for line in f.readlines():
		lst = line.split()
		new_arr.append(np.asarray(lst))

new_arr = np.asarray(new_arr)
new_arr = new_arr.astype(int)
#print(new_arr)
#print(new_arr.shape)

n,n = new_arr.shape

n,n = new_arr.shape
mapping = np.zeros(n*n)
#print(new_arr)
#print(new_arr.shape)
cnt = 0
for i in range(0,n):
	for j in range(0,n):
		if new_arr[i,j] != 1:
			mapping[n*i+j] = cnt
			cnt += 1

states = cnt;
mapping = mapping.astype(int)
inv_mapping = np.zeros(states)

for i in range(0,n):
	for j in range(0,n):
		if new_arr[i,j] != 1:
			inv_mapping[mapping[n*i+j]] = n*i+j
			
inv_mapping = inv_mapping.astype(int)
#print("numStates",states)
#print("numActions",4)
actions = 4

end = []
for i in range(0,n):
	for j in range(0,n):
		if new_arr[i,j] == 2:
			st = mapping[i*n+j]

		if new_arr[i,j] == 3:
			end.append(mapping[i*n+j])

ac = np.zeros(n*n)
with open(val_file,'r') as f:
	cnt = 0
	for line in f.readlines():
		lst = line.split()
		ac[cnt] = int(lst[1])
		cnt += 1 

state = st
ans = []
while state not in end:
	if ac[state] == 0:
		ans.append("S")
		state = mapping[inv_mapping[state]+n]
	elif ac[state] == 1:
		ans.append("N")
		state =mapping[inv_mapping[state]-n]
	elif ac[state] == 2:
		ans.append("E")
		state =mapping[inv_mapping[state]+1]
	elif ac[state] == 3:
		ans.append("W")
		state = mapping[inv_mapping[state]-1]



lans = " ".join(ans)
print(lans)

