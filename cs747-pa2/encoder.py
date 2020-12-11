import numpy as np 
#from matplotlib import pyplot as plt
import argparse
import random
#from pulp import *

parser = argparse.ArgumentParser(description='encode')
parser.add_argument('--grid', type=str, default=3,help='grid')
#parser.add_argument('--algorithm', type=str, default=3, help='algo')

args = parser.parse_args()
grid_file = args.grid

new_arr = []
with open(grid_file,'r') as f:
	for line in f.readlines():
		lst = line.split()
		new_arr.append(np.asarray(lst))

new_arr = np.asarray(new_arr)
new_arr = new_arr.astype(int)
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

mapping = mapping.astype(int)

states = cnt;
print("numStates",states)
print("numActions",4)
actions = 4

T = np.zeros((states,actions,states))
R = np.zeros((states,actions,states))
end = []
for i in range(1,n-1):
	for j in range(1,n-1):
		if new_arr[i,j] == 2:
			st = mapping[n*i+j]

		if new_arr[i,j] == 3:
			end.append(mapping[n*i + j])

		if new_arr[i,j]==0 or new_arr[i,j]==2:	
		# 	count = 0
		# 	if new_arr[i+1,j]!=1:
		# 		count += 1
		# 	if new_arr[i-1,j]!=1:
		# 		count += 1
		# 	if new_arr[i,j+1]!=1:
		# 		count += 1
		# 	if new_arr[i,j-1]!=1:
		# 		count += 1			
			
			#if count == 0:
			#	continue

			if new_arr[i+1,j]!=1:
				T[mapping[n*i+j],0,mapping[n*i+j+n]] = 1
				R[mapping[n*i+j],0,mapping[n*i+j+n]] = -1
			else:
				T[mapping[n*i+j],0,mapping[n*i+j+n]] = 1
				R[mapping[n*i+j],0,mapping[n*i+j+n]] = -10000000000

			if new_arr[i-1,j]!=1:
				T[mapping[n*i+j],1,mapping[n*i+j-n]] = 1
				R[mapping[n*i+j],1,mapping[n*i+j-n]] = -1
			else:
				T[mapping[n*i+j],1,mapping[n*i+j-n]] = 1
				R[mapping[n*i+j],1,mapping[n*i+j-n]] = -10000000000

			if new_arr[i,j+1]!=1:
				T[mapping[n*i+j],2,mapping[n*i+j+1]] = 1
				R[mapping[n*i+j],2,mapping[n*i+j+1]] = -1
			else:
				T[mapping[n*i+j],2,mapping[n*i+j+1]] = 1
				R[mapping[n*i+j],2,mapping[n*i+j+1]] = -10000000000

			if new_arr[i,j-1]!=1:
				T[mapping[n*i+j],3,mapping[n*i+j-1]] = 1
				R[mapping[n*i+j],3,mapping[n*i+j-1]] = -1
			else:
				T[mapping[n*i+j],3,mapping[n*i+j-1]] = 1
				R[mapping[n*i+j],3,mapping[n*i+j-1]] = -10000000000

		# elif new_arr[i,j] == 1:
		# 	count = 4
		# 	T[n*i+j,0,n*i+j+n] = 1
		# 	R[n*i+j,0,n*i+j+n] = -10000000000			

		# 	T[n*i+j,1,n*i+j-n] = 1
		# 	R[n*i+j,1,n*i+j-n] = -10000000000

		# 	T[n*i+j,2,n*i+j+1] = 1
		# 	R[n*i+j,2,n*i+j+1] = -10000000000

		# 	T[n*i+j,3,n*i+j-1] = 1
		# 	R[n*i+j,3,n*i+j-1] = -10000000000

		elif new_arr[i,j] == 3:
			count = 4
			T[mapping[n*i+j],0,mapping[n*i+j+n]] = 0
			R[mapping[n*i+j],0,mapping[n*i+j+n]] = -10000000000			

			T[mapping[n*i+j],1,mapping[n*i+j-n]] = 0
			R[mapping[n*i+j],1,mapping[n*i+j-n]] = -10000000000

			T[mapping[n*i+j],2,mapping[n*i+j+1]] = 0
			R[mapping[n*i+j],2,mapping[n*i+j+1]] = -10000000000

			T[mapping[n*i+j],3,mapping[n*i+j-1]] = 0
			R[mapping[n*i+j],3,mapping[n*i+j-1]] = -10000000000


end_list = [str(x) for x in end]
print("start",st)
print("end"," ".join(end_list))

for x in range(1,n-1):
	for y in range(1,n-1):
		i = x*n + y
		#x = i/n
		#y = i%n

		#if new_arr[x,y] == 3 or new_arr[x,y] == 1:
		#	continue;

		#if new_arr[x+1,y] != 1:
		if new_arr[x,y]!=1:
			print("transition",str(mapping[i]),str(0),str(mapping[i+n]),str(R[mapping[i],0,mapping[i+n]]),str(T[mapping[i],0,mapping[i+n]]))

			#if new_arr[x-1,y] != 1:
			print("transition",str(mapping[i]),str(1),str(mapping[i-n]),str(R[mapping[i],1,mapping[i-n]]),str(T[mapping[i],1,mapping[i-n]]))

			#if new_arr[x,y+1] != 1:
			print("transition",str(mapping[i]),str(2),str(mapping[i+1]),str(R[mapping[i],2,mapping[i+1]]),str(T[mapping[i],2,mapping[i+1]]))

			#if new_arr[x,y-1] != 1:
			print("transition",str(mapping[i]),str(3),str(mapping[i-1]),str(R[mapping[i],3,mapping[i-1]]),str(T[mapping[i],3,mapping[i-1]]))


print("mdptype episodic")
print("discount 1")