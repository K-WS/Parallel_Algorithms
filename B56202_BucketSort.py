from nose.tools import assert_equal
from mpi4py import MPI
#import numpy as np
import random
import string
import copy
import time
import math


#-------Assisting Functions------#
#--------------------------------#

# Function to write a text file that will later be read in a separate location
# To create graphs
def writer(arr, path):
    with open(path, 'w') as f:
        for item in arr:
            f.write("%s\n" % item)

# checker to confirm if sorted, borrowed from https://stackoverflow.com/questions/3755136/pythonic-way-to-check-if-a-list-is-sorted-or-not
def isSorted(l):
    return all(l[i] <= l[i+1] for i in range(len(l)-1))


#Method that generates a list of random numbers based on list size
def generate(size, maxNum):
    nums = []
    for i in range(size):
        nums.append(random.randint(0, maxNum))
    return nums

#Self made helper function that creates bucket sizes
def ranger(N, proc):
    results = []
    splits = N/proc

    start = 0
    end = 0

    for i in range(proc):
        start = end
        if i == proc-1:
            results.append([start+1, N])
        elif i == 0:
            end = math.floor(splits)
            results.append([start, end])
        else:
            end = math.floor(splits)
            results.append([start+1, end])
        splits += N/proc
    return results


#-----------Algorithms-----------#
#--------------------------------#
    
#-----------Bucket Sort----------#
#--------------------------------#
# Original single process algorithm taken from:
#https://www.geeksforgeeks.org/bucket-sort-2/
#Modifications to support any range of numbers and buckets added.

def insertionSort(b): 
    for i in range(1, len(b)): 
        up = b[i] 
        j = i - 1
        while j >=0 and b[j] > up:  
            b[j + 1] = b[j] 
            j -= 1
        b[j + 1] = up      
    return b      
              
def bucketSort(x, maxNum, buckets):
    arr = [] 
    rangeC = ranger(maxNum, buckets)

    for i in range(buckets):
        arr.append([]) 
  
    # Put array elements in different buckets  
    for j in x: 
        index_b = 0
        for choice in rangeC:
            if j == maxNum:
                arr[-1].append(j)
                break
            elif j >= choice[0] and j < choice[1]:
                arr[index_b].append(j)
                break
            else:
                index_b += 1

    # Sort individual buckets  
    for i in range(buckets): 
        arr[i] = insertionSort(arr[i]) 

    
    # concatenate the result 
    k = 0
    for i in range(buckets): 
        for j in range(len(arr[i])): 
            x[k] = arr[i][j] 
            k += 1
    return x 


#Parallel self-made version of bucketSort
def bucketSortPar(x, maxNum, buckets):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #Broadcast "root" (0) data so that all processes share the same correct data 
    x = comm.bcast(x, root = 0)

    arr = []
    rangeC = ranger(maxNum,buckets)[rank]

    #Put array elements in own bucket
    for j in x:
        if j >= rangeC[0] and j <= rangeC[1]:
            arr.append(j)

    #Sort own bucket
    arr = insertionSort(arr)

    #Concatenate results to root (0)
    data = comm.gather(arr, root=0)

    if rank == 0:
        data = [item for sublist in data for item in sublist]
        return data
    return None

#------------Testing-------------#
#--------------------------------#
        
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if size <= 1:
    print('Start at least 2 engines!')
else:
    timeContainer = []
    for i in range(15):
        lister = generate(2**(i+1), 2**31 - 1)
        lister2 = lister[:] #shallow copy only needed

        timer = time.time()
        result1 = bucketSort(lister, 2**31-1, size)
        origTime = time.time() - timer
        
        timer = time.time()
        result2 = bucketSortPar(lister2, 2**31-1, size)
        parTime = time.time() - timer

        if rank == 0:
            timeContainer.append("{} {}".format(origTime, parTime))
    if rank == 0:
        writer(timeContainer, "BucketSort_4.txt")
    '''
    lister = generate(10000, 2**31 - 1)
    lister2 = lister[:] #shallow copy only needed

    timer = time.time()
    result1 = bucketSort(lister, 2**31-1, size)

    origTime = time.time() - timer

    timer = time.time()
    result2 = bucketSortPar(lister2, 2**31-1, size)
    parTime = time.time() - timer
    
    if rank == 0:
        print(len(result1), len(result2))
        print(origTime, parTime)
        print(isSorted(result1), isSorted(result2))
        print(result1 == result2)
    '''
