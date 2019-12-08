from nose.tools import assert_equal
from mpi4py import MPI
#import numpy as np
import random
import string
import copy
import time


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

#List splitter, taken from https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
#We want to minimize numpy usage, which naturally makes calculations
#faster.
def splitter(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))

#-----------Algorithms-----------#
#--------------------------------#

#----------Bitonic Sort----------#
#--------------------------------#
# Original single process algorithm taken from:
#https://www.geeksforgeeks.org/bitonic-sort/

# The parameter dir indicates the sorting direction, ASCENDING 
# or DESCENDING; if (a[i] > a[j]) agrees with the direction, 
# then a[i] and a[j] are interchanged.*/ 
def compAndSwap(a, i, j, dire): 
    if (dire==1 and a[i] > a[j]) or (dire==0 and a[i] < a[j]): 
        a[i],a[j] = a[j],a[i] 
  
# It recursively sorts a bitonic sequence in ascending order, 
# if dir = 1, and in descending order otherwise (means dir=0). 
# The sequence to be sorted starts at index position low, 
# the parameter cnt is the number of elements to be sorted. 
def bitonicMerge(a, low, cnt, dire): 
    if cnt > 1: 
        k = cnt/2
        for i in range(low , low+k): 
            compAndSwap(a, i, i+k, dire) 
        bitonicMerge(a, low, k, dire) 
        bitonicMerge(a, low+k, k, dire) 
  
# This funcion first produces a bitonic sequence by recursively 
# sorting its two halves in opposite sorting orders, and then 
# calls bitonicMerge to make them in the same order 
def bitonicSort(a, low, cnt,dire): 
    if cnt > 1: 
          k = cnt/2
          bitonicSort(a, low, k, 1) 
          bitonicSort(a, low+k, k, 0) 
          bitonicMerge(a, low, cnt, dire) 
  
# Caller of bitonicSort for sorting the entire array of length N 
# in ASCENDING order 
def sort(a,N, up): 
    bitonicSort(a,0, N, up) 

#Parallel variant
#0 is the root
def bitonicSortPar(arr):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #Broadcast "root" (0) data so that all processes share the same correct data 
    arr = comm.bcast(arr, root = 0)
    
    #Pick a suitable amount of processes to use
    limit = 1
    while limit <= size:
        limit = limit * 2
    limit = limit / 2

    #Subdivide array to "limit" parts
    #spArr = np.array_split(arr, size)
    spArr = list(splitter(arr, limit))

    #Every eligible subprocess sequences (and merges) their part
    if rank < len(spArr):
        rankArr = spArr[rank]
        #print(rankArr)
        low = 0
        dire = 1
        cnt = len(rankArr)
        
        direction = rank % 2
        direction = 1 - direction
        bitonicSort(rankArr, 0, cnt, direction)
        #print(rankArr)

        #Now the ender has to only worry about finishing the final merges
        mrgStatus = list(range(limit))
        cycler = 0
        iteration = 0
        direction = 1
        
        while len(mrgStatus) > 1:
            
            #Cycler reset
            if cycler >= len(mrgStatus):
                cycler = 0

            #Protection against out of bounds
            if mrgStatus[cycler] != mrgStatus[-1]:
                
                #Receiver
                if rank == mrgStatus[cycler]:
                    R = comm.recv(source = mrgStatus[cycler+1], tag = 1000 + iteration)

                    #combine lists together
                    rankArr.extend(R)
                    #Bitonic merge the new list together
                    bitonicMerge(rankArr, 0, len(rankArr), direction)
                    #print(rankArr)

                #Sender
                elif rank == mrgStatus[cycler+1]:
                    comm.send(rankArr, dest = mrgStatus[cycler], tag = 1000 + iteration)

                #Remove the data that was merged
                mrgStatus.pop(cycler+1)

            #Update loop
            cycler += 1
            iteration += 1
            direction = 1 - direction

        if rank == mrgStatus[0]:
            return rankArr
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
        sort(lister, len(lister), 1)
        origTime = time.time() - timer
        
        timer = time.time()
        result2 = bitonicSortPar(lister2)
        parTime = time.time() - timer

        if rank == 0:
            timeContainer.append("{} {}".format(origTime, parTime))
    if rank == 0:
        writer(timeContainer, "BitonicSort_4.txt")

    '''

    lister = generate(8192, 2**31 - 1)
    #lister = generate(16, 2**31 - 1)
    #lister = [14,15,14,13,12,11,10,9,8248213902,7,6,5,4,3,2,1]
    lister2 = lister[:] #shallow copy only needed
    
    timer = time.time()
    sort(lister, len(lister), 1)
    origTime = time.time() - timer
    
    timer = time.time()
    result2 = bitonicSortPar(lister2)
    parTime = time.time() - timer
    
    if rank == 0:
        print(len(lister), len(result2))
        print(origTime, parTime)
        print(isSorted(lister), isSorted(result2))
        print(lister == result2)
    '''
