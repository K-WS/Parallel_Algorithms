from nose.tools import assert_equal
from mpi4py import MPI
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

#Merger, to parallel combine leftover list values in... mergeSort
#Note that these are designed so that 0 is always the root.
def merger(arr, size, root):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    #print(rank, len(arr))

    mrgStatus = list(range(size))
    cycler = 0
    iteration = 0
    
    while len(mrgStatus) > 1:
        
        #Cycler reset
        if cycler >= len(mrgStatus):
            cycler = 0

        #Protection against out of bounds
        if mrgStatus[cycler] != mrgStatus[-1]:
            
            #Receiver
            if rank == mrgStatus[cycler]:
                R = comm.recv(source = mrgStatus[cycler+1], tag = 1000 + iteration)

                #Merge the original arr with the data it received
                #Using copied mergeSort style to be identical in time here
                #First, assign L and R as subarrays and make arr "empty" array
                #with size of len(L)+len(R)
                L = arr[:]

                arr = [0] * (len(L) + len(R))
                
                #Rest of the proces
                i = j = k = 0
          
                # Copy data to temp arrays L[] and R[] 
                while i < len(L) and j < len(R): 
                    if L[i] < R[j]: 
                        arr[k] = L[i] 
                        i+=1
                    else: 
                        arr[k] = R[j] 
                        j+=1
                    k+=1
                  
                # Checking if any element was left 
                while i < len(L): 
                    arr[k] = L[i] 
                    i+=1
                    k+=1
                  
                while j < len(R): 
                    arr[k] = R[j] 
                    j+=1
                    k+=1

            #Sender
            elif rank == mrgStatus[cycler+1]:
                comm.send(arr, dest = mrgStatus[cycler], tag = 1000 + iteration)

            #Remove the data that was merged
            mrgStatus.pop(cycler+1)

        #Update loop
        cycler += 1
        iteration += 1

    if rank == mrgStatus[0]:
        return arr
    return None
    
#-----------Algorithms-----------#
#--------------------------------#

#-----------Merge Sort-----------#
#--------------------------------#
# Original single process algorithm taken from:
#https://www.geeksforgeeks.org/merge-sort/
def mergeSort(arr): 
    if len(arr) >1: 
        mid = len(arr)//2 #Finding the mid of the array 
        L = arr[:mid] # Dividing the array elements  
        R = arr[mid:] # into 2 halves 
  
        mergeSort(L) # Sorting the first half 
        mergeSort(R) # Sorting the second half 
  
        i = j = k = 0
          
        # Copy data to temp arrays L[] and R[] 
        while i < len(L) and j < len(R): 
            if L[i] < R[j]: 
                arr[k] = L[i] 
                i+=1
            else: 
                arr[k] = R[j] 
                j+=1
            k+=1
          
        # Checking if any element was left 
        while i < len(L): 
            arr[k] = L[i] 
            i+=1
            k+=1
          
        while j < len(R): 
            arr[k] = R[j] 
            j+=1
            k+=1

#Alternate parallel solution to Merge Sort, self made
#1, 2, 4, 8 processes, if less than one of these numbers, takes the previous
def mergeSortPar(arr):
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
    #spArr = np.array_split(arr, limit)
    spArr = list(splitter(arr, limit))

    #Now every eligible subprocess sorts their part    
    if rank < len(spArr):
        mergeSort(spArr[rank])
        #Finally, combination and sorting of sorted arrays manually
        return merger(spArr[rank], limit, 0)
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
        mergeSort(lister)
        origTime = time.time() - timer
        
        timer = time.time()
        result2 = mergeSortPar(lister2)
        parTime = time.time() - timer

        if rank == 0:
            timeContainer.append("{} {}".format(origTime, parTime))
    if rank == 0:
        writer(timeContainer, "MergeSort_4.txt")
        
            
        
    #lister = generate(10000, 2**31 - 1)
    #lister2 = lister[:] #shallow copy only needed
    
    #timer = time.time()
    #mergeSort(lister)
    #origTime = time.time() - timer
    
    #timer = time.time()
    #result2 = mergeSortPar(lister2)
    #parTime = time.time() - timer
    
    #if rank == 0:
    #    print(len(lister), len(result2))
    #    print(origTime, parTime)
    #    print(isSorted(lister), isSorted(result2))
    #    print(lister == result2)
