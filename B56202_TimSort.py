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

    print(rank, len(arr))

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

                arr.extend(R)
                arr = sorted(arr)

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

#------------Tim Sort------------#
#--------------------------------#
#Single process implementation, since it's already built in, it doesn't really
#need much to be done
def timSort(arr):
    return sorted(arr)

#Parallel implementation, note that since this is a combination of
#insertion sort and merge sort, we do a manual merge sort
#on the remaining sublists.
def timSortPar(arr):
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
        endArr = timSort(spArr[rank])
        #Finally, combination and sorting of sorted arrays manually
        return merger(endArr, limit, 0)
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
        
        lister = generate(2**(i+10), 2**31 - 1)
        lister2 = lister[:] #shallow copy only needed

        timer = time.time()
        result1 = timSort(lister)
        origTime = time.time() - timer
        
        timer = time.time()
        result2 = timSortPar(lister2)
        parTime = time.time() - timer

        if rank == 0:
            timeContainer.append("{} {}".format(origTime, parTime))
    if rank == 0:
        writer(timeContainer, "TimSort_4_ex.txt")

    '''
    lister = generate(100000, 2**31 - 1)
    #lister = [8,3,9,4,7,4,6,2,8,4,86,4,21,4,7,3]
    lister2 = lister[:] #shallow copy only needed
    
    timer = time.time()
    result1 = timSort(lister)
    origTime = time.time() - timer
    
    timer = time.time()
    result2 = timSortPar(lister2)
    parTime = time.time() - timer
    
    if rank == 0:
        print(len(result1), len(result2))
        print(origTime, parTime)
        print(isSorted(result1), isSorted(result2))
        print(result1 == result2)
        #print(result2)
    '''
