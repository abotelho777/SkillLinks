import numpy as np
from scipy import misc as msc
import os
from scipy.stats import bernoulli,norm as normal
import random

def bern(p,shape=1):
    if shape == 1:
        return bernoulli.rvs(p,size=shape)[0]
    return bernoulli.rvs(p,size=shape)

def norm(shape=1,center=0):
    if shape == 1:
        return normal.rvs(loc=center,size=shape)[0]
    return normal.rvs(loc=center,size=shape)

def rand(min=0,max=1,size=1):
    rv = []
    for i in range(0,size):
        if min==0 and max==1:
            rv.append(random.randrange(min,1000)/1000.0)
        else:
            rv.append(random.randrange(min,max))
    if size == 1:
        return rv[0]
    else:
        return rv

def MIN(x,y):
    if (x < y):
        return x
    else:
        return y

def MAX(x,y):
    if (x > y):
        return x
    else:
        return y

def diceRoll(sides):
    return rand(0,sides)

def clamp(x,min,max):
    return MIN(max, MAX(min, x))

def exists(x,arr):
    for i in range(0,len(arr)):
        if x == arr[i]:
            return True
    return False

def normalize(v, method='max'):
    m = np.nanmean(v)
    s = np.nanstd(v)
    mx = np.nanmax(v)
    if mx == 0:
        mx = 0.000001
    me = np.nanmedian(v)
    mi = np.nanmin(v)
    nv = []
    for i in range(0,len(v)):
        if method == 'max':
            nv.append(v[i]/mx)
        elif method == 'zscore':
            nv.append((v[i]-m)/s)
        elif method == 'median':
            nv.append(v[i]/me)
        elif method == 'uniform':
            nv.append((v[i]-mi)/(mx-mi))
        else:
            print("ERROR - UNKNOWN METHOD")

    return nv

def loadIMG(filename,grayscale=0):
    try:
        img = msc.imread(filename,grayscale)
    except ValueError:
        print("ERROR loading file:",filename)
        return []
    return img

def getfilenames(extension=".*",directory=os.path.dirname(os.path.realpath(__file__))):
    names = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            names.append(directory + "/" + file)
    return names

def loadCSV(filename):
    csvarr = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            # split out each comma-separated value
            name = line.strip().split(',')
            for j in range(0,len(name)):
                # try converting to a number, if not, leave it
                try:
                    name[j] = float(name[j])
                except ValueError:
                    # do nothing constructive
                    name[j] = name[j]
            csvarr.append(name)

    return csvarr

def writetoCSV(ar,filename):
    ar = np.array(ar)
    assert len(ar.shape) <= 2

    with open(filename + '.csv', 'w') as f:
        for i in range(0,len(ar)):
            if (len(ar.shape) == 2):
                for j in range(0,len(ar[i])-1):
                    f.write(str(ar[i][j]) + ',')
                f.write(str(ar[i][len(ar[i])-1]) + '\n')
            else:
                f.write(str(ar[i]) + '\n')
    f.close()

def loadCSVwithHeaders(filename):
    data = loadCSV(filename)
    headers = np.array(data[0])
    data = np.array(data)
    data = np.delete(data, 0, 0)
    return data,headers

def getColumn(ar,col,startRow = 0):
    c = []

    # get number of rows and columns
    numR = len(ar)
    # return if index out of bounds
    if startRow >= numR:
        print("INDEX OUT OF BOUNDS")
        return c
    numC = len(ar[startRow])
    # return if index out of bounds
    if col >= numC:
        print("INDEX OUT OF BOUNDS")
        return c



    # create an array from the column values
    for i in range(startRow, numR-1):
        c.append(ar[i][col])



    return c

def unique(ar):
    ulist = []

    for i in range(0,len(ar)):
        if ar[i] not in ulist:
            ulist.append(ar[i])
    return ulist

def fold(x,y,folds=2):
    assert folds > 0
    assert len(x) == len(y)
    f = []
    f_l = []

    for i in range(0,folds):
        f.append([])
        f_l.append([])

    for i in range(0,len(x)):
        rindex = np.random.randint(0, folds)
        f[rindex].append(x[i])
        f_l[rindex].append(y[i])

    return np.array(f),np.array(f_l)


def split_training_test(x,y,training_size=0.8):
    assert len(x) == len(y)
    X_Train = []
    X_Test = []
    Y_Train = []
    Y_Test = []

    for i in range(0,int(len(x)*training_size)):
        X_Train.append(x[i])
        Y_Train.append(y[i])
    for i in range(len(X_Train)+1,len(x)):
        X_Test.append(x[i])
        Y_Test.append(y[i])

    return np.array(X_Train),np.array(X_Test),np.array(Y_Train),np.array(Y_Test)

def shuffle(data,labels=None):

    if labels is not None:
        assert len(data) == len(labels)

    for i in range(0,len(data)):
        rindex = np.random.randint(0,len(data))
        tmpx = data[rindex]
        data[rindex] = data[i]
        data[i] = tmpx

        if labels is not None:
            tmpy = labels[rindex]
            labels[rindex] = labels[i]
            labels[i] = tmpy

    if labels is None:
        return data

    return data,labels

def select(ar, val, op = '==',column = None):
    aprime = []

    if column is not None:
        if (op == '=='):
            aprime = [element for element in ar if element[column] == val]
        elif (op == '<='):
            aprime = [element for element in ar if element[column] <= val]
        elif (op == '>='):
            aprime = [element for element in ar if element[column] >= val]
        elif (op == '<'):
            aprime = [element for element in ar if element[column] < val]
        elif (op == '>'):
            aprime = [element for element in ar if element[column] > val]
        elif (op == '!='):
            aprime = [element for element in ar if element[column] != val]
        else:
            print 'Unknown Operation'
    else:
        if (op == '=='):
            aprime = [element for element in ar if element == val]
        elif (op == '<='):
            aprime = [element for element in ar if element <= val]
        elif (op == '>='):
            aprime = [element for element in ar if element >= val]
        elif (op == '<'):
            aprime = [element for element in ar if element < val]
        elif (op == '>'):
            aprime = [element for element in ar if element > val]
        elif (op == '!='):
            aprime = [element for element in ar if element != val]
        else:
            print 'Unknown Operation'
    return aprime

def select_d(ar, val, op = '==',column = -1):
    aprime = []
    assert len(ar) > 0
    if (column == -1):
        for i in range(0,len(ar)):
            if (op == '=='):
                if (ar[i] == val):
                    aprime.append(ar[i])
            elif (op == '<='):
                if (ar[i] <= val):
                    aprime.append(ar[i])
            elif (op == '>='):
                if (ar[i] >= val):
                    aprime.append(ar[i])
            elif (op == '<'):
                if (ar[i] < val):
                    aprime.append(ar[i])
            elif (op == '>'):
                if (ar[i] > val):
                    aprime.append(ar[i])
            elif (op == '!='):
                if (ar[i] != val):
                    aprime.append(ar[i])
            else:
                print 'Unknown Operation'
    else:
        for i in range(0, len(ar)):
            assert len(ar[i]) > column+1
            if (op == '=='):
                if (ar[i][column] == val):
                    aprime.append(ar[i])
            elif (op == '<='):
                if (ar[i][column] <= val):
                    aprime.append(ar[i])
            elif (op == '>='):
                if (ar[i][column] >= val):
                    aprime.append(ar[i])
            elif (op == '<'):
                if (ar[i][column] < val):
                    aprime.append(ar[i])
            elif (op == '>'):
                if (ar[i][column] > val):
                    aprime.append(ar[i])
            elif (op == '!='):
                if (ar[i][column] != val):
                    aprime.append(ar[i])
            else:
                print 'Unknown Operation'
    return aprime


def numerate(ar):
    u = unique(ar)

    for i in range(0,len(ar)):
        for j in range(0,len(u)):
            if ar[i] == u[j]:
                ar[i] = j
                break
    return ar

def transpose(ar):
    L = len(ar)
    if isinstance(ar[0],float):
        for i in range(0,L):
            ar[i] = [ar[i]]
        W = 1
    else:
        W = len(ar[0])

    nArr = []

    for i in range(0, W):
        nrow = []
        for j in range(0,L):
            nrow.append(ar[j][i])
        nArr.append(nrow)

    return nArr

glob_proc = []
async_running = 0
from multiprocessing import Process, Lock, Array
lock = 0

def BuildThreadedArray(arr):
    return Array(arr,0,lock=lock)

def run_async(function,*args):
    global async_running
    global glob_proc
    global lock

    if lock == 0:
        lock = Lock()

    try: async_running += 1
    except UnboundLocalError: async_running = 1
    from multiprocessing import Process
    if (len(args)!=0):
        p = Process(target=function,args=(args))
    else:
        p = Process(target=function)


    p.start()

    glob_proc.append(p)
    return p

def wait_for_async():
    global async_running
    global glob_proc
    from multiprocessing import Process
    if async_running == 0:
        return

    while len(glob_proc) < async_running:
        pass

    for i in range(0,len(glob_proc)):
        glob_proc[i].join()

    glob_proc = []
    async_running = 0

def lock():
    global lock
    lock.acquire()

def unlock():
    global lock
    lock.release()


