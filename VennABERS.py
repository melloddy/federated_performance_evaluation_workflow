#### Modified Venn ABERS script for the MELLODDY project ####

#Original author : https://github.com/ptocca/VennABERS
# Straight-forward implementation of IVAP algorithm described in:
# Large-scale probabilistic prediction with and without validity guarantees, Vovk et al.
# https://arxiv.org/pdf/1511.00213.pdf
#
# Paolo Toccaceli
#
# https://github.com/ptocca/VennABERS


import numpy as np
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Some elementary functions to speak the same language as the paper
# (at some point we'll just replace the occurrence of the calls with the function body itself)
def push(x,stack):
    stack.append(x)

    
def pop(stack):
    return stack.pop()


def top(stack):
    return stack[-1]


def nextToTop(stack):
    return stack[-2]


# perhaps inefficient but clear implementation
def nonleftTurn(a,b,c):   
    d1 = b-a
    d2 = c-b
    return np.cross(d1,d2)<=0


def nonrightTurn(a,b,c):   
    d1 = b-a
    d2 = c-b
    return np.cross(d1,d2)>=0


def slope(a,b):
    ax,ay = a
    bx,by = b
    return (by-ay)/(bx-ax)


def notBelow(t,p1,p2):
    p1x,p1y = p1
    p2x,p2y = p2
    tx,ty = t
    m = (p2y-p1y)/(p2x-p1x)
    b = (p2x*p1y - p1x*p2y)/(p2x-p1x)
    return (ty >= tx*m+b)

kPrime = None

# Because we cannot have negative indices in Python (they have another meaning), I use a dictionary

def algorithm1(P):
    global kPrime
    
    S = []
    P[-1] = np.array((-1,-1))
    push(P[-1],S)
    push(P[0],S)
    for i in range(1,kPrime+1):
        while len(S)>1 and nonleftTurn(nextToTop(S),top(S),P[i]):
            pop(S)
        push(P[i],S)
    return S


def algorithm2(P,S):
    global kPrime
    
    Sprime = S[::-1]     # reverse the stack

    F1 = np.zeros((kPrime+1,))
    for i in range(1,kPrime+1):
        F1[i] = slope(top(Sprime),nextToTop(Sprime))
        P[i-1] = P[i-2]+P[i]-P[i-1]
        if notBelow(P[i-1],top(Sprime),nextToTop(Sprime)):
            continue
        pop(Sprime)
        while len(Sprime)>1 and nonleftTurn(P[i-1],top(Sprime),nextToTop(Sprime)):
            pop(Sprime)
        push(P[i-1],Sprime)
    return F1


def algorithm3(P):
    global kPrime

    S = []
    push(P[kPrime+1],S)
    push(P[kPrime],S)
    for i in range(kPrime-1,0-1,-1):  # k'-1,k'-2,...,0
        while len(S)>1 and nonrightTurn(nextToTop(S),top(S),P[i]):
            pop(S)
        push(P[i],S)
    return S


def algorithm4(P,S):
    global kPrime
    
    Sprime = S[::-1]     # reverse the stack
    
    F0 = np.zeros((kPrime+1,))
    for i in range(kPrime,1-1,-1):   # k',k'-1,...,1
        F0[i] = slope(top(Sprime),nextToTop(Sprime))
        P[i] = P[i-1]+P[i+1]-P[i]
        if notBelow(P[i],top(Sprime),nextToTop(Sprime)):
            continue
        pop(Sprime)
        while len(Sprime)>1 and nonrightTurn(P[i],top(Sprime),nextToTop(Sprime)):
            pop(Sprime)
        push(P[i],Sprime)
    return F0


def prepareData(calibrPoints):
    global kPrime
    
    ptsSorted = sorted(calibrPoints)
    
    xs = np.fromiter((p[0] for p in ptsSorted),float)
    ys = np.fromiter((p[1] for p in ptsSorted),float)
    ptsUnique,ptsIndex,ptsInverse,ptsCounts = np.unique(xs, 
                                                        return_index=True,
                                                        return_counts=True,
                                                        return_inverse=True)
    a = np.zeros(ptsUnique.shape)
    np.add.at(a,ptsInverse,ys)
    # now a contains the sums of ys for each unique value of the objects
    
    w = ptsCounts
    yPrime = a/w
    yCsd = np.cumsum(w*yPrime)   # Might as well do just np.cumsum(a)
    xPrime = np.cumsum(w)
    kPrime = len(xPrime)
    
    return yPrime,yCsd,xPrime,ptsUnique


def computeF(xPrime,yCsd):
    global kPrime
    P = {0:np.array((0,0))}
    P.update({i+1:np.array((k,v)) for i,(k,v) in enumerate(zip(xPrime,yCsd))})
    
    S = algorithm1(P)
    F1 = algorithm2(P,S)
    
    P = {0:np.array((0,0))}
    P.update({i+1:np.array((k,v)) for i,(k,v) in enumerate(zip(xPrime,yCsd))})    
    P[kPrime+1] = P[kPrime] + np.array((1.0,0.0))    # The paper says (1,1)
    
    S = algorithm3(P)
    F0 = algorithm4(P,S)
    
    return F0,F1


def getFVal(F0,F1,ptsUnique,testObjects):
    pos0 = np.searchsorted(ptsUnique,testObjects,side='left')
    pos1 = np.searchsorted(ptsUnique[:-1],testObjects,side='right')+1
    return F0[pos0],F1[pos1]


def ScoresToMultiProbs(calibrPoints,testObjects):
    # sort the points, transform into unique objects, with weights and updated values
    yPrime,yCsd,xPrime,ptsUnique = prepareData(calibrPoints)
    
    # compute the F0 and F1 functions from the CSD
    F0,F1 = computeF(xPrime,yCsd)
    
    # compute the values for the given test objects
    p0,p1 = getFVal(F0,F1,ptsUnique,testObjects)
                    
    return p0,p1


""" 
Additional functions below
author: lewis.mervin1@astrazeneca.com
"""

# inefficient but clear implementation
def get_VA_margin_cross(pts):
	margins = np.full(pts.shape[0], np.nan)
	p0s = np.full(pts.shape[0], np.nan)
	p1s = np.full(pts.shape[0], np.nan)
	skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=10)
	yhat = pts[:,0]
	ytrue = pts[:,1]
	for cal_index, test_index in skf.split(yhat, ytrue):
		yhat_cal, yhat_test = yhat[cal_index], yhat[test_index]
		y_cal = ytrue[cal_index]
		calibrPts = np.vstack((yhat_cal, y_cal)).T
		calibrPts = [tuple(i) for i in calibrPts]
		p0,p1 = ScoresToMultiProbs(calibrPts, yhat_test)
		margin = p1-p0
		margins[test_index]=margin
		p0s[test_index]=p0.reshape((-1,))
		p1s[test_index]=p1.reshape((-1,))
	return p0s,p1s,margins
		
# inefficient but clear implementation
def get_VA_margin_cross_external(pts,external):
	n_splits=3
	margins = np.full((external.shape[0],n_splits), np.nan)
	p0s = np.full((external.shape[0],n_splits), np.nan)
	p1s = np.full((external.shape[0],n_splits), np.nan)
	skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=10)
	yhat = pts[:,0]
	ytrue = pts[:,1]
	for idx, (cal_index, test_index) in enumerate(skf.split(yhat, ytrue)):
		yhat_cal, yhat_test = yhat[cal_index], yhat[test_index]
		y_cal = ytrue[cal_index]
		calibrPts = np.vstack((yhat_cal, y_cal)).T
		calibrPts = [tuple(i) for i in calibrPts]
		p0,p1 = ScoresToMultiProbs(calibrPts, external)
		margin = p1-p0
		margins[:,idx]=margin.reshape((-1,))
		p0s[:,idx]=p0.reshape((-1,))
		p1s[:,idx]=p1.reshape((-1,))
	return np.mean(p0s,axis=1),np.mean(p1s,axis=1),np.mean(margins,axis=1)