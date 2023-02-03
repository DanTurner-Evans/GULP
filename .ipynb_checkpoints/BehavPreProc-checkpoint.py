import re
import numpy as np
import pandas as pd

def loadBehavDat(fname):
    ''' Load the behavioral data
    '''
    
    et =[]; ro = []; fo = []; lo = []
    dx0 = []; dx1 = []; dy0 = []; dy1 = []
    cl = []; old = []; tr = []; olg = []; clg = []

    with open(fname, 'r') as fh:
        for i,line in enumerate(fh):
            if i>3:
                prts = re.split('\t|\n',line)
                for j,p in enumerate(prts):
                    if j == 1: et.append(float(prts[j]))
                    if j == 3: ro.append(float(prts[j]))
                    if j == 5: fo.append(float(prts[j]))
                    if j == 7: lo.append(float(prts[j]))
                    if j == 9: dx0.append(float(prts[j]))
                    if j == 11: dx1.append(float(prts[j]))
                    if j == 13: dy0.append(float(prts[j]))
                    if j == 15: dy1.append(float(prts[j]))
                    if j == 17: cl.append(int(prts[j]))
                    if j == 19: old.append(int(prts[j]))
                    if j == 21: tr.append(int(prts[j]))
                    if j == 23: olg.append(float(prts[j]))
                    if j == 25: clg.append(float(prts[j]))

    behavDat = pd.DataFrame({'Elapsed time': et,
                  'Rotational offset': ro, 'Forward offset': fo, 'Lateral offset': lo,
                  'dx0': dx0, 'dx1': dx1, 'dy0': dy0, 'dy1': dy1,
                  'closed': cl, 'olsdir': old, 'trans': tr,
                  'olgain': olg, 'clgain': clg
                })
    
    return behavDat

def getSYNCTimes(SYNCDatNm, fpv):
    ''' Find the framegrab and VR display points
    '''
    # Load the voltages from the synchronization data
    SYNCDat = pd.read_csv(SYNCDatNm,header=None, names = ['VFramegrab','VVR','VStim','VPuff'])

    # Get the points where each framegrab starts - use a constant fraction discriminator
    cfd_fg = SYNCDat['VFramegrab'][:-1].reset_index(drop=True)-SYNCDat['VFramegrab'][1:].reset_index(drop=True)
    tFramegrab = cfd_fg[cfd_fg>0.5].index
    tFramegrab = np.delete(tFramegrab,np.argwhere(np.diff(tFramegrab) < 20)+1)
    tFramegrab = tFramegrab[0::fpv]

    # Get the points where each R, G, or B frame is projected - use a constant fraction discriminator
    cfd_vr = SYNCDat['VVR'][:-1].reset_index(drop=True)-SYNCDat['VVR'][1:].reset_index(drop=True)
    tVR = cfd_vr[cfd_vr>0.05].index
    tVR = np.delete(tVR,np.argwhere(np.diff(tVR) < 10)+1)
    tVR = tVR[0::3]
    
    # Get the points where the iontrophoresis function generator is outputting a high signal
    tPf = SYNCDat['VPuff'][SYNCDat['VPuff'] > 1].index
    
    return [tFramegrab, tVR, tPf]

def getMatchedBehavDat(tFramegrab, tVR, behavDatNm):
    ''' Get the behavioral data for each imaging volume
    '''
    # Load the behavioral data
    behavDat = loadBehavDat(behavDatNm)

    # Select only the times where the VR was active
    framesToUse = np.where((tFramegrab > np.min(tVR)) & (tFramegrab < np.max(tVR)))

    # Create a dataframe with the relevant timepoints and behavioral values at each framegrab point
    datPts = [] ; 
    for t in tFramegrab[framesToUse]:
        datPts.append(np.argmin(np.abs(tVR-t)))
    behavDat_matched = behavDat.loc[datPts].reset_index(drop=True)
    
    return [framesToUse, behavDat_matched]

def matchDAQtoVR(tVR, behavDat, tMatch):
    ''' Match DAQ data to the behavioral data
    '''

    # Create a dataframe with the relevant timepoints and behavioral values at each framegrab point
    datPts = [] ; 
    for t in tMatch:
        datPts.append(np.argmin(np.abs(tVR-t)))
    matchedTimes = behavDat['Elapsed time'].loc[datPts].reset_index(drop=True)
    
    return matchedTimes