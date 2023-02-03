import numpy as np

def removeJumps(dat, thresh):
    datNoJump = np.array(dat)

    datDiff = np.array(dat[:-1]) - np.array(dat[1:])
    for i,d in enumerate(datDiff):
        if abs(d) > thresh:
            datNoJump[i+1] = None
            
    return datNoJump