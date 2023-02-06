import numpy as np
import pandas as pd

def removeJumps(dat, thresh):
    datNoJump = np.array(dat)

    datDiff = np.array(dat[:-1]) - np.array(dat[1:])
    for i,d in enumerate(datDiff):
        if abs(d) > thresh:
            datNoJump[i+1] = None

    return datNoJump

def corrOverTime(x,y,time, pre_time, post_time):
    dt = np.round(np.diff(time).mean(),2)
    pre_time_pts = int(np.round(pre_time/dt))
    post_time_pts = int(np.round(post_time/dt))
    pt_range = np.arange(-pre_time_pts, post_time_pts)

    corr = pd.DataFrame({'time':[],'corr':[]})
    for pt in pt_range:
        corr_val = np.corrcoef(x[max(0,pt):min(len(x),len(x)+pt)],y[max(0,-pt):min(len(y),len(y)-pt)])[0,1]
        corr = pd.concat([corr, pd.DataFrame({'time':[dt*pt],'corr':[corr_val]})])

    return corr
