from os import listdir
from os.path import sep, exists, split
import pickle
import src.utils as ut

def loadData(rootDir):
    allDat = dict()

    dates = listdir(rootDir)
    for dt in dates:
        if dt == '.DS_Store':
            continue
        filesHere = listdir(sep.join([rootDir,dt]))
        for f in filesHere:
            if 'preProc' in f:
                infile = open(sep.join([rootDir,dt,f]),'rb')
                datNow = pickle.load(infile)
                infile.close()

                for e in datNow:
                    allDat[e] = dict()
                    for t in datNow[e]:
                        datNowMatch = {k: datNow[e][t][k] \
                                       for k in list(PPDat().__dict__.keys()) \
                                       if k in list(datNow[e][t].keys())}
                        allDat[e][t] = PPDat(**datNowMatch)

    return allDat

import numpy as np
import pandas as pd
import pingouin
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from scipy.signal import savgol_filter

class PPDat:
    def __init__(self, trialName = '',
                 meanMIP_G = np.zeros((256,256)), meanMIP_R = None,
                 allROIs = np.empty((1,1)),
                 DF_G = np.empty((1,1)), DF_R = None,
                 rawF_G = np.empty((1,1)), rawF_R = None,
                 behavDat = pd.DataFrame()):
        # Define the class members
        self.trialName = trialName
        self.meanMIP_G = meanMIP_G
        self.meanMIP_R = meanMIP_R
        self.allROIs = allROIs
        self.rawF_G = rawF_G
        self.behavDat = behavDat
        self.vR = None
        self.vF = None
        self.velCalc()
        self.visPeriods = None
        self.sortVis()
        self.DF_G = PPDat.sgFiltAct(DF_G)
        [self.PVA_G, self.PVAStren_G] = PPDat.PVACalc(self.DF_G)
        if DF_R is not None:
            self.DF_R = PPDat.sgFiltAct(DF_R)
            [self.PVA_R, self.PVAStren_R] = PPDat.PVACalc(self.DF_R)

    def sortVis(self):

         # Find the dark period
        darkPer = [i for i,v in enumerate(self.behavDat['olsdir']) if v == 0]

        # Find the open loop periods
        OLPer = [i for i,v in enumerate(self.behavDat['olgain']) if v > 0]
        CWPer = [i for i,v in enumerate(self.behavDat['olsdir']) if (v > 0) & (i in OLPer)]
        CCWPer = [i for i,v in enumerate(self.behavDat['olsdir']) if (v < 0) & (i in OLPer)]

        # Find the closed loop period
        CLPer = [i for i,v in enumerate(self.behavDat['olsdir']) if (v > 0) & (not (i in OLPer))]

        self.visPeriods = {'darkPer' : darkPer,
                           'OLPer' : OLPer,
                           'CLPer' : CLPer,
                           'CWPer' : CWPer,
                           'CCWPer' : CCWPer}

    def velCalc(self, sgoOrder = 3, sgoFrames = 11):
        dt = self.behavDat['Elapsed time'].diff()[1:].mean()
        self.vR = savgol_filter(
            self.behavDat['Rotational offset'].diff()[1:]/dt,
            sgoFrames,sgoOrder)*np.pi/180
        self.vF = savgol_filter(
            np.sqrt(self.behavDat['Forward offset'].diff()[1:]**2 + self.behavDat['Lateral offset'].diff()[1:]**2)/dt,
            sgoFrames,sgoOrder)

    @staticmethod
    def sgFiltAct(DF, sgoOrder = 3, sgoFrames = 11):
        DF_sg = np.zeros(DF.shape)
        for roi in range(0,DF_sg.shape[1]):
            DF_sg[:,roi] = savgol_filter(DF[:,roi],
                                         sgoFrames,sgoOrder)
        return DF_sg

    @staticmethod
    def PVACalc(DF):
        PVA = np.zeros((DF.shape[0],1))
        PVAStren = np.zeros((DF.shape[0],1))

        angs = np.linspace(np.pi,-np.pi, DF.shape[1], endpoint=False)
        for t in range(0,len(PVA)):
            PVA[t] = pingouin.circ_mean(angs,DF[t,:])
            PVAStren[t] = pingouin.circ_r(angs,DF[t,:]-DF[t,:].min())

        return [PVA, PVAStren]

    def plotROIs(self, ax, color = 'g'):
        if color == 'g':
            ax.imshow(self.meanMIP_G,cmap='Greens')
        if color == 'r':
            ax.imshow(self.meanMIP_R,cmap='Reds')
        ax.axis('off')
        for r in self.allROIs:
            ax.add_patch(Polygon([[pt[0],pt[1]] for pt in r], closed=True,fill=False,linestyle=':'))

    def plotrawF(self, ax, ActMap = 'Greens', aspect = 5, vmin = 0, vmax = 4E5):
        rawF = ax.imshow(self.rawF_G.T,cmap=ActMap,aspect=aspect,
                         extent = [self.behavDat['Elapsed time'][0],self.behavDat['Elapsed time'].iloc[-1],
                                   -np.pi,np.pi],
                         vmin = vmin, vmax = vmax)
        return rawF

    def plotDF(self, ax, color = 'g', aspect = 5, vmin = 0, vmax = 2):
        if color == 'r':
            act = self.DF_R.T
            ActMap = 'Reds'
        if color == 'g':
            act = self.DF_G.T
            ActMap = 'Greens'
        DF = ax.imshow(act,cmap=ActMap,aspect=aspect,
                       extent = [self.behavDat['Elapsed time'][0],self.behavDat['Elapsed time'].iloc[-1],
                                 -np.pi,np.pi],
                       vmin = vmin, vmax = vmax)
        return DF

    def plotPVA(self, ax, PVAthresh, color = 'g'):
        if color == 'g':
            PVA = self.PVA_G
            PVAStren = self.PVAStren_G
        if color == 'r':
            PVA = self.PVA_R
            PVAStren = self.PVAStren_R
        PVA = ut.removeJumps(PVA, thresh = np.pi)
        PVA[PVAStren < PVAthresh] = None
        PVAPlt = ax.plot(self.behavDat['Elapsed time'],
                         PVA, color = color)
        ax.set_xlim(self.behavDat['Elapsed time'][0],self.behavDat['Elapsed time'].iloc[-1])
        ax.set_ylim(-np.pi,np.pi)
        ax.set(yticks = np.linspace(-np.pi, np.pi, 5),
               yticklabels=[u'-${\pi}$',u'-${\pi}$/2','0',u'${\pi}/2$',u'${\pi}$'])

    def plotPVAStren(self, ax, color = 'g', aspect = 5):
        if color == 'g':
            PVAStren = self.PVAStren_G.T
        if color == 'r':
            PVAStren = self.PVAStren_R.T
        PVAStrenPlt = ax.imshow(PVAStren, cmap = 'Greys', aspect = aspect,
                             extent = [self.behavDat['Elapsed time'][0],self.behavDat['Elapsed time'].iloc[-1],0,1],
                             vmin = 0, vmax = 1)

    def plotPVAStrenHist(self, ax, PVAbins = np.linspace(0,1,21), fc = (0,0,0,1), PVAthresh = 0.2):
        ax.hist(self.PVAStren_G, bins = PVAbins, fc = fc)
        ax.axvline(x = PVAthresh, linestyle = '--', color = 'k')
        ax.set_ylabel('counts')
        ax.set_xlabel('PVA strength')

    def plotvRHist(self, ax, vRbins = np.linspace(-np.pi,np.pi,41), fc = (0,0,0,1)):
        ax.hist(self.vR, bins = vRbins, fc = fc)
        ax.set_ylabel('counts')
        ax.set_xlabel('vR')

    def plotvFHist(self, ax, vFbins = np.linspace(0,2,21), fc = (0,0,0,1)):
        ax.hist(self.vF, bins = vFbins, fc = fc)
        ax.set_ylabel('counts')
        ax.set_xlabel('vF')

    def plotPVAStrenHist(self,ax, PVAbins = np.linspace(0,1,21), fc = (0,0,0,1), PVAthresh = 0.2):
        ax.hist(self.PVAStren_G,bins = PVAbins, fc = fc)
        ax.axvline(x = PVAthresh, linestyle = '--', color = 'k')

    def plotRot(self,ax):
        ax.plot(self.behavDat['Elapsed time'][self.visPeriods['darkPer']],
                self.behavDat['Rotational offset'][self.visPeriods['darkPer']] % 360,
                color = 'k')
        ax.plot(self.behavDat['Elapsed time'][self.visPeriods['CLPer']],
                self.behavDat['Rotational offset'][self.visPeriods['CLPer']] % 360,
                color = 'b')
        ax.plot(self.behavDat['Elapsed time'][self.visPeriods['OLPer']],
                self.behavDat['Rotational offset'][self.visPeriods['OLPer']] % 360,
                color = 'pink')

    def plotPos(self,ax):
        ax.plot(self.behavDat['Forward offset'][self.visPeriods['darkPer']],
                self.behavDat['Lateral offset'][self.visPeriods['darkPer']],
                color = 'k')
        ax.plot(self.behavDat['Forward offset'][self.visPeriods['CLPer']],
                self.behavDat['Lateral offset'][self.visPeriods['CLPer']],
                color = 'b')

    def PositionConverter(self, Cam1RotCalibfact, Cam2RotCalibfact, Cam1PosCalibfact, Cam2PosCalibfact):
        flyAng = 30*np.pi/180

        posRot = np.zeros((len(self.behavDat['Rotational offset']),1))
        posFor = np.zeros((len(self.behavDat['Forward offset']),1))
        posLat = np.zeros((len(self.behavDat['Lateral offset']),1))

        posRot[0] = self.behavDat['Rotational offset'][0]
        posFor[0] = self.behavDat['Forward offset'][0]
        posLat[0] = self.behavDat['Lateral offset'][0]

        dxmod0 = self.behavDat.dx0*np.cos(flyAng) + self.behavDat.dy0*np.sin(flyAng)
        dymod0 = self.behavDat.dy0*np.cos(flyAng) + self.behavDat.dx0*np.sin(flyAng)
        dxmod1 = self.behavDat.dx1*np.cos(flyAng) + self.behavDat.dy1*np.sin(-flyAng)
        dymod1 = self.behavDat.dy1*np.cos(flyAng) + self.behavDat.dx1*np.sin(-flyAng)

        deltaFor = (dymod0 / Cam1PosCalibfact + dymod1 / Cam2PosCalibfact)*np.sqrt(2) / 2
        deltaSide = (dymod0 / Cam1PosCalibfact - dymod1 / Cam2PosCalibfact)*np.sqrt(2) / 2

        RotOffset = self.behavDat['Rotational offset'][0] - (dxmod0[0]*Cam1RotCalibfact + dxmod1[0]*Cam2RotCalibfact)/2

        for i in range(1,len(dxmod0)):
            posRot[i] = (dxmod0[i]*Cam1RotCalibfact + dxmod1[i]*Cam2RotCalibfact)/2 + RotOffset;
            posFor[i] = (deltaFor[i]-deltaFor[i-1])*np.cos(posRot[i]*np.pi/180) + \
            (deltaSide[i]-deltaSide[i-1])*np.sin(posRot[i]*np.pi/180) + posFor[i-1]
            posLat[i] = (deltaFor[i]-deltaFor[i-1])*np.sin(posRot[i]*np.pi/180) - \
            (deltaSide[i]-deltaSide[i-1])*np.cos(posRot[i]*np.pi/180) + posLat[i-1]

        return [posRot, posFor, posLat]
