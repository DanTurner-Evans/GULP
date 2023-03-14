from ScanImageTiffReader import ScanImageTiffReader
import tifffile as tf
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift, gaussian_filter
import numpy as np
import math
import napari
from napari.settings import SETTINGS # Changed from from napari.utils.settings import SETTINGS
SETTINGS.application.ipy_interactive = False
import cv2 as cv
import time
import pandas as pd
from pathlib import Path
import os
from os import listdir
from os.path import sep, exists
from matplotlib import pyplot as plt
import math
import pickle
from datetime import datetime
import tkinter as tk
from tkinter import filedialog


def loadFileNames(single_file=False):
    """Prompt user to select one or multiple files

    Returns:
        list: list of filenames of trials
    """

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", 1)

    if single_file:
        trial_file_nms = filedialog.askopenfilename(title="Select files")
    else:
        trial_file_nms = filedialog.askopenfilenames(title="Select files")
    return trial_file_nms

def loadTrialInfo(rootDirs):
    """
    Gets the experiment and trial information for calcium imaging experiments

    Arguments:
        rootDir = the root directories. Each directory should contain a series of subdirectories
        with the dates of the data collection

    Returns:
        trials = a dictionary of all of the trial information keyed by the name of each experiment
    """

    trials = dict()
    for d in rootDirs:
        dates = listdir(d)
        for dt in dates:
            if dt == '.DS_Store':
                continue
            files_here = listdir(sep.join([d,dt]))
            expts_here = list(set(
                ["_".join(sep.join([d,dt,f]).split('_')[0:-2]) for f in files_here if ('tif' in f)]))
            for e in expts_here:
                trials[e] = sorted([sep.join([d,dt,f]) for f in files_here if (('tif' in f) & (e.split(sep)[-1] in f))])

    return trials

def loadTif(path):
    """ Load in a Scan Image tiff from the specified path
    """

    # Make a tiff reader object
    mytiffreader = ScanImageTiffReader(path)

    # Get the metadata
    [nCh, discardFBFrames, nDiscardFBFrames, fpv, nVols] = tifMetadata(path)

    # Load the tif data
    vol = mytiffreader.data()

    # Reshape the volume to reflect the experimental parameters
    vol = vol.reshape((int(vol.shape[0]/(fpv*nCh)),fpv,nCh,vol.shape[1], vol.shape[2]))

    # Discard the flyback frames
    stack4d = vol[:,0:fpv-nDiscardFBFrames,:,:,:]

    return [stack4d, nCh, nDiscardFBFrames, fpv]

def tifMetadata(path):
    """ Load Scan Image tiff metadata from a Scan Image tiff reader object
    """

    mytiffreader = ScanImageTiffReader(path)
    metadat = mytiffreader.metadata()

    # If metadat is empty, it is not a ScanImage tiff file so use a general tiff metadata reader
    if not metadat:
        with tf.TiffFile(path) as tif:
            imagej_metadata = tif.imagej_metadata

            SizeC = int(imagej_metadata.get('channels', 1))
            SizeT = int(imagej_metadata.get('frames', 1))
            SizeZ = int(imagej_metadata.get('images', 1) / SizeC / SizeT)

            nCh = SizeC
            discardFBFrames = None
            nDiscardFBFrames = 0 # Assuming no flyback frames
            fpv = int(SizeT / SizeZ)
            nVols = SizeZ

    else:
        # Step through the metadata, extracting relevant parameters
        for i, line in enumerate(metadat.split('\n')):
            if not 'SI.' in line: continue

            # get channel info
            if 'channelSave' in line:
                if not '[' in line:
                    nCh = 1
                else:
                    nCh = len(line.split('=')[-1].split(sep=';'))

            if 'scanFrameRate' in line:
                fpsscan = float(line.split('=')[-1].strip())

            if 'discardFlybackFrames' in line:
                discardFBFrames = line.split('=')[-1].strip()

            if 'numDiscardFlybackFrames' in line:
                nDiscardFBFrames = int(line.split('=')[-1].strip())

            if 'numFramesPerVolume' in line:
                fpv = int(line.split('=')[-1].strip())

            if 'numVolumes' in line:
                nVols = int(line.split('=')[-1].strip())

    return [nCh, discardFBFrames, nDiscardFBFrames, fpv, nVols]

def getDate(path):
    """Get the date for a tiff file

    Args:
        path (str or pathlib.Path): path to tiff file

    Returns:
        datetime: time tiff was captured
    """
    # Load metadata
    with tf.TiffFile(path) as tif:
        imagej_metadata = tif.imagej_metadata

    # Search metadata for date
    info = imagej_metadata.get('Info', None)
    searchstr = "[Acquisition Parameters Common] ImageCaputreDate ="
    date_str = None
    for line in info.splitlines():
        if searchstr in line:
            date_str = line.split("=")[-1].strip()
            date_str = date_str.replace('\'', '')
            break
    assert date_str is not None

    date = datetime.fromisoformat(date_str)

    return date

def getFmInterval(path):
    """Get the frame interval of a tiff file

    Args:
        path (str): path to tiff file

    Returns:
        float: frame interval
    """
    with tf.TiffFile(path) as tif:
        imagej_metadata = tif.imagej_metadata
    fm_interval = float(imagej_metadata.get("finterval"))
    return fm_interval

def plotMeanPlane(stack, col = 0, ncols = 4):
    """
    Plot the mean of each plane in a stack in a plot with ncol columns

    Arguments:
        stack = the imaging stack with dimensions:
            - plane, # of colors, # of pix in x, # of pix in y
        col = the color channel of interest
        ncol = the number of columns for the plot

    Returns:
        mean_fig = a figure with all of the mean planes
    """
    num_planes = stack.shape[1]
    num_plt_rows = math.ceil(num_planes/ncols)

    mean_stack = stack.mean(axis=0)

    mean_fig,axs = plt.subplots(nrows = num_plt_rows, ncols = ncols)

    for plane in range(num_planes):
        axs[math.floor(plane/ncols), plane%ncols].imshow(mean_stack[plane,col,:,:])
        axs[math.floor(plane/ncols), plane%ncols].set_axis_off()
        axs[math.floor(plane/ncols), plane%ncols].set_title('plane ' + str(plane))

    plt.show()

    return mean_fig

def stackToMIP(stack, slices):
    """
    Convert a stack to a series of maximum intensity projections (MIPs),
    where the slices to be consider in each projection are specified
    in slices

    Arguments:
        stack = the imaging stack with dimensions:
            - plane, # of colors, # of pix in x, # of pix in y
        slices = the layers that set the bounds of each MIP

    Returns:
        div_stack_MIP = the MIPs from the stack
    """

    ### Need to add error checking in case the slices fall outside of the stack dimensions
    num_vols = len(slices)-1
    div_stack_MIP = np.zeros((stack.shape[0], num_vols) + stack.shape[2:])

    for v in range(num_vols):
        div_stack_MIP[:,v,:,:,:] = stack[:,slices[v]:slices[v+1],:,:,:].max(axis=1)

    return div_stack_MIP

def getSlicesFromStack():
    """
    Specify which slices to consider in the stack

    Returns:
        slices = the slice boundaries
    """

    num_vols = int(input('How many volumes?'))

    slices = [0]*(num_vols+1)

    # Specify the volume boundaries
    slices[0] = int(input('first slice to consider?'))
    for vol in range(num_vols):
        slices[vol+1] = int(input('last slice in volume ' + str(vol+1) + '?'))

    return slices

def tifMotionCorrect(numRefImg, locRefImg, upsampleFactor, stack, sigma):
    """ Motion correct a tiff stack by using phase cross correlation
    numRefImg = the number of images to average for the reference image
    locRefImg = the initial position in the stack to use for the reference
    upsampleFactor = how much to upsample the image in order to shift the image by less than one pixel
    stack = the stack to be registered
    sigma = the sigma to use in Gaussian filtering
    """
    # Generate reference image
    refImg = np.mean(stack[locRefImg:locRefImg+numRefImg,:,:],axis=0)

    # Gaussian filter the reference image
    refImgFilt = gaussian_filter(refImg, sigma=sigma)

    # Create empty arrays to hold the registration metrics
    shift = np.zeros((2, stack.shape[0]))
    error = np.zeros(stack.shape[0])
    diffphase = np.zeros(stack.shape[0])

    # Create an empty array to hold the motion corrected stack
    stackMC = np.ones(stack.shape).astype('int16')

    # Correct each volume
    for i in range(stack.shape[0]):
        # Get the current image
        shifImg = stack[i,:,:]

        # Filter it
        shifImgFilt = gaussian_filter(shifImg, sigma=sigma)

        # Find the cross correlation between the reference image and the current image
        shift[:,i], error[i], diffphase[i] = phase_cross_correlation(refImgFilt, shifImgFilt,
                                                                     upsample_factor = upsampleFactor)

        # Shift the image in Fourier space
        offset_image = fourier_shift(np.fft.fftn(shifImg), shift[:,i])

        # Convert back and save the motion corrected image
        stackMC[i,:,:] = np.fft.ifftn(offset_image).real.astype('int16')

    return [shift, stackMC]

def divStackMIP(stack, col = 0):
    """
    Slice a stack into volumes and get the MIPs of those volumes

    Arguments:
        stack = the imaging stack with dimensions:
            - plane, # of colors, # of pix in x, # of pix in y
        col = the color channel to inspect

    Returns:
        slices = the volume boundaries
        div_stack_MIP = the divided, MIPed stack
    """
    # Plot the mean slices
    fig = plotMeanPlane(stack, col)

    # Specify the volume slices
    slices = getSlicesFromStack()

    # Calculate the MIPs for the stack
    div_stack_MIP = stackToMIP(stack, slices)

    return [slices, div_stack_MIP]

def motionCorrectSlicedStack(div_stack_MIP, num_ref_img = 100, upsample_factor = 20, sigma = 2):
    """
    Motion correct each volume MIP in a sliced stack

    Arguments:
        div_stack_MIP = the MIPs of a sliced stack
        num_ref_img, upsample_factor, sigma = see tifMotionCorrect

    Returns:
        corrected_stacks = the motion corrected stacks
    """
    loc_ref_img = round(div_stack_MIP.shape[0]/12)

    shift_dat = []
    corrected_stack_1 = np.ones(div_stack_MIP.shape[0:2] + div_stack_MIP.shape[3:]).astype('int16')

    for vol in range(div_stack_MIP.shape[1]):
        [shift_dat_now, corrected_stack_1[:,vol,:,:]] = tifMotionCorrect(num_ref_img, loc_ref_img, upsample_factor,
                                                                        div_stack_MIP[:,vol,0,:,:],sigma)
        shift_dat.append(shift_dat_now)

    if div_stack_MIP.shape[2] > 1:
        corrected_stack_2 = np.ones(div_stack_MIP.shape[0:2] + div_stack_MIP.shape[3:]).astype('int16')
        for vol in range(div_stack_MIP.shape[1]):
            for frame in range(0,shift_dat[vol].shape[1]):
                shif_img = np.squeeze(div_stack_MIP[frame,vol,1,:,:])

                # Shift the image in Fourier space
                offset_image = fourier_shift(np.fft.fftn(shif_img), shift_dat[vol][:,frame])

                # Convert back and save the motion corrected image
                corrected_stack_2[frame,vol,:,:] = np.fft.ifftn(offset_image).real.astype('uint16')

        corrected_stacks = np.concatenate((corrected_stack_1,corrected_stack_2),axis = 1)
    else:
        corrected_stacks = corrected_stack_1
    return corrected_stacks

def getROIs(stack, roiFN, oldROIs, oldType):
    """ Use napari to get ROIs from a stack, using a given ROI function
    """

    # Load the mean image in napari
    viewer = napari.Viewer()
    viewer.add_image(stack)
    if len(oldROIs) > 0:
        viewer.add_shapes(oldROIs, shape_type=oldType, name = 'Shapes')
    napari.run()

    # Use the ROIs that were drawn in napari to get image masks
    [napOut, allROIs, allMasks] = roiFN(viewer, stack)

    return [napOut, allROIs, allMasks]


def FfromROIsDiv(stack, all_masks):
    """ Calculate the raw fluorescence in each ROI in all ROIS on the given stack
    for a stack with multiple volumes
    """
    num_frames = stack.shape[0]

    # Initialie the array to hold the fluorescence data
    rawF = np.zeros((num_frames,len(all_masks)))

    # Step through each frame in the stack
    for fm in range(0,num_frames):
        fmNow = stack[fm,:,:,:]

        # Find the sum of the fluorescence in each ROI for the given frame
        for roi in range(0,len(all_masks)):
            vol = all_masks[all_masks['roi'] == roi]['layer'][0]
            rawF[fm,roi] = np.multiply(fmNow[vol,:,:], np.transpose(all_masks[all_masks['roi'] == roi]['mask'][0])).sum()

    return rawF

def FfromROIs(stack, allMasks, frameIdx=1, ch=0):
    """Calculate the raw fluorescence in each ROI in all ROIS on the given stack

    Args:
        stack (NDArray[float64]): Image stack.
        allMasks (NDArray): list of masks.
        frameIdx (int, optional): Index of stack shape that stores frames. Defaults to 1.
        ch (int, optional): Which channel to calculate florescence from. Defaults to 0.

    Returns:
        NDArray[float64]: ndarray raw florescence per frame and roi. shape = (# of frames, # of ROI's)
    """

    # Initialie the array to hold the fluorescence data
    rawF = np.zeros((stack.shape[frameIdx],len(allMasks)))

    # Step through each frame in the stack
    for fm in range(0,stack.shape[frameIdx]):
        fmNow = stack[0, fm, ch, :, :]

        # Find the sum of the fluorescence in each ROI for the given frame
        for r in range(0,len(allMasks)):
            rawF[fm,r] = np.multiply(fmNow, allMasks[r]).sum()

    return rawF

def DFoF(rawF):
    """ Calculate the DF/F given a raw fluorescence signal
    The baseline fluorescence is the mean of the lowest 10% of fluorescence signals
    """

    # Initialize the array to hold the DF/F data
    DF = np.zeros(rawF.shape)

    # Calculate the DF/F for each ROI
    for r in range(0,rawF.shape[1]):
        Fbaseline = np.sort(rawF[:,r])[0:round(0.1*rawF.shape[0])].mean()
        DF[:,r] = rawF[:,r]/Fbaseline-1

    return DF

def DFoFfromfirstfms(rawF, fm_interval, baseline_sec=10):
    """Calculate the DF/F given a raw fluorescence signal
    The baseline fluorescence is the mean of first 10 seconds of florescence

    Args:
        rawF (NDArray[float64]): ndarray of raw florescence over frame and roi.
                                 shape = (# of frames, # of ROI's)
        fm_interval (float): Time it takes to capture one frame (in seconds per frame)
        baseline_sec (float): How long into the trial to get the baseline from

    Returns:
        NDArray[float64]: ndarray of delta florescence over baseline florescence per frame and roi.
                          shape = (# of frames, # of ROI's)
    """

    # Initialize the array to hold the DF/F data
    DF = np.zeros(rawF.shape)

    # rawF axes: [frames, rois]
    baseline_sec = 10
    baseline_end_frame = round(baseline_sec / fm_interval)

    # Calculate the DF/F for each ROI
    for r in range(0, rawF.shape[1]):
        Fbaseline = rawF[0:baseline_end_frame, r].mean()
        DF[:, r] = rawF[:, r] / Fbaseline - 1

    return DF

def getRingROIs(viewer, stackMean):
    """ Get an ellipse and divide it into 16 sections
    """
    numROIs = 16
    angStep = 360/numROIs

    EBOutline = viewer.layers["Shapes"]

    ellipseCent = [int(np.mean([p[-2] for p in EBOutline.data[0]])),
              int(np.mean([p[-1] for p in EBOutline.data[0]]))]
    ellipseCentInt = (int(ellipseCent[0]),int(ellipseCent[1]))
    ellipseAx1 = np.sqrt((EBOutline.data[0][2][-1] - EBOutline.data[0][1][-1])**2 +
                        (EBOutline.data[0][2][0] - EBOutline.data[0][1][0])**2)
    ellipseAx2 = np.sqrt((EBOutline.data[0][0][-1] - EBOutline.data[0][1][-1])**2 +
                        (EBOutline.data[0][0][0] - EBOutline.data[0][1][0])**2)
    ellipseAng = 180/np.pi*np.arcsin((EBOutline.data[0][0][0] - EBOutline.data[0][1][0])/
                                    (EBOutline.data[0][0][-1] - EBOutline.data[0][1][-1]))

    rois = []
    allMasks = []
    for a in range(0,numROIs):
        mask = np.zeros((stackMean.shape[0], stackMean.shape[1]))
        pts = cv.ellipse2Poly(ellipseCentInt,
                              (int(0.5*ellipseAx1), int(0.5*ellipseAx2)),
                              int(ellipseAng),
                              int(angStep*(a-1)),int(angStep*a),
                              3)
        roiNow = np.append(pts, [np.array(ellipseCentInt)], axis=0)
        rois.append(roiNow)
        allMasks.append(cv.fillConvexPoly(mask,roiNow,1))

    return [EBOutline.data, rois, allMasks]

def getPolyROIs(viewer, stack):
    """ Make polygonal ROIs from a napari layer
    """

    # Get the ROIs from napari
    rois = viewer.layers['Shapes'].data

    # Initialize an array to hold the ROI masks
    allMasks = pd.DataFrame({'roi':int(), 'layer':int(), 'mask':[]})

    # Make the polygonal ROIs from the points
    for i,r in enumerate(rois):
        mask = np.zeros(stack.shape[1:3])
        mask = cv.fillConvexPoly(mask,np.array(r[:,1:3],dtype='int'),1)
        roiInfo = pd.DataFrame({'roi':i, 'layer':int(r[0,0]), 'mask':[mask]})
        allMasks = pd.concat([allMasks, roiInfo])

    return [rois, rois, allMasks]

def getEBROI(viewer, stack):
    """ Make polygonal ROIs from a napari layer
    """

    EBOutline = viewer.layers["Shapes"]

    ellipseCent = [int(np.mean([p[-2] for p in EBOutline.data[0]])),
              int(np.mean([p[-1] for p in EBOutline.data[0]]))]
    ellipseCentInt = (int(ellipseCent[0]),int(ellipseCent[1]))
    ellipseAx1 = np.sqrt((EBOutline.data[0][2][-1] - EBOutline.data[0][1][-1])**2 +
                        (EBOutline.data[0][2][0] - EBOutline.data[0][1][0])**2)
    ellipseAx2 = np.sqrt((EBOutline.data[0][0][-1] - EBOutline.data[0][1][-1])**2 +
                        (EBOutline.data[0][0][0] - EBOutline.data[0][1][0])**2)
    ellipseAng = 180/np.pi*np.arcsin((EBOutline.data[0][0][0] - EBOutline.data[0][1][0])/
                                    (EBOutline.data[0][0][-1] - EBOutline.data[0][1][-1]))

    pts = cv.ellipse2Poly(ellipseCentInt,
                          (int(0.5*ellipseAx1), int(0.5*ellipseAx2)),
                          int(ellipseAng),
                          0, 360,
                          3)

    # Initialize an array to hold the ROI masks
    mask = np.zeros((stack.shape[0], stack.shape[1]))
    allMasks = []
    allMasks.append(cv.fillConvexPoly(mask,pts,1))

    return [EBOutline.data, pts, allMasks]

def incr_bbox(bounding_box, scale_factor):
    """Scale a bounding box keeping it centered at the same spot

    Args:
        bounding_box (ndarray): Bounding box of shape (2,2): [x or y, min or max]
        scale_factor (float): Amount to scale each side of the bounding box by

    Returns:
        NDArray[float64]: scaled bounding box
    """
    view_box = np.empty(shape=(2, 2))
    for dim in range(2):
        for lim in range(2):
            if lim == 0:  # min
                sign = -1
            if lim == 1:  # max
                sign = 1
            length = bounding_box[dim, 1] - bounding_box[dim, 0]
            scale_amount = sign * (scale_factor - 1) / 2 * length
            view_box[dim, lim] = bounding_box[dim, lim] + scale_amount
    return view_box

def get_bbox(rois, scale_factor=1.5):
    """Given a list of rois, return a bounding box, a scale factor of 1 is a tight box

    Args:
        rois (list[ndarray]): List of rois, each roi is an ndarray of the points that make up the roi.
        scale_factor (float, optional): Amount to scale each side of the bounding box by. Defaults to 1.5.

    Returns:
        NDArray[float64]: Bounding box of shape (2,2): [x or y, min or max]
    """
    XCOL = 0
    YCOL = 1
    # roi_bound axes: [roi, x or y, min or max]
    roi_bounds = np.empty(shape=(len(rois), 2, 2))

    # Get min and max for each roi x and y
    for i, r in enumerate(rois):
        roi_bounds[i][0][0], roi_bounds[i][1][0] = r.min(axis=0)[XCOL : YCOL + 1]
        roi_bounds[i][0][1], roi_bounds[i][1][1] = r.max(axis=0)[XCOL : YCOL + 1]

    # Get the coords for the bounding box, using upper left corner to lower right
    # bounding_box axes: [x or y, min or max]
    bounding_box = np.empty(shape=(2, 2))
    bounding_box[:, 0] = roi_bounds[:, :, 0].min(axis=0)
    bounding_box[:, 1] = roi_bounds[:, :, 1].max(axis=0)

    # Create a larger bounding box to not cut off parts of the PB
    view_box = incr_bbox(bounding_box, scale_factor)
    return view_box

def plot_colorbar(fig, cbaraxes, F_plot, F_lims, cbarlabel):
    """Plot the colorbar for the given F_plot

    Args:
        fig (~matplotlib.figure.Figure): Figure to add colorbar to
        cbaraxes (list): shape of cbar in format [x, y, width, height]
        F_plot (AxesImage): Plot of florescence
        F_lims (list): min and max florescence values
        cbarlabel (str): label of colorbar
    """
    # Plot colorbar
    cbar_ax = fig.add_axes(cbaraxes)
    cbar = fig.colorbar(F_plot, cax=cbar_ax)
    if ((F_lims[0] > 0) and (F_lims[1] > 0)) or (F_lims[0] < 0) and (F_lims[1] < 0):
        # Doesn't pass through 0 (eg. raw florescence)
        ticks = [F_lims[0], F_lims[1]]
        cbar.set_label(cbarlabel, labelpad=-25)
    else:
        # Passes through 0 (eg. delta florescence)
        ticks = [F_lims[0], 0, F_lims[1]]
        cbar.set_label(cbarlabel, labelpad=-12)
    cbar.set_ticks(ticks)
    if (F_lims[0] > 10**2) or (F_lims[1] > 10**2):
        tick_labels = [f"{lim:g}" for lim in ticks]
    else:
        tick_labels = [f"{lim:.2g}" for lim in ticks]
    cbar.ax.set_yticklabels(tick_labels)


def plot_florescence(
        F,panel,cmap,aspect,roi_num,fm_interval,F_lims,norm=None,
        withcbar=False,fig=None,cbaraxes=None,cbarlabel=None,
        ):
    """Plot the florescence of F
    if with cbar is true, need to provide fig, and axes of cbar
    Note: Plotting fails if trial is too long, maximum length is 18 minutes

    Args:
        F (NDArray[float64]): ndarray of florescence, with shape (# of ROI's, # of frames)
        panel (Axes): Axes to draw plot in.
        cmap (str or Colormap): Colormap to use.
        aspect (float): Vertical to horizontal ratio of heatmap pixel, of form aspect:1.
        norm (TwoSlopeNorm): TwoSlopeNorm object with range of data.
        fm_interval (float): Time it takes to capture one frame (in seconds per frame).
        withcbar (bool, optional): Set to true to add a colorbar. Defaults to False.
        fig (~matplotlib.figure.Figure, optional): Figure to add colorbar to. Defaults to None.
        cbaraxes (list, optional): shape of cbar in format [x, y, width, height]. Defaults to None.
        cbarlabel (str, optional): label of colorbar. Defaults to None.
    """
    # Plot florescence
    F_plot = panel.imshow(
        F,
        cmap=cmap,
        interpolation="nearest",
        aspect=aspect,
        norm=norm
    )
    # panel.title.set_text(title)
    num_frames = F.shape[1]
    panel.set_xlabel("sec", labelpad=0)
    panel.set_xticks(
        [int(sec / fm_interval) 
         for sec in np.arange(0, num_frames * fm_interval, 5)],
        [f"{sec:.0f}" if sec % 2 == 0 else "" 
         for sec in np.arange(0, num_frames * fm_interval, 5)],
    )
    panel.set_ylabel("ROI", labelpad=-2)
    roi_num = F.shape[0]
    panel.set_yticks(
        [i for i in range(roi_num) if i % 2 == 0],
        [i + 1 for i in range(roi_num) if i % 2 == 0],
    )
    panel.invert_yaxis()
    # Plot colorbar
    if withcbar:
        plot_colorbar(fig, cbaraxes, F_plot, F_lims, cbarlabel)

def saveDFDat(fileNm, expt, expt_dat):
    """
    Save a dictionary of the processed data

    Arguments:
        fileNm = the file name
        expt = the name of the experiment
        expt_dat = the processed experimental data
    """
    allDat = dict()

    # Open the previously saved data
    if os.path.isfile(fileNm):
        infile = open(fileNm,'rb')
        allDat = pickle.load(infile)
        infile.close()

    # Add the new data
    allDat[expt] = expt_dat

    # Save the data
    with open(fileNm, 'wb') as outfile:
        pickle.dump(allDat, outfile)

def formatDate(year, month):
    """Formats year and month together into form YYYY_MM

    Args:
        year (int)
        month (int)

    Returns:
        str: formated year and month string
    """
    formatted_date = f"{year}_{month:02d}"
    return formatted_date

def getPicklePath(trialNm, folderNm):
    """Create path for pickle file. 
    Under the folder given, the pickle file is stored in a year and month folder (year_month)

    Args:
        folder (str): path to folder to store pickle file in
        trialNm (str): filename of trial

    Returns:
        str: path of pickle file
    """
    trial_date = getDate(trialNm)
    year_month = formatDate(trial_date.year, trial_date.month)
    dirPath = os.path.join(folderNm, year_month)

    timestamp = trial_date.strftime("%Y%m%d-%H%M")
    baseNm = timestamp + '_' + os.path.basename(trialNm).split('.')[0] + ".pickle"
    fullPath = Path(dirPath, baseNm)
    return fullPath

def loadProcData(proc_data_fn):
    """Return processed data

    Args:
        proc_data_fn (str): path to processed data

    Returns:
        dict: dictionary of trial data
    """
    assert os.path.isfile(proc_data_fn)
    with open(proc_data_fn, 'rb') as infile:
        data = pickle.load(infile)
    return data

def saveTrials(expt_dat, folderNm):
    # Given a dictionary of trials,
    # save each trial in a seperate pickle file
    for trialNm, trial in expt_dat.items():
        trial_date = getDate(trialNm)
        year_month = formatDate(trial_date.year, trial_date.month)
        dirPath = os.path.join(folderNm, year_month)
        os.makedirs(dirPath, exist_ok=True)

        timestamp = trial_date.strftime("%Y%m%d-%H%M")
        basename = timestamp + '_' + os.path.basename(trialNm).split('.')[0] + ".pickle"
        fullPath = os.path.join(dirPath, basename)
        with open(fullPath, 'wb') as outfile:
            pickle.dump(trial, outfile)
