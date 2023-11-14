import sys
import numpy as np

from scipy.optimize import curve_fit

"""
Utility functions used by the LCO_redux script.

Author: Zach Vanderbosch (Caltech)
Last Updated: 2023-11-13
"""



def progress_bar(count,total,action):
    """
    Simple Progress bar. Code from:
    https://stackoverflow.com/questions/3002085

    Parameters:
    -----------
    count: int
        The current iteration number.
    total: int
        The total number of iterations.
    action: str
        A description of the action being performed.
    """

    sys.stdout.write('\r')
    sys.stdout.write(action)
    sys.stdout.write("[%-20s] %d%%  %d/%d" % ('='*int((count*20/total)),\
                                              count*100/total,\
                                              count,total))
    sys.stdout.flush()
    return

def gauss2d_func(M,Z,A,ux,uy,sigx,sigy,theta):
    """
    2D Gaussian Function

    Parameters:
    -----------
    M: array
        x and y positions
    Z: float
        Vertical offset
    A: float
        Amplitude
    ux,uy: float
        The x and y centroids
    sigx,sigy: float
        The x and y standard deviations
    theta: float
        The rotation angle

    Returns:
    --------
    gauss2d: array
        The 2D-Gaussian function
    """

    x,y = M
    a = (np.cos(theta)**2/(2*sigx**2)) + \
        (np.sin(theta)**2/(2*sigy**2))
    b = (np.sin(2*theta)/(2*sigx**2)) - \
        (np.sin(2*theta)/(2*sigy**2))
    c = (np.sin(theta)**2/(2*sigx**2)) + \
        (np.cos(theta)**2/(2*sigy**2))
    ta = -a*((x-ux)**2)
    tb = -b*(x-ux)*(y-uy)
    tc = -c*((y-uy)**2)
    gauss2d = Z + (A * np.exp(ta+tb+tc))
    
    return gauss2d


def gauss2d_fitter(im,width):
    """
    Function fitting 2D-Gaussian to image data.

    Parameters:
    -----------
    im: array
        2D image data array
    width: float
        Image stamp size in pixels to use for
        Gaussian fitting.

    Returns:
    --------
    gfit: list
        The best fit 2D-Gaussian parameters.
        Returns None if fit fails.
    """

    dmin = min(im.flatten())
    dmax = max(im.flatten())
    par_init = [dmin,dmax-dmin,width/2,width/2,3.0,3.0,0.0]
    y, x = np.indices(im.shape,dtype=float)
    xydata = np.vstack((x.ravel(),y.ravel()))
    try:
        gfit,_ = curve_fit(gauss2d_func, xydata, im.ravel(), p0=par_init)
    except: # For MaxIter and Optimization Errors
        gfit = None
    return gfit
