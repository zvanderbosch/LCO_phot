import sys
import json
import requests
import numpy as np

from scipy.optimize import curve_fit
from astropy.io import ascii
from astropy.table import Table

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


def ps1_query(ra,dec,rad,save_result=False,save_dir="./",
    table="mean",release="dr1",format="csv",columns=None,**constraints):
    """
    Function to perform Pan-STARRS1 (PS1) cone search

    Parameters:
    -----------
    ra: float
        Right ascension in decimal degrees
    dec: float
        Declination in decimal degrees
    radius: float
        Search radius in arcseconds.
    ndetect: int
        Lower limit on number of PS1 detections
    save_result: bool
        Whether to save PS1 search results to file
    save_dir: str
        Director to save search results into.

    Returns:
    --------
    ps_tab: DataFrame
        Query results as a pandas DataFrame. 
        Returns None if query unsuccessful.
    """

    # Define columns to return from query
    if columns is None:
        columns = """objID,raMean,decMean,nDetections,ng,nr,ni,nz,ny,
                gMeanPSFMag,gMeanPSFMagErr,rMeanPSFMag,rMeanPSFMagErr,
                iMeanPSFMag,iMeanPSFMagErr,zMeanPSFMag,zMeanPSFMagErr,
                yMeanPSFMag,yMeanPSFMagErr,gMeanKronMag,gMeanKronMagErr,
                rMeanKronMag,rMeanKronMagErr,iMeanKronMag,iMeanKronMagErr,
                zMeanKronMag,zMeanKronMagErr,yMeanKronMag,yMeanKronMagErr""".split(',')
        columns = [x.strip() for x in columns]
        columns = [x for x in columns if x and not x.startswith('#')]

    # Perform the Cone Search
    results = ps1cone(ra,dec,radius/3600.,release='dr2',columns=columns,**constraints)
    if results == '':
        return None

    # Convert Results into an Astropy Table, improve formatting,
    # and then create a Pandas Table
    apy_tab = ascii.read(results)
    for f in 'grizy':
        col1 = f+'MeanPSFMag'
        col2 = f+'MeanPSFMagErr'
        try:
            apy_tab[col1].format = ".4f"
            apy_tab[col1][apy_tab[col1] == -999.0] = np.nan
        except KeyError:
            print("{} not found".format(col1))
        try:
            apy_tab[col2].format = ".4f"
            apy_tab[col2][apy_tab[col2] == -999.0] = np.nan
        except KeyError:
            print("{} not found".format(col2))
    ps_tab = apy_tab.to_pandas()  

    # Save the query for future use
    if save_results:
        ps_tab.to_csv(f'{save_dir}ps_query.csv',index=False)

    return ps_tab



def ps1cone(ra,dec,radius,table="mean",release="dr1",format="csv",columns=None,
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
           **kw):
    """Do a cone search of the PS1 catalog
    
    Parameters:
    -----------
    ra: float 
        (degrees) J2000 Right Ascension
    dec: float
        (degrees) J2000 Declination
    radius: float
        (degrees) Search radius (<= 0.5 degrees)
    table: str 
        mean, stack, or detection
    release: str
        dr1 or dr2
    format: str
        csv, votable, json
    columns: list
        List of column names to include (None means use defaults)
    baseurl: str
        Base URL for the request
    verbose: bool
        Whether to print info about request
    **kw: dict
        Other parameters (e.g., {'nDetections.min':2})

    Returns:
    --------
    search_result: str
        The raw text response from the ps1search function
    """
    
    data = kw.copy()
    data['ra'] = ra
    data['dec'] = dec
    data['radius'] = radius
    search_result = ps1search(
        table=table,release=release,format=format,columns=columns,
        baseurl=baseurl, verbose=verbose, **data
    )

    return search_result



def ps1search(table="mean",release="dr1",format="csv",columns=None,
    baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
    **kw):
    """
    Do a general search of the PS1 catalog (possibly without ra/dec/radius)
    
    Parameters
    -----------
    table: str 
        mean, stack, or detection
    release: str
        dr1 or dr2
    format: str
        csv, votable, json
    columns: list
        List of column names to include (None means use defaults)
    baseurl: str
        Base URL for the request
    verbose: bool
        Whether to print info about request
    **kw: dict
        Other parameters (e.g., {'nDetections.min':2}) Note this is required!

    Returns:
    --------
    query_result: str or dict
        The result of the query. Returns a string if format is csv 
        or votable, and a dict if format is JSON.

    """
    
    # Check for search params
    data = kw.copy()
    if not data:
        raise ValueError("You must specify some parameters for search")

    # Check table/release are valid
    checklegal(table,release)
    if format not in ("csv","votable","json"):
        raise ValueError("Bad value for format")


    url = "{baseurl}/{release}/{table}.{format}".format(**locals())
    if columns:
        # check that column values are legal
        # create a dictionary to speed this up
        dcols = {}
        for col in ps1metadata(table,release)['name']:
            dcols[col.lower()] = 1
        badcols = []
        for col in columns:
            if col.lower().strip() not in dcols:
                badcols.append(col)
        if badcols:
            raise ValueError('Some columns not found in table: {}'.format(', '.join(badcols)))
        # Specify a list of column values in the API
        data['columns'] = '[{}]'.format(','.join(columns))

    # Perform query
    r = requests.get(url, params=data, timeout=600)

    if verbose:
        print(r.url)
    r.raise_for_status()
    if format == "json":
        query_result = r.json()
    else:
        query_result = r.text

    return query_result


def checklegal(table,release):
    """
    Checks if a combination of table and release is acceptable.
    Raises a VelueError exception if there is problem.

    Parameters:
    -----------
    table: str
        mean, stack, or detection
    release: str 
        dr1 or dr2
    """
    
    releaselist = ("dr1", "dr2")
    if release not in ("dr1","dr2"):
        raise ValueError("Bad value for release (must be one of {})".format(', '.join(releaselist)))
    if release=="dr1":
        tablelist = ("mean", "stack")
    else:
        tablelist = ("mean", "stack", "detection")
    if table not in tablelist:
        raise ValueError("Bad value for table (for {} must be one of {})".format(release, ", ".join(tablelist)))


def ps1metadata(table="mean",release="dr1",
    baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"):
    """
    Return metadata for the specified PS1 catalog and table
    
    Parameters:
    -----------
    table: str
        mean, stack, or detection
    release: str 
        dr1 or dr2
    baseurl: str
        Base URL for the request
    
    Returns:
    --------
    tab: Table
        An astropy table with columns name, type, description
    """
    
    checklegal(table,release)
    url = "{baseurl}/{release}/{table}/metadata".format(**locals())
    r = requests.get(url)
    r.raise_for_status()
    v = r.json()
    # convert to astropy table
    tab = Table(
        rows=[(x['name'],x['type'],x['description']) for x in v],
        names=('name','type','description')
    )
    return tab