import sys
import json
import tqdm
import requests
import numpy as np

from scipy.optimize import curve_fit
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import EarthLocation

"""
Utility functions used by the LCO_redux script.

Author: Zach Vanderbosch (Caltech)
Last Updated: 2023-11-14
"""


def angle_sep(d1,d2,a1,a2):
    """
    Function to calculate angular separation between two 
    RA/Dec coordinates.

    Parameters:
    -----------
    d1,d2: float
        Declinations to compare (decimal degrees)
    a1,a2: float
        Right Ascensions to compare (decimal degrees)

    Returns:
    --------
    sep: float
        Separation in arcseconds
    """
    rad = np.radians(np.asarray([d1,d2,a1,a2]))
    asep = np.arccos(np.sin(rad[0])*np.sin(rad[1])+
                     np.cos(rad[0])*np.cos(rad[1])*
                     np.cos(rad[3]-rad[2]))
    sep = np.degrees(asep)*3600.0 # Arcseconds 
    return sep


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



def get_centroid(data,xpix,ypix,bw=15):
    """
    Function to calcuate the centroid of an object

    Parameters:
    -----------
    data: array
        Image pixel data
    xpix: list
        List of x-pixel coordinates for objects
    ypix: list 
        List of y-pixel coordinates for objects
    bw: int
        Image cutout size in pixels
    
    Returns:
    --------
    xcentroids: array
        X-centroids
    ycentroids: array
        Y-centroids
    xstddevs: array
        X-standard deviations
    ystddevs: array
        Y-standard deviations
    """

    Ncen = len(xpix)
    xcentroids = np.zeros(Ncen,dtype=float)
    ycentroids = np.zeros(Ncen,dtype=float)
    xstddevs = np.zeros(Ncen,dtype=float)
    ystddevs = np.zeros(Ncen,dtype=float)

    action = 'Calculating Centroids...'
    bar_fmt = "%s{l_bar}{bar:40}{r_bar}{bar:-40b}" %action
    with tqdm.tqdm(total=Ncen, bar_format=bar_fmt) as pbar:

        for k in range(Ncen):
            xmid,ymid = int(round(xpix[k])),int(round(ypix[k]))
            xlow,xupp = xmid-bw,xmid+bw
            ylow,yupp = ymid-bw,ymid+bw
            dx,dy = xupp-xlow,yupp-ylow
            data_range = data[ylow:yupp,xlow:xupp]
            gfit = gauss2d_fitter(data_range,bw*2)


            # Check that the fit converged and that the centroid
            # has not moved too far from its original location. 
            # If it has, just use the original provided location.
            if gfit is None:
                xcentroids[k] = xpix[k]
                ycentroids[k] = ypix[k]
                xstddevs[k] = np.nan
                ystddevs[k] = np.nan
            else:
                shift_dist = (np.abs(gfit[2] + xlow - xpix[k])**2 +
                              np.abs(gfit[3] + ylow - ypix[k])**2)**0.5
                if shift_dist > 5.0:
                    xcentroids[k] = xpix[k]
                    ycentroids[k] = ypix[k]
                else:
                    xcentroids[k] = gfit[2] + xlow
                    ycentroids[k] = gfit[3] + ylow
                xstddevs[k] = gfit[4]
                ystddevs[k] = gfit[5]

            pbar.update(1)

    return xcentroids,ycentroids,xstddevs,ystddevs


def poly_linear_fixed(x,z):
    """Linear polynomial with slope = 1
    """
    return x + z

def poly_linear(x,z,a):
    """ Linear polynomial with variable slope
    """
    return a*x + z


def polyfit(fit_x,fit_y,err_x,err_y,fit_option, num_iter=10, sigma_threshold=2.0):
    """
    Polynomial fitting routine with iterative sigma-clipping.

    Parameters:
    -----------
    fit_x: array
        x values to fit
    fit_y: array
        y values to fit
    err_x: array
        x value uncertainties
    err_y: array
        y value uncertainties
    fit_option: str
        '1' or '2', determines which function will be used
            '1' = poly_linear_fixed
            '2' = poly_linear
    num_iter: int
        Number of rejection iterations
    sigma_threshold: float
        Sigma clipping theshold in numbers of standard deviations

    Returns:
    --------
    pfit: list
        Best fit parameters
    cov: list
        Covariance matrix
    fit_x: array
        x values after sigma-clipping
    fit_y: array
        y values after sigma-clipping
    err_x: array
        x value uncertainties after sigma-clipping
    err_y: array
        y value uncertainties after sigma-clipping
    """

    for i in range(num_iter):

        # Calculate the average P2P scatter
        Nv = len(fit_x)
        next_values = np.append(fit_y[1:],fit_y[-2])
        pp_avg = (sum((fit_y-next_values)**2)/Nv)**(0.5)

        # Do a Linear fix with slope = 1
        if fit_option == '1':
            pfit,cov = curve_fit(poly_linear_fixed, fit_x, fit_y)#, sigma=err_y)
            residual = fit_y - poly_linear_fixed(fit_x,*pfit)
        if fit_option == '2':
            pfit,cov = curve_fit(poly_linear, fit_x, fit_y, sigma=err_y)
            residual = fit_y - poly_linear(fit_x,*pfit)

        # Do sigma clipping
        good_idx = np.where(abs(residual) < sigma_threshold*pp_avg)

        # Redefine the data to be fit
        fit_x = fit_x[good_idx]
        fit_y = fit_y[good_idx]
        err_x = err_x[good_idx]
        err_y = err_y[good_idx]

    return pfit,cov,fit_x,fit_y,err_x,err_y


def get_location(sitename):
    """
    Function that returns astropy EarthLocation based
    on given sitename from LCO image header.

    Parameters:
    -----------
    sitename: str
        The value of the SITE header keyword in LCO image
    
    Returns:
    --------
    loc: EarthLocation object
        EarthLocation corresponding to sitename
    """

    if 'McDonald' in sitename:
        loc = EarthLocation.of_site('mcdonald')
    elif 'Haleakala' in sitename:
        loc = EarthLocation.of_site('haleakala')
    elif 'Cerro Tololo' in sitename:
        loc = EarthLocation.of_site('ctio')
    elif 'Siding Spring' in sitename:
        loc = EarthLocation.of_site('sso')
    elif 'SAAO' in sitename:
        loc = EarthLocation.of_site('SAAO')
    elif 'Tenerife' in sitename:
        loc = EarthLocation.from_geodetic(
            lon=20.301111*u.deg,
            lat=-16.510556*u.deg,
            height=2390.*u.m)

    return loc


def ps1_query(ra, dec, radius=3.0, save_result=False, save_dir=None,
    save_name=None, table="mean", release="dr1", format="csv",
    columns=None, constraints=None):
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
    results = ps1cone(
        ra,dec,
        radius/3600.,
        release='dr2',
        columns=columns,
        constraints=constraints
    )
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
    if save_result:
        if save_dir is None:
            save_dir = "./"
        if save_name is None:
            save_name = "ps_query.csv"
        ps_tab.to_csv(f'{save_dir}{save_name}',index=False)

    return ps_tab



def ps1cone(ra,dec,radius,table="mean",release="dr1",format="csv",columns=None,
    baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
    constraints=None):
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
    
    if constraints is None:
        data = {}
    else:
        data = constraints.copy()
    data['ra'] = ra
    data['dec'] = dec
    data['radius'] = radius

    search_result = ps1search(
        table=table,release=release,format=format,columns=columns,
        baseurl=baseurl, verbose=verbose, data=data
    )

    return search_result



def ps1search(table="mean",release="dr1",format="csv",columns=None,
    baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
    data=None):
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
    data: dict
        Other parameters (e.g., {'nDetections.min':2}) Note this is required!

    Returns:
    --------
    query_result: str or dict
        The result of the query. Returns a string if format is csv 
        or votable, and a dict if format is JSON.

    """
    
    # Check for search params
    if data is None:
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