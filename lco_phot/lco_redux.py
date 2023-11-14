import os
import sys
import warnings
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib as mpl

from astropy import wcs
from astropy.io import fits
from astropy.io import ascii
from astropy.time import Time
from astropy.stats import mad_std
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval
from astropy.utils.exceptions import AstropyWarning
from astropy.coordinates import EarthLocation
from astropy.coordinates import SkyCoord
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from photutils import DAOStarFinder
from photutils import aperture_photometry
from photutils import CircularAperture
from photutils import CircularAnnulus

from utils import ps1_query, angle_sep
# from ps1_tools import ps1cone, angle_sep

"""
Python-based reduction of Las Cumbres Observatory (LCO)
imaging data from SINISTRO instruments.

Written by Zach Vanderbosch
Last Updated 11/13/2023
"""

# Ignore all warnings
# warnings.filterwarnings("ignore")
# Supress annoying Astropy Warning Messages
# warnings.simplefilter('ignore', category=AstropyWarning)

# #############################################################
# ##
# ##  Progress Bar Code. I got this code from Stack Overflow,
# ##  "Python to print out status bar and percentage"
# ##
# #############################################################

# ## Provide the interation counter (count=int)
# ## and the action being performed (action=string)
# def progress_bar(count,total,action):
#     sys.stdout.write('\r')
#     sys.stdout.write(action)
#     sys.stdout.write("[%-20s] %d%%  %d/%d" % ('='*int((count*20/total)),\
#                                               count*100/total,\
#                                               count,total))
#     sys.stdout.flush()
#     return

# def gauss2d_func(M,Z,A,ux,uy,sigx,sigy,theta):
#     x,y = M
#     a = (np.cos(theta)**2/(2*sigx**2)) + \
#         (np.sin(theta)**2/(2*sigy**2))
#     b = (np.sin(2*theta)/(2*sigx**2)) - \
#         (np.sin(2*theta)/(2*sigy**2))
#     c = (np.sin(theta)**2/(2*sigx**2)) + \
#         (np.cos(theta)**2/(2*sigy**2))
#     ta = -a*((x-ux)**2)
#     tb = -b*(x-ux)*(y-uy)
#     tc = -c*((y-uy)**2)
#     return Z + (A * np.exp(ta+tb+tc))

# def gauss2d_fitter(im,width):
#     dmin = min(im.flatten())
#     dmax = max(im.flatten())
#     par_init = [dmin,dmax-dmin,width/2,width/2,3.0,3.0,0.0]
#     y, x = np.indices(im.shape,dtype=float)
#     xydata = np.vstack((x.ravel(),y.ravel()))
#     try:
#         gfit,_ = curve_fit(gauss2d_func, xydata, im.ravel(), p0=par_init)
#     except: # For MaxIter and Optimization Errors
#         gfit = None
#     return gfit


# Function used to calcuate the centroid for each object
def get_centroid(data,xpix,ypix,f,progress=True):
    """
    data = image pixel data
    xpix = list of x-pixel coordinates for objects
    ypix = list of y-pixel coordinates for objects
    """

    Ncen = len(xpix)
    xcentroids = np.zeros(Ncen,dtype=float)
    ycentroids = np.zeros(Ncen,dtype=float)
    xstddevs = np.zeros(Ncen,dtype=float)
    ystddevs = np.zeros(Ncen,dtype=float)
    bw = 15 # Box width

    if progress:
        action1 = 'Calculating Centroids...'
        progress_bar(0,Ncen,action1)
    for k in range(Ncen):
        xmid,ymid = int(round(xpix[k])),int(round(ypix[k]))
        xlow,xupp = xmid-bw,xmid+bw
        ylow,yupp = ymid-bw,ymid+bw
        dx,dy = xupp-xlow,yupp-ylow
        data_range = data[ylow:yupp,xlow:xupp]
        gfit = gauss2d_fitter(data_range,bw*2)

        # Get Z-Scale Normalization vmin and vmax
        ZS = ZScaleInterval(nsamples=10000, contrast=0.15, max_reject=0.5, 
                        min_npixels=5, krej=2.5, max_iterations=5)
        vmin,vmax = ZS.get_limits(data_range)


        # Check that the fit converged and that the centroid
        # has not moved too far from its original location. 
        # If it has, just use the original location from DAOStarFinder
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
        if progress:
            progress_bar(k+1,Ncen,action1)

        # Save plot of target's centroid location
        if (Ncen > 1) & (k == 0) & (gfit is not None):
            figt = plt.figure('t')
            tx = figt.add_subplot(111)
            tx.imshow(data_range,vmin=vmin,vmax=vmax)
            tx.plot(xpix[k]-xlow,ypix[k]-ylow,ls='None',
                    marker='x',c='r',mew=2,ms=5,label='original')
            tx.plot(gfit[2],gfit[3],ls='None',
                    marker='+',c='b',mew=2,ms=6,label='re-calculated')
            tx.plot(xcentroids[k]-xlow,ycentroids[k]-ylow,ls='None',
                    marker='o',mfc='None',mec='k',mew=2,ms=10,label='adopted')
            tx.legend(fontsize=10)
            plt.savefig('{}_target_centroid.pdf'.format(f.split(".")[0]),bbox_inches='tight')
            plt.close()

    return xcentroids,ycentroids,xstddevs,ystddevs

#########################################################
#########################################################

def lco_redux(fits_name, query_constraints=None):

    # RA-Dec Coords of Target & Comps in ProEM Frame
    target = [132.693196,19.939404]
    #comp1 = [132.7007711,19.9413390] 
    comp1 = [132.6966642,19.9262534]


    # Pan-STARRS1 Magnitudes of Target
    PSmagg = 20.0764
    PSmagr = 19.2666
    PSmagi = 18.9730
    PSmagg_err = 0.0556
    PSmagr_err = 0.0063
    PSmagi_err = 0.0223

    # Get Data & Header Info for first image
    hdu0 = fits.open(fits_name)
    image0 = hdu0[1].data
    header0 = hdu0[1].header
    hdu0.close()

    # Get WCS and other info from the first image header
    w0 = wcs.WCS(header0)
    wpix0 = w0.wcs_world2pix(np.array([target]),1)
    site = header0['SITE']
    tele = header0['TELESCOP']
    date = header0['DATE-OBS']
    filt = header0['FILTER']
    texp = header0['EXPTIME']
    obra = header0['CAT-RA'].replace(":","")
    obdc = header0['CAT-DEC'].replace(":","")
    imcols = header0['NAXIS1']
    imrows = header0['NAXIS2']
    linlimit = header0['MAXLIN'] # Linearity/Saturation Limit
    timeobs = header0['DATE-OBS']
    platescale = header0['PIXSCALE']
    obj_name = 'SDSSJ{}{}'.format(obra[0:4],obdc[0:5])

    # Define Astropy coordinate object for the target
    if 'McDonald' in site:
        loc = EarthLocation.of_site('mcdonald')
    elif 'Haleakala' in site:
        loc = EarthLocation.of_site('haleakala')
    elif 'Cerro Tololo' in site:
        loc = EarthLocation.of_site('ctio')
    elif 'Siding Spring' in site:
        loc = EarthLocation.of_site('sso')
    elif 'SAAO' in site:
        loc = EarthLocation.of_site('SAAO')
    elif 'Tenerife' in site:
        loc = EarthLocation.from_geodetic(lon=20.301111*u.deg,
                                          lat=-16.510556*u.deg,
                                          height=2390.*u.m)
    tcoord = SkyCoord(target[0]*u.deg,target[1]*u.deg,frame="icrs")


    # Perform an Initial 2DGaussian Fit to Target for FWHM estimate
    targ_pixcoord = w0.wcs_world2pix([target],1)
    comp_pixcoord = w0.wcs_world2pix([comp1],1)
    _,_,xsig_init,ysig_init = get_centroid(image0,comp_pixcoord[:,0],
                                           comp_pixcoord[:,1],fits_name,progress=False)
    fwhm_init = abs(2.3548*(xsig_init[0] + ysig_init[0]) / 2.0)
    if fwhm_init > 15:
        fwhm_init = 6.0


    # Get Z-Scale Normalization vmin and vmax
    ZS = ZScaleInterval(nsamples=10000, contrast=0.15, max_reject=0.5, 
                        min_npixels=5, krej=2.5, max_iterations=5)
    vmin0,vmax0 = ZS.get_limits(image0)


    # Find sources in first frame with DAOfind and sort 
    # results by order of decreasing object brightness
    def get_object_idx(Tx,Ty,Ox,Oy,step):
        diff = [((Tx - x)**2 + (Ty - y)**2)**0.5 for x,y in zip(Ox[::step],Oy[::step])]
        if min(diff) < 1.0:
            match=True
        else:
            match=False
        idx = diff.index(min(diff))
        return idx,match

    # Create Mask Regions to block off Bright Stars
    mask = np.zeros_like(image0, dtype=bool)


    # Get sources with DAOfind and sort
    print('\nIdentifying Sources in LCO Image...')
    bkgmed = np.nanmedian(image0.flatten()) 
    bkgMAD = mad_std(image0)  
    # daofind0 = DAOStarFinder(fwhm=1.0*fwhm_init,
    #                          threshold=bkgmed+3.0*bkgMAD,
    #                          sharphi=0.6,
    #                          peakmax=0.9*linlimit)
    daofind0 = DAOStarFinder(fwhm=1.4*fwhm_init,
                             threshold=0.5*bkgmed,
                             sharphi=0.6,
                             peakmax=linlimit)  
    sources0 = daofind0(image0, mask=mask)
    if sources0 is None: # No sources found
        print('No Sources Detected!')
        return


    # Convert source table to pandas DataFrame and sort by peak flux
    df_sources0 = sources0.to_pandas()
    df_sources0 = df_sources0.sort_values('peak',ascending=False).reset_index(drop=True)

    # Detrmine which pixel coordinates to use for the target and
    # eliminate all comp stars within 100 pixels of the edge
    xsources = df_sources0['xcentroid'].values
    ysources = df_sources0['ycentroid'].values

    idx_step = 1
    target_idx,match = get_object_idx(wpix0[0][0],wpix0[0][1],xsources,ysources,idx_step)
    if match:
        target_xcoord = xsources[target_idx]
        target_ycoord = ysources[target_idx]
        comp_coords = [[x,y] for x,y in zip(xsources[::idx_step],ysources[::idx_step]) if 
                       (x > 100) & (x < imcols-100) & 
                       (y > 100) & (y < imrows-100) &
                       (x != target_xcoord)]
    else:
        target_xcoord = targ_pixcoord[0][0]
        target_ycoord = targ_pixcoord[0][1]
        comp_coords = [[x,y] for x,y in zip(xsources[::idx_step],ysources[::idx_step]) if 
                       (x > 100) & (x < imcols-100) & 
                       (y > 100) & (y < imrows-100)]

    # Place the target coordinates at the front of the list
    object_coords = [[target_xcoord,target_ycoord]] + comp_coords


    # Calculate the World Coordinates for each saved object
    Nobj = len(object_coords)
    print('{} Sources Identified'.format(Nobj))
    object_world_coords = w0.wcs_pix2world(object_coords,1)


    #########################################################
    ## Perform a Pan-STARRS Query which covers all detected
    ## sources in the image, or load a previously saved query. 
    ## NOTE: Performing a single query for each source is slow!

    # Create an Empty DataFrame
    columns = """objID,raMean,decMean,nDetections,ng,nr,ni,nz,ny,
            gMeanPSFMag,gMeanPSFMagErr,rMeanPSFMag,rMeanPSFMagErr,
            iMeanPSFMag,iMeanPSFMagErr,zMeanPSFMag,zMeanPSFMagErr,
            yMeanPSFMag,yMeanPSFMagErr,gMeanKronMag,gMeanKronMagErr,
            rMeanKronMag,rMeanKronMagErr,iMeanKronMag,iMeanKronMagErr,
            zMeanKronMag,zMeanKronMagErr,yMeanKronMag,yMeanKronMagErr""".split(',')
    columns = [x.strip() for x in columns]
    columns = [x for x in columns if x and not x.startswith('#')]
    tab = pd.DataFrame(columns=columns+["sep"])

    # First Check for a Previously Saved Query
    qpath = '/home/zachvanderbosch/data/object/ZTF/ztf_data/WDJ0850+1956/LCO/'
    #qpath = '/Users/zvander/data/object/SDSSJ0852+2130/LCO/'
    if os.path.isfile(qpath + 'ps_query.csv'):
        print('\nLoading Pan-STARRS Query...\n')
        ps_tab = pd.read_csv(qpath + 'ps_query.csv')
    else: # If no query, perform a new one 

        # Query Constraints
        if query_constraints is None:
            query_constraints = {
                'ng.gt':3,
                'nr.gt':3,
                'ni.gt':3,
                'nDetections.gt':11,
                'gMeanPSFMag.gt':14.0,
                'rMeanPSFMag.gt':14.0,
                'iMeanPSFMag.gt':14.0
            }

        # Define central RA/Dec of the image
        center = w0.wcs_pix2world([[float(imcols)/2,float(imrows)/2]],1)
        ra_center = center[0][0]
        dec_center = center[0][1]
        # Search Radius Defined to Reach From Center-to-Corner of Image
        #radius = (header0['PIXSCALE'] * (2.0**0.5) * float(header0['NAXIS1'])/2) / 3600.0
        #radius = 25.0/60.0
        radius = 18.0/60.0
        print('\nSearch Radius = {:.2f} arcmin'.format(radius*60.0))

        # Perform the Cone Search
        print('Performing the Pan-STARRS Query...')
        # results = ps1cone(ra_center,dec_center,radius,release='dr2',columns=columns,**constraints)
        results = ps1_query(
            ra_center,
            dec_center,
            radius,
            release='dr2',
            table='mean',
            columns=None,
            **constraints
        )
        print('Complete...\n')

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
        ps_tab.to_csv(qpath+'ps_query.csv',index=False)


    # Iterate Through Each Object and Find Matches
    action0 = 'Matching Objects to Pan-STARRS Photometry...' # Progress bar message
    progress_bar(0,len(object_world_coords),action0) # Initiate the progress bar
    object_coords_keep = []
    for i,c in enumerate(object_world_coords):

        # Get RA/Dec for the object
        ra = c[0]
        dec = c[1]

        # Get subset of ps_tab with objects within ~5 arcseconds
        # of the chosen coordinates for each loop
        ps_tab_subset = ps_tab[(ps_tab.raMean < ra+0.00139) & 
                               (ps_tab.raMean > ra-0.00139) &
                               (ps_tab.decMean < dec+0.00139) & 
                               (ps_tab.decMean > dec-0.00139)].copy(deep=True)

        # If No matches, skip the object
        if len(ps_tab_subset) == 0:
            progress_bar(i+1,len(object_world_coords),action0)
            continue


        # Find Separations Between Source and Panstarrs Objects
        all_seps = []
        for r,d in zip(ps_tab_subset.raMean.values, ps_tab_subset.decMean.values):
            sep = angle_sep(dec,d,ra,r)
            all_seps.append(sep)
        ps_tab_subset['sep'] = all_seps

        # If a Pan-Starrs object exists within 1.5 arcsec, keep it
        if min(all_seps) < 1.5:
            # First Check that the object is not a galaxy using PSF-Kron Magnitudes
            ps_tab_matched = ps_tab_subset[ps_tab_subset.sep == min(ps_tab_subset.sep)]
            star_gal_check = ps_tab_matched.iMeanPSFMag - ps_tab_matched.iMeanKronMag
            if star_gal_check.values[0] > 0.01:
                progress_bar(i+1,len(object_world_coords),action0)
                continue
            else:
                if len(ps_tab_matched) > 1:
                    continue
                else:
                    tab = tab.append(ps_tab_matched,ignore_index=True)
                    object_coords_keep.append(object_coords[i])

        # Update Progress Bar
        progress_bar(i+1,len(object_world_coords),action0)

    Nobj_keep = len(tab)
    print('\n{} out of {} Objects Matched\n'.format(Nobj_keep,Nobj))
    if Nobj_keep < 50: # Too few sources for calibration
        return


    # #########################################################
    # #########################################################
    # ## Perform the Aperture Photometry on Sources 
    # ## with Matched PS1 Photometry

    # Get pixel coordinates of objects and find centroids
    wpix = np.asarray(object_coords_keep)
    xc,yc,_,_ = get_centroid(image0,wpix[:,0],wpix[:,1],fits_name,progress=True)


    # fig = plt.figure('test',figsize=(8,8))
    # ax = fig.add_subplot(111)

    # ax.imshow(image0, vmin=vmin0,vmax=vmax0,cmap='Greys_r')
    # ax.plot(xsources,ysources,ls='None',marker='o',mfc='None',mec='r',ms=8)
    # ax.plot(xc,yc,ls='None',marker='s',mfc='None',mec='b',ms=8)
    # plt.show()

    # Calculate the FWHM for Aperture Selection
    # Define the Gaussian function for fitting the stellar profile
    def gaussian(x, A, sigma):
        return A*np.exp(-(x)**2/(2.*sigma**2))

    # Define function that returns the distance, in pixels, between two points:
    def distance(x1,y1,x2,y2):
        return np.sqrt((x1-x2)**2.+(y1-y2)**2.)

    fwhm_all = np.zeros(Nobj_keep)
    print('\n\nDetermining Aperture Size from FWHM...')
    for i,coord in enumerate(object_coords_keep):

        x0 = coord[0] # x center of star
        y0 = coord[1] # y center of star
        pdist = 15 # distance in pixels to examine
        dist = [] #empty list to hold pixel distances
        pval = [] #empty list to hold pixel values
        for x in np.arange(round(x0)-pdist,round(x0)+pdist):
            for y in np.arange(round(y0)-pdist,round(y0)+pdist):
                dist.append(distance(x,y,x0,y0))
                pval.append(image0[int(y),int(x)])

        p0=[10000.,1.] #initial guesses of height and sigma
        popt,_  = curve_fit(gaussian,dist,np.array(pval)-bkgmed,p0=p0,maxfev=10000)
        fwhm = np.abs(popt[-1])*2.3548*platescale
        fwhm_all[i] = fwhm

    # Remove objects from sample which have outlier FWHM values
    fwhm_all_med = np.median(fwhm_all)
    fwhm_all_madstd = mad_std(fwhm_all)
    sigma = 5.0
    fwhm_idx = (fwhm_all > fwhm_all_med - sigma*fwhm_all_madstd) & \
               (fwhm_all < fwhm_all_med + sigma*fwhm_all_madstd)

    # Calculate the physical bin size for FWHM histogram
    bin_num = 30
    fwhm_span = max(fwhm_all[fwhm_idx]) - min(fwhm_all[fwhm_idx])
    bw_fwhm = fwhm_span / bin_num

    # Estimate the Mode of the FWHM distribution with KDE
    fwhm_grid = np.linspace(0.01,5.0,5000)
    kde_fwhm = gaussian_kde(fwhm_all[fwhm_idx], 
                 bw_method=bw_fwhm/fwhm_all[fwhm_idx].std(ddof=1))
    kdevals_fwhm = kde_fwhm.evaluate(fwhm_grid)
    fwhm_mode = fwhm_grid[kdevals_fwhm == max(kdevals_fwhm)][0]

    # Redefine the FWHM Median & MAD-STD values with the "good" dataset
    fwhm_med = np.median(fwhm_all[fwhm_idx])
    fwhm_madstd = mad_std(fwhm_all[fwhm_idx])

    # Define Aperture with radius = FWHM (in pixels)
    ap_choice = fwhm_mode/platescale
    print('Aperture Radius (Pixels) = {:.2f}'.format(ap_choice))
    print('Aperture Radius (arcsec) = {:.2f}'.format(ap_choice*platescale))


    # Define the apertures and sky annuli and perform aperture photometry
    print('\nPerforming Aperture Photometry...')
    ap_phot = np.zeros(len(object_coords_keep))
    positions = [(x,y) for x,y in zip(xc,yc)]
    aperture = CircularAperture(positions, r=ap_choice)
    #annulus = CircularAnnulus(positions, r_in=32., r_out=48.)
    annulus = CircularAnnulus(positions, r_in=20., r_out=30.)
    #annulus = CircularAnnulus(positions, r_in=10., r_out=20.)
    phot_table = aperture_photometry(image0, [aperture,annulus])

    # Calculate the median sky annulus counts
    annulus_masks = annulus.to_mask(method='center')
    bkg_median = np.zeros(len(annulus_masks))
    for i,mask in enumerate(annulus_masks):
        annulus_data = mask.multiply(image0)
        try:
            annulus_data_1d = annulus_data[mask.data > 0]
        except:
            print(i)
            print(annulus_data)
        _,median,_ = sigma_clipped_stats(annulus_data_1d)
        bkg_median[i] = median


    #bkg_sum = phot_table['aperture_sum_1'] * (aperture.area/annulus.area)
    bkg_sum = bkg_median * aperture.area
    ap_phot[:] = phot_table['aperture_sum_0'] - bkg_sum
    ap_mags = 25.0 - 2.5*np.log10(ap_phot)

    print(' Median Background: {:.3f} c/pix'.format(bkg_median[0]))
    print('Average Background: {:.3f} c/pix'.format(phot_table['aperture_sum_1'][0]/annulus.area))

    # Calculate Photometry Errors & Signal-to-Noise
    readn = header0['RDNOISE'] # Read Noise
    darkn = 0.002 # Dark Noise (e-/pixel/s)
    phot_err = np.sqrt(ap_phot + bkg_sum + (0.002*texp+readn**2)*aperture.area)
    phot_sn = ap_phot/phot_err
    mag_err = 1.087/phot_sn

    # Remove NaN entries from photometry
    nan_idx = ~np.isnan(ap_mags)
    gidx = np.where(nan_idx & fwhm_idx) # Set of "good indices" to use

    # Get Values for Target
    fwhm_target = fwhm_all[0]
    imag_target = ap_mags[0]
    ierr_target = mag_err[0]
    sn_target = phot_sn[0]


    ##################################################
    ## Fit a Linear Model to the Mag v. Mag trend

    # Choose Relevant PS Magnitudes Based on LCO filter
    if filt == 'gp':
        ps_mags = tab.gMeanPSFMag.values
        ps_errs = tab.gMeanPSFMagErr.values
        ps_mags_opp = tab.rMeanPSFMag.values
        ps_errs_opp = tab.rMeanPSFMagErr.values
        color = ps_mags - ps_mags_opp # g-r colors of sample
        color_errs = (ps_errs**2 + ps_errs_opp**2)**0.5
    elif filt == 'rp':
        ps_mags = tab.rMeanPSFMag.values
        ps_errs = tab.rMeanPSFMagErr.values
        ps_mags_opp = tab.gMeanPSFMag.values
        ps_errs_opp = tab.gMeanPSFMagErr.values
        color = ps_mags_opp - ps_mags # g-r colors of sample
        color_errs = (ps_errs**2 + ps_errs_opp**2)**0.5
    elif filt == 'ip':
        ps_mags = tab.iMeanPSFMag.values
        ps_errs = tab.iMeanPSFMagErr.values
        ps_mags_opp = tab.rMeanPSFMag.values
        ps_errs_opp = tab.rMeanPSFMagErr.values
        color = ps_mags_opp - ps_mags # r-i colors of sample
        color_errs = (ps_errs**2 + ps_errs_opp**2)**0.5


    # Estimate the Mode of the Color Term with KDE
    bw_color = 0.04
    color_grid = np.linspace(0.1,1.5,5000)
    kde = gaussian_kde(color[gidx], bw_method=bw_color/color[gidx].std(ddof=1))
    kdevals = kde.evaluate(color_grid)
    mode_color = color_grid[kdevals == max(kdevals)][0]


    # This function fits a linear trend to
    # PS1 Magnitude versus Instrumental Magnitude 
    def poly(x,c):
        return x + c

    # This function fits a linear trend to the 
    # color difference with the free parameters being the 
    # magnitude zero point (z) and the color term (c).
    def poly2(color,z,c):
        return z + (c*color)

    # The Fitting Procedure with Sigma Clipping
    def polyfit(fit_x,fit_y,err_x,err_y,fit_option):
        # Number of rejection iterations to perform
        num_iter = 10
        for i in range(num_iter):
            # Calculate the average P2P scatter
            Nv = len(fit_x)
            next_values = np.append(fit_y[1:],fit_y[-2])
            pp_avg = (sum((fit_y-next_values)**2)/Nv)**(0.5)
            # Do a Linear Polyfit
            if fit_option == '1':
                pfit,cov = curve_fit(poly, fit_x, fit_y)#, sigma=err_y)
                residual = fit_y - poly(fit_x,*pfit)
            if fit_option == '2':
                pfit,cov = curve_fit(poly2, fit_x, fit_y, sigma=err_y)
                residual = fit_y - poly2(fit_x,*pfit)
            # Do sigma clipping
            good_idx = np.where(abs(residual) < 2.0*pp_avg)
            # Redefine the data to be fit
            fit_x = fit_x[good_idx]
            fit_y = fit_y[good_idx]
            err_x = err_x[good_idx]
            err_y = err_y[good_idx]

        return pfit,cov,fit_x,fit_y,err_x,err_y


    # Perform polynomial fit
    params,covar,goodx,goody,errx,erry = polyfit(ap_mags[gidx],ps_mags[gidx],
                                                 mag_err[gidx],ps_errs[gidx],'1')
    psap_diff = ps_mags[gidx]-ap_mags[gidx]
    psap_diff_err = np.sqrt(ps_errs[gidx]**2 + mag_err[gidx]**2)
    params2,covar2,goodx2,goody2,errx2,erry2 = polyfit(color[gidx],psap_diff,
                                                       color_errs[gidx],psap_diff_err,'2')

    # Calculate Magnitude of Target Using Best Fit Lines
    if (filt == 'gp') or (filt == 'rp'):
        PScolor = PSmagg - PSmagr
    elif filt == 'ip':
        PScolor = PSmagr - PSmagi
    psmag_target = poly(imag_target,*params)
    psmag_target2 = poly2(PScolor,*params2) + imag_target

    # Generate a Best fit Line (for plotting purposes later)
    fitx = np.linspace(min(ap_mags[gidx]),max(ap_mags[gidx]),1000)
    fitx2 = np.linspace(min([min(color[gidx]),PScolor]),
                        max(color[gidx]),1000)
    fity = poly(fitx,*params)
    fity2 = poly2(fitx2,*params2)

    # Calculate Magnitude Errors
    xerr = covar[0][0]
    psmagerr_target = np.sqrt(xerr**2 + ierr_target**2)
    Zerr = covar2[0][0]
    cerr = covar2[1][1]
    psmagerr_target2 = np.sqrt(Zerr**2 + 
                               ierr_target**2 + 
                               (-0.01*cerr)**2 + 
                               (params2[1]*PSmagg_err)**2 + 
                               (params2[1]*PSmagr_err)**2)


    # Convert Observation Time to MJD and BJD Formats
    # Calculate BJD & MJD times using Astropy Time & TimeDelta objects
    t = Time(timeobs, format='isot', scale='utc',location=loc)  # UTC Times
    ltt_bary = t.light_travel_time(tcoord)  # Light travel time to barycenter
    tmjd = t.mjd                            # MJD times
    tbjd = t.tdb.jd + ltt_bary.jd           # Barycentric times (rescaled UTC + LTT)


    # Print out some info
    print('\n           Target Values        ')
    print('------------------------------------')
    print(' Instrumental Mag = {:.2f} +- {:.2f}'.format(imag_target,ierr_target))
    print('  Pan-STARRS Mag1 = {:.2f} +- {:.2f}'.format(psmag_target,psmagerr_target))
    print('  Pan-STARRS Mag2 = {:.2f} +- {:.2f}'.format(psmag_target2,psmagerr_target2))
    print('             FWHM = {:.2f}"'.format(fwhm_all_med))
    print('              S/N = {:.2f}'.format(sn_target))
    print('           Filter = {}'.format(filt))
    print('             Texp = {:.3f}'.format(texp))
    print('          UT Date = {}'.format(timeobs[0:10]))
    print('              MJD = {:.7f}'.format(tmjd))
    print('              BJD = {:.7f}'.format(tbjd))
    print('        Mode(g-r) = {:.4f}'.format(mode_color))
    print('   Zero-Point Mag = {:.4f}'.format(params2[0]))
    print('       Color Term = {:.4f}'.format(params2[1]))


    # Save useful data into a csv file
    siteid = header0['SITEID']
    airmass = header0['AIRMASS']
    humidity = header0['WMSHUMID']
    temp = header0['WMSTEMP']*1.8 + 32.0  # Converted to Degree F
    wind = header0['WINDSPEE']/1.6 # Converted to mph instead of kph
    sky = header0['WMSSKYBR'] # Sky brightness in mag/arcsec^2
    moonfrac = header0['MOONFRAC']
    moondist = header0['MOONDIST']
    moonalt = header0['MOONALT']
    csv_dat = {'mjd':tmjd,'bjd':tbjd,'date':timeobs.split("T")[0],
               'time':timeobs.split("T")[1],'filter':filt,'texp':texp,
               'mag':psmag_target,'magerr':psmagerr_target,
               'Zp':params2[0],'Zp_err':Zerr,'cterm':params2[1],
               'cterm_err':cerr,'Nfit':len(goodx2),'sn':sn_target,
               'fwhm':fwhm_all_med,'aper_rad (pix)':ap_choice,
               'airmass':airmass,'site_id':siteid,
               'telescope':tele,'temp':temp,'humidity':humidity,
               'wind':wind,'sky':sky,'moonfrac':moonfrac,
               'moondist':moondist,'moonalt':moonalt}
    csv_out = pd.DataFrame(csv_dat,index=[0])
    csv_name = 'phot_file_{}.csv'.format(fits_name.split("/")[-1].split(".")[0])
    csv_out.to_csv(csv_name,index=False)


    print("\nFinished! \n")

    return












