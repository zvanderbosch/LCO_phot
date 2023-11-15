import os
import sys
import tqdm
import warnings
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

from astropy import wcs
from astropy.io import fits
from astropy.io import ascii
from astropy.time import Time
from astropy.stats import mad_std
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval
from astropy.utils.exceptions import AstropyWarning
from astropy.coordinates import SkyCoord
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from photutils import DAOStarFinder
from photutils import aperture_photometry
from photutils import CircularAperture
from photutils import CircularAnnulus

# Load custom utility functions
from utils import ps1_query, angle_sep
from utils import get_centroid, gauss2d_fitter
from utils import get_location
from utils import polyfit, poly_linear, poly_linear_fixed

"""
Python-based reduction of Las Cumbres Observatory (LCO)
imaging data from SINISTRO instruments.

Written by Zach Vanderbosch
Last Updated 11/14/2023
"""

# Supress some annoying Astropy Warning Messages
warnings.simplefilter('ignore', category=AstropyWarning)

# The pixel origin used for WCS transformations
WCS_ORIGIN = 0



def lco_redux_target(
    fits_name, target_coord, comp_coord=None, source_match_limit=50,
    r_aperture=1.0, r_inner_annulus=5.0, r_outer_annulus=10.0,
    DAO_fwhm=1.0, DAO_bkgscale=3.0, DAO_sharp_high=0.6, DAO_peaklimit=1.0,
    target_query_constraints=None, target_query_save=False, target_query_dir=None, 
    target_query_name=None, target_query_format='csv',full_query_constraints=None, 
    full_query_save=False, full_query_dir=None, full_query_name=None, 
    full_query_format='csv', phot_save=False, phot_dir=None, phot_name=None, 
    plot_cutout=False, plot_save=False, plot_dir=None, plot_name=None, 
    verbose=False):
    """
    Aperture photometry routine returning a Pan-STARRS1 calibrated
    magnitude and uncertainty for a single target of interest.

    Parameters:
    -----------
    fits_name: str
        FITS filename
    target_coord: SKyCoord object
        coordinates of target object
    comp_coords: SkyCoord object
        Coordinates of comparison star. Only used for initial
        FWHM estimate. If None, target object used for FWHM.
    source_match_limit: int
        Minimum number of PS1-matched sources allowed 
        for good photometric calibration. 
    r_aperture: float
        Aperture radius, in number of FWHM
    r_inner_annulus: float
        Inner annulus radius, in number of FWHM
    r_outer_annulus: float
        Inner annulus radius, in number of FWHM
    DAO_fwhm: float
        DAOStarFinder fwhm scaling factor
    DAO_bkgscale: float
        DAOStarFinder background scaling factor
    DAO_sharp_high: float
        DAOStarFinder sharpness upper limit
    DAO_peaklimit: float
        DAOStarFinder peakmax scaling factor
    target_query_constraints: dict
        Dict containing PS1 query constraints for target query
        e.g. {'nDetections.gt': 10}
    target_query_save: bool 
        Whether to save target query results
    target_query_dir: str 
        Directory to save target query results in
    target_query_name: str
        Filename to save target query results as
    target_query_format: str
        Query return format: csv, votable, or json
    full_query_constraints: dict
        Dict containing PS1 query constraints for full FOV query
        e.g. {'nDetections.gt': 10}
    full_query_save: bool 
        Whether to save full FOV query results
    full_query_dir: str 
        Directory to save full FOV query results in
        If None, chooses current directory.
    full_query_name: str
        Filename to save full FOV query results as.
        If None, filename is ps_query.csv
    full_query_format: str
        Query return format: csv, votable, or json
    phot_save: bool
        Whether to save photometry results to file 
    phot_dir: str
        Directory to save photometry results in
        If None, chooses current directory.
    phot_name: str
        Filename to save photometry results as.
        If None, filename is <fits_name>_phot.csv
    plot_cutout: bool
        Whether to plot and show and image cutout
    plot_save: bool
        Whether to save image cutout to file 
    plot_dir: str
        Directory to save image cutout in
        If None, chooses current directory.
    plot_name: str
        Filename to save image cutout as.
        If None, filename is <fits_name>_cutout.png
    verbose: bool
        Whether to print photometry results to stdout at end.


    Returns:
    --------
    phot_dat: DataFrame
        Photometry results. Returns None if no sources
        detected inimage, or if number of PS1-matched 
        sources less than source_match_limit.
    """

    # RA-Dec Coords of Target & Comp
    target = [target_coord.ra.deg, target_coord.dec.deg]
    if comp_coord is not None:
        comp = [comp_coord.ra.deg, comp_coord.dec.deg]

    # Perform Pan-STARRS1 query to get source magnitudes
    print('Performing Target Pan-STARRS Query...',end='')
    ps1_target = ps1_query(
        target[0],
        target[1],
        radius=3.0, # arcsec
        release='dr2',
        table='mean',
        columns=None,
        format=target_query_format,
        constraints=target_query_constraints,
        save_result=target_query_save,
        save_dir=target_query_dir,
        save_name=target_query_name
    )
    print('Complete\n')


    # Get Pan-STARRS1 Magnitudes of Target
    if ps1_target is None:
        raise(f'Unable to Retrieve Pan-STARRS1 Magnitudes for Target at RA = {target[0]:.6f}, Dec = {target[1]:.6f}')
    else: 
        PSmagg = ps1_target.gMeanPSFMag.iloc[0]
        PSmagr = ps1_target.rMeanPSFMag.iloc[0]
        PSmagi = ps1_target.iMeanPSFMag.iloc[0]
        PSmagg_err = ps1_target.gMeanPSFMagErr.iloc[0]
        PSmagr_err = ps1_target.rMeanPSFMagErr.iloc[0]
        PSmagi_err = ps1_target.iMeanPSFMagErr.iloc[0]

    # Get Data & Header Info for image
    with fits.open(fits_name) as hdul:
        image = hdul[1].data
        header = hdul[1].header


    # Get WCS and other info from the image header
    w = wcs.WCS(header)
    site = header['SITE']
    tele = header['TELESCOP']
    date = header['DATE-OBS']
    filt = header['FILTER']
    texp = header['EXPTIME']
    obra = header['CAT-RA'].replace(":","")
    obdc = header['CAT-DEC'].replace(":","")
    imcols = header['NAXIS1']
    imrows = header['NAXIS2']
    linlimit = header['MAXLIN'] # Linearity/Saturation Limit
    timeobs = header['DATE-OBS']
    platescale = header['PIXSCALE']
    siteid = header['SITEID']
    airmass = header['AIRMASS']
    humidity = header['WMSHUMID']
    temp = header['WMSTEMP']*1.8 + 32.0  # Converted to Degree F
    wind = header['WINDSPEE']/1.6 # Converted to mph instead of kph
    sky = header['WMSSKYBR'] # Sky brightness in mag/arcsec^2
    moonfrac = header['MOONFRAC']
    moondist = header['MOONDIST']
    moonalt = header['MOONALT']

    # Get pixel coordinates of target and comp
    targ_pixcoord = w.wcs_world2pix([target],WCS_ORIGIN)
    if comp_coord is not None:
        comp_pixcoord = w.wcs_world2pix([comp],WCS_ORIGIN)

    # Perform an Initial 2DGaussian Fit for FWHM estimate
    _,_,xsig_init,ysig_init = get_centroid(
        image,
        comp_pixcoord[:,0],
        comp_pixcoord[:,1]
    )
    fwhm_init = abs(2.3548*(xsig_init[0] + ysig_init[0]) / 2.0)
    print(f'Initial FWHM Estimate: {fwhm_init:.2f} pixels ({fwhm_init*platescale:.2f} arcsec)')


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
    mask = np.zeros_like(image, dtype=bool)


    # Get sources with DAOfind and sort
    bkgmed = np.nanmedian(image.flatten()) 
    bkgMAD = mad_std(image) 
    bkgThreshold = bkgmed+DAO_bkgscale*bkgMAD
    print('\nBackground Statistics:')
    print('------------------------')
    print(f'   Median: {bkgmed:.2f}') 
    print(f'      MAD: {bkgMAD:.2f}') 
    print(f'Threshold: {bkgThreshold:.2f}')

    print('\nIdentifying Sources in LCO Image...')
    daofind0 = DAOStarFinder(
        fwhm=DAO_fwhm*fwhm_init,
        threshold=bkgThreshold,
        sharphi=DAO_sharp_high,
        peakmax=linlimit
    )  
    sources0 = daofind0(image, mask=mask)
    if sources0 is None: # No sources found
        print('No Sources Detected!')
        return None


    # Convert source table to pandas DataFrame and sort by peak flux
    df_sources0 = sources0.to_pandas()
    df_sources0 = df_sources0.sort_values('peak',ascending=False).reset_index(drop=True)

    # Detrmine which pixel coordinates to use for the target and
    # eliminate all comp stars within 100 pixels of the edge
    xsources = df_sources0['xcentroid'].values
    ysources = df_sources0['ycentroid'].values

    idx_step = 1
    target_idx,match = get_object_idx(targ_pixcoord[0][0],targ_pixcoord[0][1],xsources,ysources,idx_step)
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
    object_world_coords = w.wcs_pix2world(object_coords,WCS_ORIGIN)



    """ Perform Pan-STARRS Query, or Load Prior Results """


    # First Check for a Previously Saved Query
    if full_query_dir is None:
        full_query_dir = "./"
    if full_query_name is None:
        full_query_name = "ps_query.csv"
    if os.path.isfile(f'{full_query_dir}{full_query_name}'):
        print('\nLoading Pan-STARRS Query...\n')
        ps_tab = pd.read_csv(f'{full_query_dir}{full_query_name}')
    else: # If no query, perform a new one 

        # Query Constraints
        if full_query_constraints is None:
            full_query_constraints = {
                'ng.gt':3,
                'nr.gt':3,
                'ni.gt':3,
                'nDetections.gt':11,
                'gMeanPSFMag.gt':14.0,
                'rMeanPSFMag.gt':14.0,
                'iMeanPSFMag.gt':14.0
            }

        # Define central RA/Dec of the image
        center = w.wcs_pix2world([[float(imcols)/2,float(imrows)/2]],WCS_ORIGIN)
        ra_center = center[0][0]
        dec_center = center[0][1]
        radius = 18.0*60. # 18 arcmin radius converted to arcsec
        print('\nSearch Radius = {:.2f} arcmin'.format(radius/60))

        # Perform the Cone Search
        print('Performing the Pan-STARRS Query...',end='')
        ps_tab = ps1_query(
            ra_center,
            dec_center,
            radius=radius, # arcsec
            release='dr2',
            table='mean',
            columns=None,
            format=full_query_format,
            constraints=full_query_constraints,
            save_result=full_query_save,
            save_dir=full_query_dir,
            save_name=full_query_name
        )
        print('Complete\n')



    """ Match Detected Sources to Pan-STARRS Sources """


    # Set up the progress bar
    action0 = 'Matching Objects to PS1...'
    bar_fmt = "%s{l_bar}{bar:40}{r_bar}{bar:-40b}" %action0

    # Iterate Through Each Object and Find Matches
    Ncheck = len(object_world_coords)
    matched_entries = []
    object_coords_keep = []
    with tqdm.tqdm(total=Ncheck, bar_format=bar_fmt) as pbar:
        for i,c in enumerate(object_world_coords):

            # Get RA/Dec for the object
            ra = c[0]
            dec = c[1]

            # Get subset of ps_tab with objects within ~5 arcseconds
            # of the chosen coordinates for each loop
            ps_tab_subset = ps_tab[
                (ps_tab.raMean < ra+0.00139) & 
                (ps_tab.raMean > ra-0.00139) &
                (ps_tab.decMean < dec+0.00139) & 
                (ps_tab.decMean > dec-0.00139)
            ].copy(deep=True)

            # If No matches, skip the object
            if len(ps_tab_subset) == 0:
                pbar.update(1)
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
                    pbar.update(1)
                    continue
                else:
                    if len(ps_tab_matched) > 1:
                        pbar.update(1)
                        continue
                    else:
                        matched_entries.append(ps_tab_matched)
                        object_coords_keep.append(object_coords[i])

            # Update Progress Bar
            pbar.update(1)

    # Combine matched results into new DataFrame
    tab = pd.concat(matched_entries,ignore_index=True)
    Nobj_keep = len(tab)
    print('\n{} out of {} Objects Matched\n'.format(Nobj_keep,Nobj))
    if Nobj_keep < source_match_limit : # Too few sources for calibration
        print(f'Number of objects matched below source_match_limit = {source_match_limit}')
        return None



    """ Calculate Centroids for all Matched Sources """

    # Get pixel coordinates of objects and find centroids
    wpix = np.asarray(object_coords_keep)
    xc,yc,_,_ = get_centroid(image, wpix[:,0], wpix[:,1])



    """ Estimate Mode of FWHM Distribution for Matched Sources """


    def gaussian(x, A, sigma):
        """Simple Gaussian Function 
        """
        return A*np.exp(-(x)**2/(2.*sigma**2))

    def distance(x1,y1,x2,y2):
        """Calculate pixel distance between two sources
        """
        return np.sqrt((x1-x2)**2.+(y1-y2)**2.)


    # Calculate FWHM for all sources
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
                pval.append(image[int(y),int(x)])

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
    fwhm_grid = np.arange(0.01,10.0,0.01)
    kde_fwhm = gaussian_kde(fwhm_all[fwhm_idx], 
                 bw_method=bw_fwhm/fwhm_all[fwhm_idx].std(ddof=1))
    kdevals_fwhm = kde_fwhm.evaluate(fwhm_grid)
    fwhm_mode = fwhm_grid[kdevals_fwhm == max(kdevals_fwhm)][0]

    # Define Aperture with radius = FWHM (in pixels)
    ap_choice = r_aperture*fwhm_mode/platescale
    print('Aperture Radius (Pixels) = {:.2f}'.format(ap_choice))
    print('Aperture Radius (arcsec) = {:.2f}'.format(ap_choice*platescale))



    """ Plot Cutout Image """

    if plot_cutout or plot_save:

        # Get Z-Scale Normalization vmin and vmax
        ZS = ZScaleInterval(nsamples=10000, contrast=0.15, max_reject=0.5, 
                        min_npixels=5, krej=2.5, max_iterations=5)
        

        # Get image cutouts extents
        cutout_width = int(1.5 * r_outer_annulus * fwhm_mode/platescale)
        xmid,ymid = int(round(xc[0])),int(round(yc[0]))
        xlow,xupp = xmid-cutout_width,xmid+cutout_width
        ylow,yupp = ymid-cutout_width,ymid+cutout_width
        implot = image[ylow:yupp,xlow:xupp]
        vmin,vmax = ZS.get_limits(implot)

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111,projection=w)

        ax.imshow(implot, vmin=vmin, vmax=vmax, cmap='Greys_r', origin='lower',
            extent=(xlow-1,xlow+2*cutout_width,
                    ylow-1,ylow+2*cutout_width))

        c1 = mpatches.Circle(
            (xc[0],yc[0]), 
            r_aperture*fwhm_mode/platescale, 
            ec="c", fc='None',lw=2.5)
        c2 = mpatches.Circle(
            (xc[0],yc[0]), 
            r_inner_annulus*fwhm_mode/platescale, 
            ec="w", fc='None',lw=4, ls='-',alpha=0.5)
        c3 = mpatches.Circle(
            (xc[0],yc[0]), 
            r_outer_annulus*fwhm_mode/platescale, 
            ec="w", fc='None',lw=4, ls='-',alpha=0.5)
        c4 = mpatches.Circle(
            (xc[0],yc[0]), 
            r_inner_annulus*fwhm_mode/platescale, 
            ec="m", fc='None',lw=2, ls='--')
        c5 = mpatches.Circle(
            (xc[0],yc[0]), 
            r_outer_annulus*fwhm_mode/platescale, 
            ec="m", fc='None',lw=2, ls='--')
        ax.add_patch(c1)
        ax.add_patch(c2)
        ax.add_patch(c3)
        ax.add_patch(c5)
        ax.add_patch(c4)

        # Add FWHM text
        fig_text = f"FWHM = {fwhm_mode:.2f}$''$"
        ax.text(0.05,0.92,fig_text, fontsize=16,ha='left', c='r',fontweight='heavy',
            transform=ax.transAxes, path_effects=[pe.withStroke(linewidth=4, foreground="silver")])

 
        ax.set_xlabel('R.A.',fontsize=14,color='silver')
        ax.set_ylabel('Declination',fontsize=14,color='silver')
        ax.tick_params(which='both', color='silver', labelcolor='silver')
        for spine in ax.spines.values():
            spine.set_edgecolor('silver')
        fig.set_facecolor('k')

        # Save results
        if plot_save:
            if plot_dir is None:
                plot_dir = "./"
            if plot_name is None:
                plot_name = '{}_cutout.png'.format(fits_name.split("/")[-1].split(".")[0])

            plot_name = f'{plot_dir}{plot_name}'
            plt.savefig(plot_name,dpi=200,bbox_inches='tight')

        plt.show()



    """ Perform the Aperture Photometry on Matched Sources """


    # Define the apertures and sky annuli and perform aperture photometry
    print('\nPerforming Aperture Photometry...')
    ap_phot = np.zeros(len(object_coords_keep))
    positions = [(x,y) for x,y in zip(xc,yc)]
    aperture = CircularAperture(positions, r=ap_choice)
    annulus = CircularAnnulus(
        positions, 
        r_in=r_inner_annulus*fwhm_mode/platescale, 
        r_out=r_outer_annulus*fwhm_mode/platescale
    )
    phot_table = aperture_photometry(image, [aperture,annulus])

    # Calculate the median sky annulus counts
    annulus_masks = annulus.to_mask(method='center')
    bkg_median = np.zeros(len(annulus_masks))
    for i,mask in enumerate(annulus_masks):
        annulus_data = mask.multiply(image)
        try:
            annulus_data_1d = annulus_data[mask.data > 0]
        except:
            print(i)
            print(annulus_data)
        _,median,_ = sigma_clipped_stats(annulus_data_1d)
        bkg_median[i] = median

    bkg_sum = bkg_median * aperture.area
    ap_phot[:] = phot_table['aperture_sum_0'] - bkg_sum
    ap_mags = 25.0 - 2.5*np.log10(ap_phot)

    # Calculate Photometry Errors & Signal-to-Noise
    readn = header['RDNOISE'] # Read Noise
    darkn = 0.002 # Dark Noise (e-/pixel/s)
    phot_err = np.sqrt(ap_phot + bkg_sum + (darkn*texp+readn**2)*aperture.area)
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


    """ Fit a Linear Model to the Mag-Diff vs. PS1-Color trend """

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
    bw_color = 0.04 # bin width
    color_grid = np.linspace(0.1,1.5,5000)
    kde = gaussian_kde(color[gidx], bw_method=bw_color/color[gidx].std(ddof=1))
    kdevals = kde.evaluate(color_grid)
    mode_color = color_grid[kdevals == max(kdevals)][0]


    # Perform polynomial fit to PS-color versus magnitude difference
    psap_diff = ps_mags[gidx]-ap_mags[gidx]
    psap_diff_err = np.sqrt(ps_errs[gidx]**2 + mag_err[gidx]**2)

    params,covar,goodx,_,_,_ = polyfit(
        color[gidx],
        psap_diff,
        color_errs[gidx],
        psap_diff_err,
        '2' # Uses poly_linear function
    )

    # Calculate Magnitude of Target Using Best Fit Lines
    if (filt == 'gp') or (filt == 'rp'):
        PScolor = PSmagg - PSmagr
        PScolorerr = np.sqrt(PSmagg_err**2 + PSmagr_err**2)
    elif filt == 'ip':
        PScolor = PSmagr - PSmagi
        PScolorerr = np.sqrt(PSmagr_err**2 + PSmagi_err**2)
    psmag_target = poly_linear(PScolor,*params) + imag_target


    # Calculate Magnitude Uncertainty
    color_coeff = params[1]
    Zerr = covar[0][0]
    cerr = covar[1][1]
    psmagerr_target = np.sqrt(
        Zerr**2 + 
        ierr_target**2 + 
        (PScolor*cerr)**2 + 
        (PScolorerr*color_coeff)**2
    )


    """ Apply Barycentric Time Corrections """

    loc = get_location(site)
    t = Time(timeobs, format='isot', scale='utc',location=loc)  # UTC Times
    ltt_bary = t.light_travel_time(target_coord)  # Light travel time to barycenter
    tmjd = t.mjd                            # MJD times
    tbjd = t.tdb.jd + ltt_bary.jd           # Barycentric times (rescaled UTC + LTT)


    """ Print out the Results """

    if verbose:
        print('\n           Target Values        ')
        print('------------------------------------')
        print('  Instrumental Mag = {:.2f} +- {:.2f}'.format(imag_target,ierr_target))
        print('PS1-Calibrated Mag = {:.2f} +- {:.2f}'.format(psmag_target,psmagerr_target))
        print('              FWHM = {:.2f}"'.format(fwhm_all_med))
        print('               S/N = {:.2f}'.format(sn_target))
        print('            Filter = {}'.format(filt))
        print('              Texp = {:.3f}'.format(texp))
        print('           UT Date = {}'.format(timeobs[0:10]))
        print('               MJD = {:.7f}'.format(tmjd))
        print('               BJD = {:.7f}'.format(tbjd))
        print('         Mode(g-r) = {:.4f}'.format(mode_color))
        print('    Zero-Point Mag = {:.4f}'.format(params[0]))
        print('        Color Term = {:.4f}'.format(params[1]))



    """ Combine Results and Metadata in DataFrame and Save to File """


    phot_dat = {
        'mjd':tmjd,'bjd':tbjd,'date':timeobs.split("T")[0],
        'time':timeobs.split("T")[1],'filter':filt,'texp':texp,
        'mag':psmag_target,'magerr':psmagerr_target,
        'Zp':params[0],'Zp_err':Zerr,'cterm':params[1],
        'cterm_err':cerr,'Nfit':len(goodx),'sn':sn_target,
        'fwhm':fwhm_mode,'aper_rad (pix)':ap_choice,
        'airmass':airmass,'site_id':siteid,
        'telescope':tele,'temp':temp,'humidity':humidity,
        'wind':wind,'sky':sky,'moonfrac':moonfrac,
        'moondist':moondist,'moonalt':moonalt
    }
    phot_dat = pd.DataFrame(phot_dat,index=[0])


    # Save results
    if phot_save:
        if phot_dir is None:
            phot_dir = "./"
        if phot_name is None:
            phot_name = '{}_phot.csv'.format(fits_name.split("/")[-1].split(".")[0])

        phot_name = f'{phot_dir}{phot_name}'
        phot_dat.to_csv(phot_name,index=False)

    print("\nFinished! \n")

    return phot_dat

