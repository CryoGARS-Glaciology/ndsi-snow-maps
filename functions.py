#! /usr/bin/env python

import os
import geopandas as gpd
import ee
import geedim as gd
import numpy as np
import datetime
import math
import datetime
import wxee as wx
import time
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import xarray as xr

def convert_wgs_to_utm(lon: float, lat: float):
    """
    Return best UTM epsg-code based on WGS84 lat and lon coordinate pair

    Parameters
    ----------
    lon: float
        longitude coordinate
    lat: float
        latitude coordinate

    Returns
    ----------
    epsg_code: str
        optimal UTM zone, e.g. "32606"
    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
        return epsg_code
    epsg_code = '327' + utm_band
    return epsg_code

def plot_image_snow_map(im_xr, im_snow, aoi_utm, rgb_bands):
    # Define colormap for snow
    snow_color = '#2166ac'
    snow_cmap = matplotlib.colors.ListedColormap(['white', snow_color])
    # Set up figure
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    # RGB image
    ax[0].imshow(np.dstack([im_xr[rgb_bands[0]].data[0], im_xr[rgb_bands[1]].data[0], im_xr[rgb_bands[2]].data[0]]),
                extent=(np.min(im_xr.x)/1e3, np.max(im_xr.x)/1e3, 
                        np.min(im_xr.y)/1e3, np.max(im_xr.y)/1e3))
    ax[0].set_xlabel('Easting [km]')
    ax[0].set_ylabel('Northing [km]')
    # Snow image
    ax[1].imshow(im_snow.data[0], cmap=snow_cmap, clim=(0,1),
                extent=(np.min(im_snow.x)/1e3, np.max(im_snow.x)/1e3, 
                        np.min(im_snow.y)/1e3, np.max(im_snow.y)/1e3))
    ax[1].set_xlabel('Easting [km]')    
    # dummy points for legend
    xmin, xmax = ax[1].get_xlim()
    ymin, ymax = ax[1].get_ylim()
    ax[1].plot(0,0, 's', markersize=10, markerfacecolor='w', markeredgecolor='gray', linewidth=0.5, label='Snow-free')
    ax[1].plot(0,0, 's', markersize=10, markerfacecolor=snow_color, markeredgecolor='gray', linewidth=0.5, label='Snow')
    ax[1].set_xlim(xmin, xmax)
    ax[1].set_ylim(ymin, ymax)
    ax[1].legend(loc='best')
    # Plot AOI
    for axis in ax:
        axis.plot(np.divide(aoi_utm.geometry[0].exterior.coords.xy[0], 1e3),
                  np.divide(aoi_utm.geometry[0].exterior.coords.xy[1], 1e3), '-k')

    return fig, ax

def calculate_aoi_coverage(im_ee, aoi_ee):
    # Create binary image of masked (0) and unmasked (1) pixels
    unmaskedPixels = im_ee.mask().reduce(ee.Reducer.allNonZero()).selfMask() 
    # Calculate the area of unmasked pixels in the ROI
    pixel_area = ee.Image.pixelArea()
    unmaskedArea = unmaskedPixels.multiply(pixel_area).reduceRegion(
        reducer = ee.Reducer.sum(),
        geometry = aoi_ee,
        scale = 30,  
        maxPixels = 1e13
        ).get('all')
    # Calculate the total area of the ROI
    aoi_area = aoi_ee.area()
    # Calculate the percentage of the AOI covered by unmasked pixels
    percentage_unmasked = ee.Number(unmaskedArea).divide(aoi_area).multiply(100)
  
    return im_ee.set('percent_AOI_coverage', percentage_unmasked).copyProperties(im_ee)


def query_imagery_classify_snow(aoi_utm, dataset, start_date, end_date, start_month, end_month, 
                                percent_aoi_coverage, out_path, figures_path, site_name, ndsi_threshold=0.4):
    """
    Query GEE for imagery, filter, mosaic images captured same day, and classify snow using the Normalized Difference Snow Index (NDSI) threshold.

    Parameters
    ----------
    aoi_utm: geopandas.geodataframe.GeoDataFrame
        area of interest used for querying and clipping images in the CRS of the optimal UTM zone
    dataset: str
        imagery dataset to query. Options: "Sentinel-2_SR", "Sentinel-2_TOA", or "Landsat"
    start_date: str
        "YYYY-MM-DD" of the start date for image querying
    end_date: str
        "YYYY-MM-DD" of the end date for image querying
    start_month: int
        start of month range for image querying, inclusive
    end_month: int
        end of month range for image querying, inclusive
    percent_aoi_coverage: int
        minimum AOI coverage required to download image and classify snow, e.g. 50 = 50% unmasked pixels over the AOI
    out_path: str or Path
        where output snow maps will be saved
    figures_path: str or Path
        where output figures will be saved
    ndsi_threshold: float
        threshold to apply to the NDSI for identifying snow 

    Returns
    ----------
    None
    """

    # Make sure paths for outputs exist
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        print('Made directory for outputs:', out_path)
    if not os.path.exists(figures_path):
        os.mkdir(figures_path)
        print('Made directory for figures:', figures_path)

    # Reformat AOI for querying GEE
    crs_utm = f"EPSG:{aoi_utm.crs.to_epsg()}"
    aoi_latlon = aoi_utm.to_crs("EPSG:4326")
    aoi_ee = ee.Geometry.Polygon(list(zip(aoi_latlon.geometry[0].exterior.coords.xy[0], 
                                          aoi_latlon.geometry[0].exterior.coords.xy[1])))

    # Get unique days in image collection, define collection-specific characteristics
    print(f'Querying GEE for {dataset} images')
    def get_unique_image_days(col_name, start_date, end_date, start_month, end_month, mask, aoi_ee):
        im_col = (gd.MaskedCollection.from_name(col_name)
                  .search(start_date=start_date, end_date=end_date, mask=mask, region=aoi_ee))
        # Get list of image dates
        im_ids = list(im_col.properties)
        im_dts = [datetime.datetime.fromtimestamp(im_col.properties[im_id]['system:time_start']/1000) 
                  for im_id in im_ids]
        # Filter dates outside month range
        im_dts  = [dt for dt in im_dts if (dt.month >= start_month) & (dt.month <= end_month)]
        # Convert to list of unique day strings
        im_days = np.unique(np.array([np.datetime64(dt).astype('datetime64[D]') for dt in im_dts]))
        im_days = [str(day) for day in im_days]
        return im_days
    
    if dataset=='Sentinel-2_SR':
        col_name = 'COPERNICUS/S2_SR_HARMONIZED'
        ndsi_bands = ['B3', 'B11']
        rgb_bands = ['B4', 'B3', 'B2']
        resolution = 10 # m
        im_days = get_unique_image_days(col_name, start_date, end_date, start_month, end_month, 
                                        mask=True, aoi_ee=aoi_ee)
    elif dataset=='Sentinel-2_TOA':
        col_name = 'COPERNICUS/S2_HARMONIZED'
        ndsi_bands = ['B3', 'B11']
        rgb_bands = ['B4', 'B3', 'B2']
        resolution = 10 # m
        im_days = get_unique_image_days(col_name, start_date, end_date, start_month, end_month, 
                                        mask=True, aoi_ee=aoi_ee)
    elif 'Landsat' in dataset: 
        ndsi_bands = ['SR_B3', 'SR_B6']
        rgb_bands = ['SR_B4', 'SR_B3', 'SR_B2']
        resolution = 30 # m
        im_days_L8 = get_unique_image_days("LANDSAT/LC08/C02/T1_L2", start_date, end_date, start_month, end_month, 
                                           mask=True, aoi_ee=aoi_ee)
        im_days_L9 = get_unique_image_days("LANDSAT/LC09/C02/T1_L2", start_date, end_date, start_month, end_month, 
                                           mask=True, aoi_ee=aoi_ee)
        im_days = set(im_days_L8 + im_days_L9)
    else:
        print("Collection name not recognized, please choose 'Sentinel-2_SR', 'Sentinel-2_TOA', or 'Landsat'")
        return

    # Define function to apply image scale factors
    def apply_scale_factors(im, dataset):
        if 'Sentinel-2' in dataset:
            return ee.Image(im).divide(1e4).copyProperties(ee.Image(im))
        elif 'Landsat' in dataset:
            return ee.Image(im).multiply(0.0000275).add(-0.2).copyProperties(ee.Image(im))
    
    # Iterate over days
    print('Iterating over unique dates in collection')
    for day in tqdm(im_days):
        # Mosaic all images from that day
        if 'Sentinel-2' in dataset:
            im_day = (gd.MaskedCollection.from_name(col_name)
                      .search(start_date=day, end_date=str(np.datetime64(day) + np.timedelta64(1, 'D')),
                              region=aoi_ee, mask=True)).composite().ee_image
        elif 'Landsat' in dataset:
            im_day_L8 = (gd.MaskedCollection.from_name("LANDSAT/LC08/C02/T1_L2")
                         .search(start_date=day, end_date=str(np.datetime64(day) + np.timedelta64(1, 'D')),
                                 region=aoi_ee, mask=True))
            im_day_L9 = (gd.MaskedCollection.from_name("LANDSAT/LC09/C02/T1_L2")
                         .search(start_date=day, end_date=str(np.datetime64(day) + np.timedelta64(1, 'D')),
                                 region=aoi_ee, mask=True))
            # check which collection(s) have image results
            nL8, nL9 = len(list(im_day_L8.properties)), len(list(im_day_L9.properties))
            if (nL8 > 0) & (nL9 > 0):
                im_day = gd.MaskedCollection.from_list([im_day_L8, im_day_L9]).composite().ee_image
            elif nL8 > 0:
                im_day = im_day_L8.composite().ee_image
            elif nL9 > 0:
                im_day = im_day_L9.composite().ee_image

        # Select only the bands we need
        im_ee = im_day.select(list(set(ndsi_bands + rgb_bands))) 

        # Calculate percent coverage of the AOI
        im_ee = calculate_aoi_coverage(im_ee, aoi_ee)
        aoi_coverage = im_ee.get('percent_AOI_coverage').getInfo()
        if aoi_coverage < percent_aoi_coverage:
            print(f'Image only covers {np.round(aoi_coverage, 2)} % of the AOI, skipping')
            continue

        # Apply image scalar
        im_ee = apply_scale_factors(ee.Image(im_ee), dataset)
        dt = datetime.datetime.strptime(day, '%Y-%m-%d')
        im_ee = im_ee.set({'system:time_start': time.mktime(dt.timetuple()) * 1e3})

        # Convert ee.Image to xarray.Dataset
        im_xr = ee.Image(im_ee).wx.to_xarray(region=aoi_ee, scale=resolution, crs='EPSG:4326')

        # Reproject to optimal UTM zone
        im_xr = im_xr.rio.reproject(crs_utm)

        # Apply NDSI thresholding to identify snow
        im_xr['NDSI'] = (im_xr[ndsi_bands[0]] - im_xr[ndsi_bands[1]]) / (im_xr[ndsi_bands[0]] + im_xr[ndsi_bands[1]])
        im_snow = xr.where(im_xr['NDSI'] >= ndsi_threshold, 1, 0)

        # Prepare snow image for saving
        # set no data values to -9999 to save as int
        im_snow = xr.where(im_xr[rgb_bands[0]] == im_xr.attrs['_FillValue'], -9999, im_snow).astype(int)
        # assign attributes
        im_snow = im_snow.assign_attrs({'Description': 'Binary snow image, classified using NDSI thresholding.',
                                        'Classes': '0 = Snow-free, 1 = Snow-covered',
                                        '_FillValue': -9999,
                                        'Image source': dataset,
                                        'Image date': day,
                                        'NDSI threshold': ndsi_threshold
                                        })
        # make sure CRS is set
        im_snow = im_snow.rio.write_crs(crs_utm)

        # Save binary snow image to file
        im_snow_fn = os.path.join(out_path, f'{day}_{dataset}_{site_name}_snow_map.tif')
        im_snow.rio.to_raster(im_snow_fn)
        print('Snow image saved to file:', im_snow_fn)

        # Plot results
        fig, ax = plot_image_snow_map(im_xr, im_snow, aoi_utm, rgb_bands)
        fig.suptitle(f'{day}_{dataset}_{site_name}')
        # Save figure
        fig_fn = os.path.join(figures_path, os.path.basename(im_snow_fn).replace('.tif', '.png'))
        fig.savefig(fig_fn, dpi=300, bbox_inches='tight')
        print('Figure saved to file:', fig_fn)
        plt.close()

    return