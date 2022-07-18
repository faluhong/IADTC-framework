"""

Demo code of the IADTC framework to generate the global spatiotemporally seamless daily mean land surface temperature.

Major steps include:
(1) using the multi-type ATC model to reconstruct the under-cloud LST for each MODIS overpass time.
(2) using the linear interpolation to fill the NaN values in the overpass time series.
(3) estimating the daily mean land surface with the DTC model

Here the demo data ranging from 15 N to 53.5 N and from 73 E to 135.5 E in 2019 was provided.
This is the stand-alone version base on python 3.8, so you can direct run the code to get the results.
The running time for one day is around 3 min. For the whole year, it may take around 18 h.

The generated global spatiotemporally seamless daily mean land surface temperature product from 2003 to 2019
is publicly available at: https://doi.org/10.5281/zenodo.6287052

You may refer to the following papers for reference.
[1] Hong et al., (2022).
A global dataset of spatiotemporally seamless daily mean land surface temperatures: generation, validation, and analysis.
Earth System Science Data, 14(7), 3091-3113.
https://essd.copernicus.org/articles/14/3091/2022/

[2] Hong et al., (2021).
A simple yet robust framework to estimate accurate daily mean land surface temperature
from thermal observations of tandem polar orbiters.
Remote Sensing of Environment, 264, 112612.
https://www.sciencedirect.com/science/article/pii/S0034425721003321

If you have any questions, feel free to contact me at hongfalu@foxmail.com

"""

import time
import numpy as np
from osgeo import gdal, gdalconst, gdal_array
import os
from os.path import join
from scipy.optimize import curve_fit
import math
import warnings
warnings.filterwarnings("ignore")

LEN_DAY = 362.25

def data_read(path_data):
    """
    read the example data

    Args:
        path_data:
    Returns:
        land_boundary
        merra2_sat
        ori_lst_terra_day
        ori_lst_aqua_day
        ori_lst_terra_night
        ori_lst_aqua_night
        ori_time_terra_day
        ori_time_aqua_day
        ori_time_terra_night
        ori_time_aqua_night
    """

    merra2_sat = gdal_array.LoadFile(join(path_data, 'MERRA2_SAT.tif'))

    ori_lst_terra_day = gdal_array.LoadFile(join(path_data, 'Ori_LST_Terra_Day.tif'))
    ori_lst_aqua_day = gdal_array.LoadFile(join(path_data, 'Ori_LST_Aqua_Day.tif'))
    ori_lst_terra_night = gdal_array.LoadFile(join(path_data, 'Ori_LST_Terra_Night.tif'))
    ori_lst_aqua_night = gdal_array.LoadFile(join(path_data, 'Ori_LST_Aqua_Night.tif'))

    ori_time_terra_day = gdal_array.LoadFile(join(path_data, 'Ori_Time_Terra_Day.tif'))
    ori_time_aqua_day = gdal_array.LoadFile(join(path_data, 'Ori_Time_Aqua_Day.tif'))
    ori_time_terra_night = gdal_array.LoadFile(join(path_data, 'Ori_Time_Terra_Night.tif'))
    ori_time_aqua_night = gdal_array.LoadFile(join(path_data, 'Ori_Time_Aqua_Night.tif'))

    land_boundary = gdal_array.LoadFile(join(path_data, 'land_mask.tif'))

    return land_boundary, merra2_sat, ori_lst_terra_day, ori_lst_aqua_day, ori_lst_terra_night, ori_lst_aqua_night, \
           ori_time_terra_day, ori_time_aqua_day, ori_time_terra_night, ori_time_aqua_night


def time_interpolation(img_time):
    """
    fill the missing overpass time using the simple linear interpolation

    Args:
        img_time: array stored the original overpass time waiting for interpolation
    Returns:
        img_time_return: the seamless overpass time
    """

    img_time_return = np.zeros((np.shape(img_time)), dtype=np.float32)

    for i in range(0, np.shape(img_time)[1]):
        for j in range(0, np.shape(img_time)[2]):

            time_whole_year = img_time[:, i, j].copy()
            if np.isnan(time_whole_year).all():
                img_time_return[:, i, j] = np.nan
            else:
                ok = ~np.isnan(time_whole_year)
                xp = ok.ravel().nonzero()[0]
                fp = time_whole_year[~np.isnan(time_whole_year)]
                x = np.isnan(time_whole_year).ravel().nonzero()[0]
                time_whole_year[np.isnan(time_whole_year)] = np.interp(x, xp, fp)
                img_time_return[:, i, j] = time_whole_year

    return img_time_return


def atc_single_sinusoidal_original(day, *param):
    """
    Original naive ATC model with using the single sinusoidal function
    Args:
        day: day of year
        *param: parameters of the ATC model
    Returns:
        lst: LST estimated from the ATC model
    """

    mast, yast, theta = param[0], param[1], param[2]
    lst = mast + yast * np.sin(2 * np.pi * day / LEN_DAY + theta)
    return lst


def atc_single_sinusoidal_enhance(xdata, *param):
    """
    Enhanced ATC model with using the single sinusoidal function
    Args:
        xdata: the input data contains the day of year and delta temperature
        *param: parameters of the ATC model
    Returns:
        lst: LST estimated from the ATC model
    """

    day, delta_temperature = xdata
    omega = 2 * np.pi * day / LEN_DAY

    mast, YAST, theta, k = param[0], param[1], param[2], param[3]
    lst = mast + YAST * np.sin(omega + theta) + k * delta_temperature
    return lst


def atc_double_sinusoidal_original(day, *param):
    """
    Original naive ATC model with using the double sinusoidal function
    Args:
        day: day of year
        *param: parameters of the ATC model
    Returns:
        lst: LST estimated from the ATC model
    """

    mast, yast_1, theta_1, yast_2, theta_2 = param[0], param[1], param[2], param[3], param[4]
    lst = mast + yast_1 * np.sin(2 * np.pi * day / LEN_DAY + theta_1) \
          + yast_2 * np.sin(4 * np.pi * day / LEN_DAY + theta_2)
    return lst


def atc_double_sinusoidal_enhance(xdata, *param):
    """
    Enhanced ATC model with using the double sinusoidal function
    Args:
        xdata: the input data contains the day of year and delta temperature
        *param: parameters of the ATC model
    Returns:
        lst: LST estimated from the ATC model
    """

    day, delta_temperature = xdata
    omega = 2 * np.pi * day / LEN_DAY

    mast, yast_1, theta_1, yast_2, theta_2, k = param[0], param[1], param[2], param[3], param[4], param[5]
    lst = mast + yast_1 * np.sin(omega + theta_1) + yast_2 * np.sin(
        4 * np.pi * day / LEN_DAY + theta_2) + k * delta_temperature
    return lst


def run_atc_single_sinusoidal_version(ori_lst_year, sat_whole_year):
    """
    running the enhanced single-sinusoidal version of ATC model with the input LST and Surface Air Temperature (SAT)
    Args:
        ori_lst_year: 1-d array contains the annual LST time series
        sat_whole_year: 1-d array contains the annual SAT time series
    Returns:
        img_lst_return: annual LST reconstructed by ATC model
    """

    day_whole_year = np.arange(1, 1 + len(sat_whole_year))
    clear_sky_flag = ~np.isnan(ori_lst_year)

    try:
        p0_ori_atc = [np.nanmean(sat_whole_year),
                      np.nanmax(sat_whole_year) - np.nanmin(sat_whole_year), 1.5 * np.pi]

        popt_ori_atc, pcov = curve_fit(atc_single_sinusoidal_original, day_whole_year, sat_whole_year,
                                       p0_ori_atc)

        ori_atc_sat = atc_single_sinusoidal_original(day_whole_year, popt_ori_atc[0], popt_ori_atc[1],
                                                     popt_ori_atc[2])
        delta_temperature = sat_whole_year - ori_atc_sat

        xdata_mask = np.vstack([day_whole_year[clear_sky_flag], delta_temperature[clear_sky_flag]])
        Xdata = np.vstack([day_whole_year, delta_temperature])

        p0_enhance_atc = [np.nanmean(ori_lst_year), np.nanmax(ori_lst_year) - np.nanmin(ori_lst_year),
                          1.5 * np.pi, 1.5]

        popt_enhance, pcov = curve_fit(atc_single_sinusoidal_enhance, xdata_mask,
                                       ori_lst_year[clear_sky_flag], p0_enhance_atc)

        lst_enhance_atc = atc_single_sinusoidal_enhance(Xdata, popt_enhance[0], popt_enhance[1],
                                                        popt_enhance[2], popt_enhance[3])
        lst_return = lst_enhance_atc

    except Exception:
        lst_return = None

    return lst_return


def run_atc_double_sinusoidal_version(ori_lst_year, sat_whole_year):
    """
    running the enhanced double-sinusoidal version of ATC model with the input LST and Surface Air Temperature (SAT)
    Args:
        ori_lst_year: 1-d array contains the annual LST time series
        sat_whole_year: 1-d array contains the annual SAT time series
    Returns:
        img_lst_return: annual LST reconstructed by ATC model
    """

    day_whole_year = np.arange(1, 1 + len(sat_whole_year))
    clear_sky_flag = ~np.isnan(ori_lst_year)

    try:
        mast_initial, yast_1_initial, theta_1 = np.nanmean(sat_whole_year), \
                                                np.nanmax(sat_whole_year) - np.nanmin(sat_whole_year), 1.5 * np.pi
        popt_ori_atc = [mast_initial, yast_1_initial, theta_1, 0.5 * yast_1_initial, 0.5 * theta_1]

        popt_ori, pcov = curve_fit(atc_double_sinusoidal_original, day_whole_year, sat_whole_year,
                                   popt_ori_atc)

        ori_atc_sat = atc_double_sinusoidal_original(day_whole_year, popt_ori[0], popt_ori[1], popt_ori[2],
                                                     popt_ori[3], popt_ori[4])
        delta_temperature = sat_whole_year - ori_atc_sat

        xdata_mask = np.vstack([day_whole_year[clear_sky_flag], delta_temperature[clear_sky_flag]])
        Xdata = np.vstack([day_whole_year, delta_temperature])

        mast_initial, yast_1_initial, theta_1 = np.nanmean(ori_lst_year), \
                                                np.nanmax(ori_lst_year) - np.nanmin(ori_lst_year), 1.5 * np.pi
        p0_enhance_atc = [mast_initial, yast_1_initial, theta_1, 0.5 * yast_1_initial, 0.5 * theta_1, 1.5]

        popt_enhance, pcov = curve_fit(atc_double_sinusoidal_enhance, xdata_mask, ori_lst_year[clear_sky_flag],
                                       p0_enhance_atc)

        lst_enhance_atc = atc_double_sinusoidal_enhance(Xdata, popt_enhance[0], popt_enhance[1], popt_enhance[2],
                                                        popt_enhance[3], popt_enhance[4], popt_enhance[5])

        lst_return = lst_enhance_atc

    except Exception:
        lst_return = None

    return lst_return


def under_cloud_lst_reconstruction_atc(ori_lst, sat, img_latitude, land_boundary):
    """
    reconstruct the under-cloud LST with the multi-type ATC model
    Args:
        ori_lst: 3-d array contains the original clear-sky LST observations
        sat: 3-d array contains the surface air temperature provided by MERRA2 reanalysis data
        img_latitude: 2-d array contains the latitude for each pixel
        land_boundary: 2-d array contains the land and water mask, 1 denotes land and 0 denotes water
    Returns:
        img_lst_return: 3-d array contains the seamless LST reconstructed by ATC model
    """

    len_day, n_row, n_col = np.shape(ori_lst)
    img_lst_return = np.zeros((np.shape(ori_lst)), dtype=np.float32)

    for row in range(0, n_row):
        for col in range(0, n_col):

            if land_boundary[row, col] == 1:

                latitude = img_latitude[row, col]
                sat_whole_year = sat[:, row, col]
                ori_lst_year = ori_lst[:, row, col]

                if (latitude < 23.5) & (latitude > -23.5):
                    # print('double atc run at row: {} col: {}'.format(row, col))
                    interpolation_lst = run_atc_double_sinusoidal_version(ori_lst_year, sat_whole_year)

                elif (latitude > 66.5) | (latitude < -66.5):
                    # print('double atc run at row: {} col: {}'.format(row, col))
                    interpolation_lst = run_atc_double_sinusoidal_version(ori_lst_year, sat_whole_year)

                else:
                    # print('single atc run at row: {} col: {}'.format(row, col))
                    interpolation_lst = run_atc_single_sinusoidal_version(ori_lst_year, sat_whole_year)

                if interpolation_lst is None:
                    # print('atc model failed at row: {} col: {}'.format(row, col))
                    pass
                else:
                    img_lst_return[:, row, col] = interpolation_lst

    img_lst_return[img_lst_return == 0] = np.nan

    # the original clear-sky LST observations replace the ATC-model modelled 'clear-sky' LSTs
    img_lst_return[~np.isnan(ori_lst)] = ori_lst[~np.isnan(ori_lst)]

    return img_lst_return


def sunrise_sunset_time_calculate(delta, latitude):
    """
    calculate the sunrise and sunset time
    Args:
        delta: solar declination of the pixel
        latitude: latitude of the pixel
    Returns:
        sunrise_time:
        sunset_time:
    """

    flag = -math.tan(latitude / 180.0 * np.pi) * math.tan(delta / 180.0 * np.pi)

    # set to 0 or 24 when polar night or polar day phenomenon happens
    if flag >= 1:
        omega = 0
    elif flag <= -1:
        omega = 24
    else:
        omega = 2.0 / 15 * math.acos(flag) * 180.0 / np.pi

    if omega == 0:
        sunrise_time, sunset_time = 0, 0
    else:
        sunrise_time = 12 - omega / 2
        sunset_time = 12 + omega / 2

    return sunrise_time, sunset_time


def prepare_dtc_fitting_data(sunrise_time, lst_terra_day, lst_aqua_day,
                             lst_terra_night, lst_aqua_night,
                             time_terra_day, time_aqua_day,
                             time_terra_night, time_aqua_night):
    """
    prepare the DTC model fitting data
    Args:
        sunrise_time: sunrise time
        lst_terra_day: LST at the Terra Day overpass time
        lst_aqua_day: LST at the Aqua Day overpass time
        lst_terra_night: LST at the Terra Night overpass time
        lst_aqua_night: LST at the Aqua Night overpass time
        time_terra_day: Terra Day overpass time
        time_aqua_day: Aqua Day overpass time
        time_terra_night: Terra Night overpass time
        time_aqua_night: Aqua Night overpass time
    Returns:
        sunrise_time:
        sunset_time:
    """

    # if the overpass time of Terra Night observation is before the sunrise time,
    # move the Terra Night observation to the second day for DTC model fitting
    if time_terra_night < sunrise_time:
        time_terra_night = time_terra_night + 24

    # if the overpass time of Aqua Night observation is before the sunrise time,
    # move the Terra Night observation to the second day for DTC model fitting
    if time_aqua_night < sunrise_time:
        time_aqua_night = time_aqua_night + 24

    time_four = np.array([time_terra_day,
                          time_aqua_day,
                          time_terra_night,
                          time_aqua_night])

    time_arg = np.argsort(time_four)
    time_four = time_four[time_arg]

    lst_four = np.array([lst_terra_day,
                         lst_aqua_day,
                         lst_terra_night,
                         lst_aqua_night])
    lst_four = lst_four[time_arg]

    return time_four, lst_four


def got09_dT_001_model(time, *param):
    """
    the GOT09-dT-tao model

    Ref:
    [1] Göttsche, F. M., & Olesen, F. S. (2009).
    Modelling the effect of optical thickness on diurnal cycles of land surface temperature.
    Remote Sensing of Environment, 113(11), 2306-2316.
    https://www.sciencedirect.com/science/article/pii/S0034425709001850

    [2] Hong et al., (2018).
    Comprehensive assessment of four-parameter diurnal land surface temperature cycle models under clear-sky.
    ISPRS Journal of Photogrammetry and Remote Sensing, 142, 190-204.
    https://www.sciencedirect.com/science/article/pii/S0924271618301710

    Args:
        time:
        *param: parameters of GOT09-dT-tao model
    Returns:
        temperature: temperature estimated by GOT09-dT-tao model
    """

    T0, Ta, tm, ts = param[0], param[1], param[2], param[3]
    global latitude, delta

    theta = np.pi / 12 * (time - tm)
    theta_s = np.pi / 12 * (ts - tm)
    mask_LT = time < ts
    mask_GE = time >= ts

    transmittance = 0.01
    Re, H = 6371, 8.43

    cos_sza = math.sin(delta / 180 * np.pi) * math.sin(latitude / 180 * np.pi) + \
              math.cos(delta / 180 * np.pi) * math.cos(latitude / 180 * np.pi) * np.cos(theta)
    cos_sza_s = math.sin(delta / 180 * np.pi) * math.sin(latitude / 180 * np.pi) + math.cos(
        delta / 180 * np.pi) * math.cos(latitude / 180 * np.pi) * math.cos(theta_s)
    sin_sza_s = math.sqrt(1 - cos_sza_s * cos_sza_s)
    cos_sza_min = math.sin(delta / 180 * np.pi) * math.sin(latitude / 180 * np.pi) + math.cos(
        delta / 180 * np.pi) * math.cos(latitude / 180 * np.pi)

    m_val = -Re / H * cos_sza + np.sqrt(pow((Re / H * cos_sza), 2) + 2 * Re / H + 1)
    m_sza_s = -Re / H * cos_sza_s + math.sqrt(pow((Re / H * cos_sza_s), 2) + 2 * Re / H + 1)
    m_min = -Re / H * cos_sza_min + math.sqrt(pow((Re / H * cos_sza_min), 2) + 2 * Re / H + 1)

    sza_derive_s = math.cos(delta / 180 * np.pi) * math.cos(latitude / 180 * np.pi) \
                   * math.sin(theta_s) / math.sqrt(1 - cos_sza_s * cos_sza_s)
    m_derive_s = Re / H * sin_sza_s - pow(Re / H, 2) * \
                 cos_sza_s * sin_sza_s / math.sqrt(pow(Re / H * cos_sza_s, 2) + 2 * Re / H + 1)

    k1 = 12 / np.pi / sza_derive_s
    k2 = transmittance * cos_sza_s * m_derive_s
    k = k1 * cos_sza_s / (sin_sza_s + k2)

    temperature1 = T0 + Ta * cos_sza[mask_LT] * np.exp(transmittance * (m_min - m_val[mask_LT])) / cos_sza_min

    temp1 = math.exp(transmittance * (m_min - m_sza_s)) / cos_sza_min
    temp2 = np.exp(-12 / np.pi / k * (theta[mask_GE] - theta_s))
    temperature2 = T0 + Ta * cos_sza_s * temp1 * temp2

    temperature = np.concatenate((temperature1, temperature2))

    return temperature


def daily_mean_lst_calculate(doy, land_boundary, img_latitude,
                             atc_lst_terra_day_single_day, atc_lst_aqua_day_single_day,
                             atc_lst_terra_night_single_day, atc_lst_aqua_night_single_day,
                             interpolate_time_terra_day_single_day, interpolate_time_aqua_day_single_day,
                             interpolate_time_terra_night_single_day, interpolate_time_aqua_night_single_day):
    """
    daily mean LST calculation

    Args:
        doy: day of year
        land_boundary: land_boundary: 2-d array contains the land and water mask, 1 denotes land and 0 denotes water
        atc_lst_terra_day_single_day:
        atc_lst_terra_day_single_day:
        atc_lst_aqua_day_single_day:
        atc_lst_terra_night_single_day:
        atc_lst_aqua_night_single_day:
        interpolate_time_terra_day_single_day:
        interpolate_time_aqua_day_single_day:
        interpolate_time_terra_night_single_day:
        interpolate_time_aqua_night_single_day:
    Returns:
        daily_mean_lst: calculated daily mean LST
        scenario_flag: the scenario flag
    """

    n_row, n_col = np.shape(atc_lst_terra_day_single_day)

    global delta, latitude
    delta = 23.45 * np.sin(2 * np.pi / (LEN_DAY * 1.0) * (284 + doy))

    daily_mean_lst = np.zeros((n_row, n_col), dtype=np.float32)
    scenario_flag = daily_mean_lst.copy()

    for row in range(0, n_row):
        for col in range(0, n_col):

            if land_boundary[row, col] == 1:

                latitude = img_latitude[row, col]
                sunrise_time, sunset_time = sunrise_sunset_time_calculate(delta, latitude)

                time_four, lst_four = prepare_dtc_fitting_data(sunrise_time,
                                                               atc_lst_terra_day_single_day[row, col],
                                                               atc_lst_aqua_day_single_day[row, col],
                                                               atc_lst_terra_night_single_day[row, col],
                                                               atc_lst_aqua_night_single_day[row, col],
                                                               interpolate_time_terra_day_single_day[row, col],
                                                               interpolate_time_aqua_day_single_day[row, col],
                                                               interpolate_time_terra_night_single_day[row, col],
                                                               interpolate_time_aqua_night_single_day[row, col])

                dtr_four = np.nanmax(lst_four) - np.nanmin(lst_four)

                if dtr_four < 5:
                    # Scenario #1
                    # DTR_four < 5.0 K, daily mean LST is calculated as the mean of LSTs at four overpass times
                    daily_mean_lst[row, col] = np.nanmean(lst_four)
                    scenario_flag[row, col] = 1

                else:
                    p0 = [np.min(lst_four), np.max(lst_four) - np.min(lst_four), 13.0, sunset_time - 1]
                    p0_bounds = ([np.min(lst_four) - 50, 0, 5, 7],
                                 [np.min(lst_four) + 50,
                                  np.max(lst_four) - np.min(lst_four) + 50,
                                  23,
                                  np.max(time_four) + 5])
                    try:
                        popt, pcov = curve_fit(got09_dT_001_model, time_four, lst_four, p0, bounds=p0_bounds)
                    except Exception as e:
                        popt = np.nan

                    if ~np.isnan(popt).any():
                        time_all_day = np.arange(sunrise_time, sunrise_time + 24, 0.1)
                        lst_all_day = got09_dT_001_model(time_all_day, popt[0], popt[1], popt[2], popt[3])

                        drt_diurnal_lst = np.nanmax(lst_all_day) - np.nanmin(lst_all_day)
                        delta_dtr = np.abs(drt_diurnal_lst - dtr_four)

                        if delta_dtr >= 20:
                            # Scenario #3
                            # DTR_four >= 5.0 K and Delta_DTR >= 20.0 K,
                            # daily mean LST is calculated as the mean of LSTs at four overpass times
                            daily_mean_lst[row, col] = np.nanmean(lst_four)
                            scenario_flag[row, col] = 3

                            # print('scenario #3 at row: {} col:{}'.format(row, col))
                        else:
                            # Scenario #2
                            # DTR_four >= 5.0 K and Delta_DTR < 20.0 K,
                            # daily mean LST is calculated as the mean of LSTs at four overpass times
                            daily_mean_lst[row, col] = np.average(lst_all_day)
                            scenario_flag[row, col] = 2

                            # print('scenario #2 at row: {} col:{}'.format(row, col))

                    else:
                        # Scenario #3
                        # the fitting of DTC model failed
                        # daily mean LST is calculated as the mean of LSTs at four overpass times
                        daily_mean_lst[row, col] = np.nanmean(lst_four)
                        scenario_flag[row, col] = 3

    daily_mean_lst[daily_mean_lst == 0] = np.nan
    scenario_flag[scenario_flag == 0] = np.nan

    return daily_mean_lst, scenario_flag


def output_daily_mean_lst(doy, img_daily_mean_lst, output_path):
    """
    output the daily mean LST

    Args:
        doy: day of year
        img_daily_mean_lst: 2-d array contains the daily mean LST
        output_path:
    Returns:
    """

    src_geotrans = (73.0, 0.5, 0.0, 53.5, 0.0, -0.5)

    src_wkt = ('GEOGCS["WGS 84",DATUM["WGS_1984",'
               'SPHEROID["WGS 84",6378137,298.257223563,'
               'AUTHORITY["EPSG","7030"]], AUTHORITY["EPSG","6326"]],'
               'PRIMEM["Greenwich",0, AUTHORITY["EPSG","8901"]], '
               'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
               'AUTHORITY["EPSG","4326"]]')

    output_folder = join(output_path, 'results')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    output_name = join(output_folder, 'daily_mean_lst_{:03d}.tif'.format(doy))

    [nrows, ncols] = np.shape(img_daily_mean_lst)

    dst_out = gdal.GetDriverByName('GTiff').Create(output_name, ncols, nrows, 1, gdalconst.GDT_Float32)
    dst_out.SetGeoTransform(src_geotrans)
    dst_out.SetProjection(src_wkt)

    band = dst_out.GetRasterBand(1)
    band.WriteArray(img_daily_mean_lst)

    band.FlushCache()

    dst = None


def main():
    start_time = time.perf_counter()

    pwd = os.getcwd()
    path_data = join(pwd, 'data')

    land_boundary, merra2_sat, ori_lst_terra_day, ori_lst_aqua_day, ori_lst_terra_night, ori_lst_aqua_night, \
    ori_time_terra_day, ori_time_aqua_day, ori_time_terra_night, ori_time_aqua_night = data_read(path_data)

    interpolate_time_terra_day = time_interpolation(ori_time_terra_day)
    interpolate_time_aqua_day = time_interpolation(ori_time_aqua_day)
    interpolate_time_terra_night = time_interpolation(ori_time_terra_night)
    interpolate_time_aqua_night = time_interpolation(ori_time_aqua_night)

    len_day, n_row, n_col = np.shape(merra2_sat)
    img_latitude = np.ones((n_row, n_col)) * np.arange(53.5, 15, -0.5).reshape(-1, 1)
    img_longitude = (np.ones((n_col, n_row)) * np.arange(73, 135.5, 0.5).reshape(-1, 1)).T

    atc_lst_terra_day = under_cloud_lst_reconstruction_atc(ori_lst_terra_day, merra2_sat, img_latitude, land_boundary)
    atc_lst_aqua_day = under_cloud_lst_reconstruction_atc(ori_lst_aqua_day, merra2_sat, img_latitude, land_boundary)
    atc_lst_terra_night = under_cloud_lst_reconstruction_atc(ori_lst_terra_night, merra2_sat, img_latitude,
                                                             land_boundary)
    atc_lst_aqua_night = under_cloud_lst_reconstruction_atc(ori_lst_aqua_night, merra2_sat, img_latitude, land_boundary)

    daily_mean_lst = np.zeros(np.shape(merra2_sat), dtype=float)
    scenario_flag = daily_mean_lst.copy()

    # for doy in range(1, len_day + 1, 1):
    for doy in range(1, 2):
        print('calculate the daily mean lst on {:03d}'.format(doy))
        daily_mean_lst[doy - 1, :, :], \
        scenario_flag[doy - 1, :, :] = daily_mean_lst_calculate(doy, land_boundary, img_latitude,
                                                                atc_lst_terra_day[doy - 1, :, :],
                                                                atc_lst_aqua_day[doy - 1, :, :],
                                                                atc_lst_terra_night[doy - 1, :, :],
                                                                atc_lst_aqua_night[doy - 1, :, :],
                                                                interpolate_time_terra_day[doy - 1, :, :],
                                                                interpolate_time_aqua_day[doy - 1, :, :],
                                                                interpolate_time_terra_night[doy - 1, :, :],
                                                                interpolate_time_aqua_night[doy - 1, :, :],
                                                                )

        output_daily_mean_lst(doy, daily_mean_lst[doy - 1, :, :], pwd)

    end_time = time.perf_counter()
    print('running time is {}'.format(end_time - start_time))


if __name__ == "__main__":
    main()
