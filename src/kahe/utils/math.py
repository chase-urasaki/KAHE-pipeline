import numpy as np
import scipy.interpolate
import scipy.signal
from scipy.optimize import curve_fit
import astropy.stats
import matplotlib.pyplot as plt
import time
from NIRSPEC_CONSTANTS import OVERSCAN_WIDTH
import configparser
import os

def subtract_background(image, edge_traces):
    image = np.copy(image)
    for o in range(len(edge_traces)):
        backgrounds = []
        top_edges, bottom_edges = edge_traces[o]
        for c in range(image.shape[1]):
            background = np.ma.median(image[int(top_edges[c]) : int(bottom_edges[c]), c])
            image[int(top_edges[c]) : int(bottom_edges[c]), c] -= background
    return image

def fit_lorentzian(xs, ys, gamma_guess=1):
    #If xs or ys are masked arrays, get only valid entries
    mask = np.zeros(len(xs), dtype=bool)
    if xs is np.ma.MaskedArray:
        mask = xs.mask
    if ys is np.ma.MaskedArray:
        mask = np.logical_or(mask, ys.mask)
    xs = xs[~mask]
    ys = ys[~mask]

    def lorentzian(xs, *p):
        A, mu, gamma, pedestal = p
        zs = (xs - mu) / gamma
        return A / np.pi / gamma / (1 + zs**2) + pedestal
        #return A*np.exp(-(xs-mu)**2/(2.*sigma**2)) + pedestal

    amplitude_guess = np.max(ys)
    mean_guess = xs[np.argmax(ys)]
    p0 = [amplitude_guess, mean_guess, gamma_guess, 0]
    coeff, var_matrix = curve_fit(lorentzian, xs, ys, p0=p0)
    return coeff, lorentzian(xs, *coeff)

def fit_gaussian(xs, ys, errors=None, std_guess=1):
    if errors is None:
        errors = np.ones(len(xs))
        
    #If xs or ys are masked arrays, get only valid entries
    mask = np.zeros(len(xs), dtype=bool)
    if xs is np.ma.MaskedArray:
        mask = xs.mask
    if ys is np.ma.MaskedArray:
        mask = np.logical_or(mask, ys.mask)
    xs = xs[~mask]
    ys = ys[~mask]
    errors = errors[~mask]

    #print(errors)
    def gauss(xs, *p):
        A, mu, sigma, pedestal = p
        return A*np.exp(-(xs-mu)**2/(2.*sigma**2)) + pedestal

    amplitude_guess = np.max(ys)
    mean_guess = xs[np.argmax(ys)]
    p0 = [amplitude_guess, mean_guess, std_guess, 0]
    coeff, var_matrix = curve_fit(gauss, xs, ys, p0=p0, sigma=errors)
    if np.any(np.diag(var_matrix) < 0):
        #ill conditioned fit, set to infinite variance
        fit_errors = np.ones(len(p0)) * np.inf
    else:
        assert(not np.any(np.diag(var_matrix) < 0))
        fit_errors = np.sqrt(np.diag(var_matrix))
    return coeff, fit_errors, gauss(xs, *coeff)


def robust_polyfit(xs, ys, deg, target_xs=None, include_residuals=False, inverse_sigma=None):
    if target_xs is None: target_xs = xs
    ys = astropy.stats.sigma_clip(ys)
    residuals = ys - np.polyval(np.ma.polyfit(xs, ys, deg), xs)
    ys.mask = astropy.stats.sigma_clip(residuals).mask
    last_mask = np.copy(ys.mask)
    while True:
        coeffs = np.ma.polyfit(xs, ys, deg, w=inverse_sigma)
        predicted_ys = np.polyval(coeffs, xs)
        residuals = ys - predicted_ys
        ys.mask = astropy.stats.sigma_clip(residuals).mask
        if np.all(ys.mask == last_mask):
            break
        else:
            last_mask = np.copy(ys.mask)
        
    result = np.polyval(coeffs, target_xs)
    if include_residuals:
        return result, residuals
    return result

def trace_edge(image, y_at_center, lower_edge=False, right_margin=30, window=10):
    edges = scipy.ndimage.sobel(image, axis=0)
    if lower_edge: edges *= -1
    edges[edges < 0] = 0
    edges[0:OVERSCAN_WIDTH + 2] = 0
    columns = np.arange(image.shape[1])
    edge_positions = np.ma.MaskedArray(np.zeros(len(columns)))

    prev_pos = y_at_center
    for c in np.arange(int(image.shape[1]/2), image.shape[1]):
        if c > len(columns) - right_margin: 
            edge_positions[c] = np.ma.masked
            continue

        y_lower = max(0, int(prev_pos - window))
        y_upper = min(int(prev_pos + window), image.shape[0] - 1)
        ys = np.arange(y_lower, y_upper)
        
        try:
            coeffs, _, gaussian = fit_gaussian(ys, edges[y_lower : y_upper, c])
            if np.abs(coeffs[1] - prev_pos) > 1 and c != int(image.shape[1]/2):
                edge_positions[c] = np.ma.masked
                continue
            
            edge_positions[c] = coeffs[1]
            prev_pos = edge_positions[c]
        except RuntimeError:
            edge_positions[c] = np.ma.masked

    prev_pos = y_at_center
    for c in np.arange(int(image.shape[1]/2), 0, -1):
        y_lower = max(0, int(prev_pos - window))
        y_upper = min(int(prev_pos + window), image.shape[0] - 1)
        ys = np.arange(y_lower, y_upper)
        
        try:
            coeffs, _, gaussian = fit_gaussian(ys, edges[y_lower : y_upper, c])
            if np.abs(coeffs[1] - prev_pos) > 1 and c != int(image.shape[1]/2):
                edge_positions[c] = np.ma.masked
                continue
            
            edge_positions[c] = coeffs[1]
            prev_pos = edge_positions[c]
        except RuntimeError:
            edge_positions[c] = np.ma.masked
            
    #plt.plot(columns, edge_positions)
    
    fitted_edge_positions = robust_polyfit(columns, edge_positions, 2)
    '''plt.imshow(image)
    plt.axhline(y_lower, color='black')
    plt.axhline(y_upper, color='black')
    if lower_edge:
        plt.plot(fitted_edge_positions, color='g')
    else:
        plt.plot(fitted_edge_positions, color='r')
    plt.show()'''

    return fitted_edge_positions


def interpolate_missing(image, is_bad_pixel):
    corrected_image = np.copy(image)
    for r in range(image.shape[0]):
        all_y = np.arange(image.shape[1])
        good_y = all_y[~is_bad_pixel[r]]
        good_values = image[r][~is_bad_pixel[r]]
        if len(good_y) == 0:
            corrected_image[r] = 0
        else:
            interpolator = scipy.interpolate.interp1d(good_y, good_values, bounds_error=False, fill_value=np.median(good_values))
            all_values = interpolator(all_y)
            corrected_image[r] = all_values
    
    return corrected_image


def remove_outliers(image, is_bad_pixel=None, window=4, sigma=5):
    if is_bad_pixel is None:
        is_bad_pixel = np.zeros(image.shape, dtype=bool)
        corrected_image = np.copy(image)
    else:
        corrected_image = interpolate_missing(image, is_bad_pixel)        
        is_bad_pixel = np.copy(is_bad_pixel)
        
    x_means = scipy.signal.convolve2d(corrected_image, np.ones((1, window)), mode='same') / window
    y_means = scipy.signal.convolve2d(corrected_image, np.ones((window, 1)), mode='same') / window
    x_flags = corrected_image - x_means
    y_flags = corrected_image - y_means
    
    for r in range(corrected_image.shape[0]):
        outliers = np.abs(y_flags[r] - np.median(y_flags[r])) > 5 * np.std(y_flags[r])
        is_bad_pixel[r][outliers] = True
        
    for c in range(corrected_image.shape[1]):
        outliers = np.abs(x_flags[:, c] - np.median(x_flags[:, c])) > 5 * np.std(x_flags[:, c])
        is_bad_pixel[:, c][outliers] = True
    
    corrected_image = interpolate_missing(corrected_image, is_bad_pixel)
    return corrected_image, is_bad_pixel

def remove_outliers_square(image, binwidth=11, nsigma=6, iternumb=5, plotflags=False):
    '''Flag and replace bad pixels by comparing pixel values with their neighbors 
    in a binwidth by binwidth region.  Although nsigma=6 may seem high,
    due to the limited number of samples from which the standard deviation is calculated,
    sigma > 5 is encountered fairly frequently with artificial 512 x 512 pixel images'''

    image = np.copy(image)
    kernel = np.ones((binwidth, binwidth))
    kernel[int(binwidth/2), int(binwidth/2)] = 0 #exclude central pixel

    for iter in range(iternumb):
        #Find mean and standard deviation of the binwidth x binwidth region
        #surrounding every pixel, using Var(X) = E(X^2) - E(X)^2
        mean_within_kernel = scipy.ndimage.convolve(image, kernel)/np.sum(kernel)
        mean_square_within_kernel = scipy.ndimage.convolve(image**2, kernel)/np.sum(kernel)
        std_within_kernel = np.sqrt(mean_square_within_kernel - mean_within_kernel**2)
        z_scores = np.abs(image-mean_within_kernel)/std_within_kernel
        outlier_ys, outlier_xs = np.where(z_scores > nsigma)

        if len(outlier_ys) == 0: break
        for y,x in zip(outlier_ys, outlier_xs):
            min_y = max(0, y-int(binwidth/2))
            max_y = min_y + binwidth
            image[y, x] = np.median(image[min_y : max_y, x])
            assert(not np.isnan(image[y,x]))
    return image

def remove_outliers_2D(image, binwidth=11, nsigma=6, iternumb=5, plotflags=False):
        '''Flag and replace bad pixels by comparing pixel values with their neighbors 
        in a binwidth by binwidth region.  Although nsigma=6 may seem high,
        due to the limited number of samples from which the standard deviation is calculated,
        sigma > 5 is encountered fairly frequently with artificial 512 x 512 pixel images'''

        kernel = np.ones((binwidth, binwidth))
        kernel[int(binwidth/2), int(binwidth/2)] = 0 #exclude central pixel
        
        for iter in range(iternumb):
            #Find mean and standard deviation of the binwidth x binwidth region
            #surrounding every pixel, using Var(X) = E(X^2) - E(X)^2
            mean_within_kernel = scipy.ndimage.convolve(image, kernel)/np.sum(kernel)
            mean_square_within_kernel = scipy.ndimage.convolve(image**2, kernel)/np.sum(kernel)
            std_within_kernel = np.sqrt(mean_square_within_kernel - mean_within_kernel**2)
            z_scores = np.abs(image-mean_within_kernel)/std_within_kernel
            outlier_ys, outlier_xs = np.where(z_scores > nsigma)

            if len(outlier_ys) == 0: break
            for y,x in zip(outlier_ys, outlier_xs):
                min_y = max(0, y-int(binwidth/2))
                max_y = min_y + binwidth
                image[y, x] = np.median(image[min_y : max_y, x])
                assert(not np.isnan(image[y,x]))

def interpolate_missing_2D(image, is_bad_pixel):
    image = np.copy(image)
    good_ys, good_xs = np.where(~is_bad_pixel)
    interpolator = scipy.interpolate.interp2d(good_xs, good_ys, image[good_ys, good_xs])
    bad_ys, bad_xs = np.where(is_bad_pixel)
    replace_values = interpolator(bad_ys, bad_ys)
    image[bad_ys, bad_xs] = replace_values
    return image

from pathlib import Path
import glob
import os

def find_files_and_write(pattern, output_file, base_dir = '.'):
    search_pattern = os.path.join(base_dir, pattern)
    matched_files = glob.glob(search_pattern) 
    print(matched_files)

    # Write results to a text file
    with open(output_file, 'w') as f:
        for file in matched_files:
            f.write(file + '\n')

    print("Wrote matched files to", output_file)

    return matched_files

def read_config_file(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def estimate_pwv_mm(temp_c, relative_humidity, pressure_hpa, altitude_m=4200):
    """
    Estimate Precipitable Water Vapor (PWV) in mm using surface meteorological conditions,
    scaled to a high-altitude site like Mauna Kea.

    Parameters:
        temp_c (float): Ambient temperature in Celsius.
        dewpoint_c (float): Dewpoint temperature in Celsius.
        pressure_hpa (float): Atmospheric pressure in hPa.
        altitude_m (float): Altitude in meters (default is Mauna Kea summit ~4200 m).

    Returns:
        float: PWV in millimeters, scaled for altitude.
    """
    # Constants
    Rv = 461.0  # J/(kg·K)
    rho_w = 1000  # kg/m^3 (liquid water)
    H = 2000  # scale height in meters

    # Calculate water vapor pressure (kPa) using Teten's formula 
    e_s = 0.611*10**(7.5*temp_c/(temp_c+237.3)) 

    # Convert to hPa
    e_s_hpa = e_s * 10

    # Find actual vapor pressure (hPa)
    e = relative_humidity / 100 * e_s_hpa

    # Convert to Pa 
    e_Pa = e * 100

    # Find the water vapor density (kg/m^3)
    rho_v0 = e_Pa / (Rv* (temp_c + 273.15))

    # Find precipitable water vapor (PWV) in meters 
    PMV_m = (rho_v0 * H) / rho_w  # height in m
    # Convert to mm
    PWV_mm = PMV_m * 1000 

    return PWV_mm 