import sys
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import os
from NIRSPEC_CONSTANTS import READ_NOISE, GAIN, N, OVERSCAN_WIDTH

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

from utils import robust_polyfit
from utils import fit_gaussian

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


def subtract_crosstalk(img, coeffs_filename="mix_coeffs.npy"):
    coeffs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), coeffs_filename)
    coeffs = np.load(coeffs_path)
    #print(coeffs[0,11]*1e6)
    #coeffs[0, 11] = -1000e-6 #550e-6

     
    #temp = np.array([ 0.00000000e+00, -5.78506825e-05,  3.36828893e-05, -9.07654064e-03, 4.35397856e-04, -5.72491753e-05, -1.01765366e-02,  7.08962868e-04, -1.86239925e-02,  3.90835154e-04, -1.56137264e-02,  4.59122269e-04, 8.32785285e-04, -1.49368488e-04,  8.15674993e-04,  4.58055127e-04, 5.08265547e-04,  1.03126968e-03, -3.10582105e-02,  4.65004200e-04, 4.29638875e-04,  4.05390152e-04, -4.16156844e-02,  5.56670478e-04, 4.42925353e-04,  3.72492778e-04,  4.67418887e-04, -2.97453256e-02, 4.23506527e-04,  5.27850262e-04,  4.83045554e-04,  8.64591765e-04])
    reliable_channels = np.array([1,2,4,5,7,9,11,12,15,16,19,20,21,23,24,25,26,28,29,30,31])
    #coeffs[0][reliable_channels] = temp[reliable_channels]

    #coeffs[27, 29] = 0 #-330e-6
    channels = []


def subtract_overscan(image, width=4):
    image = np.copy(image)
    left_right_overscan = np.hstack([image[:, 0:OVERSCAN_WIDTH], image[:, -OVERSCAN_WIDTH:]])
    row_biases = np.median(left_right_overscan, axis=1)

    for r in range(image.shape[0]):
        if r % 128 == 0 or r % 128 == 64:
            image[r] -= row_biases[r]
    return image
