import sys
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import os
from kahe.instruments.nirspec import READ_NOISE, GAIN, N, OVERSCAN_WIDTH

import numpy as np
import scipy.interpolate
import scipy.signal
from scipy.optimize import curve_fit
import astropy.stats
import matplotlib.pyplot as plt
import time
import os
import configparser

from kahe.utils.math import robust_polyfit
from kahe.utils.math import fit_gaussian

def read_config_file(config_path: str) -> configparser.ConfigParser:
    """ Read and parse the pipeline configuration file. 
    
    Arguments:
        config_path (str): Path to the configuration file.
        
    Returns:
        config (configparser.ConfigParser): Parsed configuration object.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def ds9(a):
    """ ds9 shrotcut from Mike Bottom"""
    fits.writeto('temp.fits', np.array(a), overwrite=True)
    os.system('ds9 temp.fits -zoom to fit &')
    return

def trace_edge(image, y_positions, x_positions, lower_edge=False, right_margin=30, window=10):
    """
    Trace an order edge using three known positions to create an initial fit.
    
    Args:
        image: The flat field image
        y_positions: List of 3 y-positions [left, mid, right] for the edge
        x_positions: List of 3 x-positions [left, mid, right] where y_positions were detected
        lower_edge: True if tracing a lower edge (flips edge detection)
        right_margin: Pixels to exclude from right edge
        window: Search window size around expected position
    
    Returns:
        fitted_edge_positions: Polynomial fit of the traced edge
    """
    # Create edge-detected image
    edges = scipy.ndimage.sobel(image, axis=0)
    if lower_edge: 
        edges *= -1
    edges[edges < 0] = 0
    edges[0:OVERSCAN_WIDTH + 2] = 0
    
    columns = np.arange(image.shape[1])
    edge_positions = np.ma.MaskedArray(np.zeros(len(columns)))
    
    # Create initial polynomial fit from the three known positions
    # This gives us a prior for where the edge should be at each column
    initial_poly_coeffs = np.polyfit(x_positions, y_positions, 2)
    initial_fit = np.polyval(initial_poly_coeffs, columns)
    
    # Trace all columns using the initial fit as a guide
    for c in columns:
        if c > len(columns) - right_margin - 1:
            edge_positions[c] = np.ma.masked
            continue
        
        # Use the initial polynomial fit as the expected position
        expected_pos = initial_fit[c]
        
        y_lower = max(0, int(expected_pos - window))
        y_upper = min(int(expected_pos + window), image.shape[0] - 1)
        ys = np.arange(y_lower, y_upper)
        
        try:
            coeffs, _, gaussian = fit_gaussian(ys, edges[y_lower : y_upper, c])
            # Check if the detected position is reasonably close to expected
            if np.abs(coeffs[1] - expected_pos) > 2 * window:
                edge_positions[c] = np.ma.masked
                continue
            
            edge_positions[c] = coeffs[1]
        except (RuntimeError, ValueError):
            edge_positions[c] = np.ma.masked
    
    # Final polynomial fit using all detected positions
    fitted_edge_positions = robust_polyfit(columns, edge_positions, 2)
    
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


class NestedInterpolator:
    def __init__(self, target_xs, xs, data, ancestor=None):
        self.target_xs = target_xs
        
        if ancestor is not None:
            self.xs = xs #ancestor.xs
            self.data = data #ancestor.data
            transform = self.get_transform()
            self.transform = self.merge_transforms(ancestor.transform, transform)

            self.xs = ancestor.xs
            self.data = ancestor.data
        else:
            self.xs = xs
            self.data = data
            self.transform = self.get_transform()

    def get_transform(self):
        target_xs = np.atleast_1d(self.target_xs)
        locations = np.searchsorted(self.xs, target_xs)
        transform = []
        
        for i in range(len(target_xs)):
            if locations[i] == 0:
                transform.append([[0, 1]])
            elif locations[i] == len(self.xs):
                transform.append([[len(self.xs) - 1, 1]])
            else:
                right = locations[i]
                left = right - 1
                assert(self.xs[left] <= target_xs[i] and self.xs[right] >= target_xs[i])
                dist = self.xs[right] - self.xs[left]
                right_weight = (target_xs[i] - self.xs[left]) / dist
                left_weight = (self.xs[right] - target_xs[i]) / dist
                #print(i, left, left_weight, right, right_weight)
                transform.append([[left, left_weight], [right, right_weight]])

        return transform

    def evaluate(self):
        transform = self.transform
        transformed = []
        for i in range(len(transform)):
            val = 0.
            for j in range(len(transform[i])):
                index, weight = transform[i][j]
                val += weight * self.data[index]
            transformed.append(val)
        return np.array(transformed)

    def merge_transforms(self, first, second):
        combined = []
        for i in range(len(second)):
            combined.append([])
            index_dict = {}
            for first_index, first_weight in second[i]:
                for original_index, original_weight in first[first_index]:
                    if original_index not in index_dict:
                        combined[-1].append([original_index, first_weight * original_weight])
                        index_dict[original_index] = len(combined[-1]) - 1
                    else:
                        loc = index_dict[original_index]
                        combined[-1][loc][1] += first_weight * original_weight
                    
        #for i in range(len(combined)):
        #    print(i, np.array(combined[i]).shape)

        return combined
    
    #@profile
    def get_matrix(self, errors=None):
        transform = self.transform
        if errors is None:
            errors = np.ones(len(self.data))
        #transform = np.array(transform)
        contributions = np.zeros((len(transform), len(self.data)))
        for i in range(len(transform)):
            indices, weights = np.array(transform[i]).T
            indices = np.array(indices, dtype=int)
            #print(indices, weights)
            contributions[i][indices] = weights

        #import pdb
        #pdb.set_trace()
        covariances = contributions.dot(np.diag(errors**2).dot(contributions.T))
        #covariances += np.random.normal(0, 1e-10, covariances.shape)
        '''plt.plot(np.sum(contributions, axis=0))
        plt.plot(np.sum(contributions, axis=1))

        plt.figure()
        plt.imshow(contributions)
        plt.figure()
        plt.imshow(covariances)
        plt.figure()
        plt.imshow(np.linalg.inv(covariances))
        plt.figure()
        plt.imshow(np.linalg.pinv(covariances))
        plt.show()
        #plt.imshow(contributions)
        #plt.show()'''
        return covariances
        
        plt.imshow(contributions.dot(np.diag(contributions.shape[1] * [1.0]).dot(contributions.T)))
        plt.figure()
        #plt.imshow(contributions)
        plt.imshow(covariances)
        plt.figure()
        plt.imshow(np.linalg.inv(covariances))
        #plt.imshow(contributions)
        plt.show()

'''np.random.seed(1337)
data = np.random.normal(0, 1, 213)
xs = np.arange(213)

target_xs = np.linspace(1, 210, 230)
#target_xs = np.array([1.5, 2.9, 5.8, 5.9, 6.0, 6.1, 6.2])
interpolator = NestedInterpolator(target_xs, xs, data)
covariances = interpolator.get_matrix()
interpolated = interpolator.evaluate()

print(np.allclose(interpolated, np.interp(target_xs, xs, data)))

#plt.plot(target_xs, interpolated)
#plt.plot(target_xs, np.interp(target_xs, xs, data))
#plt.show()

print(interpolated.T.dot(scipy.linalg.solve(covariances, interpolated)),
      interpolated.T.dot(np.linalg.pinv(covariances).dot(interpolated)))

second_target_xs = np.linspace(1.1, 205, 250)
#second_target_xs = [1.8, 3.2, 5, 5.5, 6.05]
second_interpolator = NestedInterpolator(second_target_xs, target_xs, interpolated, interpolator)
second_interpolator.get_matrix()
second_interpolated = second_interpolator.evaluate()

plt.plot(second_target_xs, second_interpolated)
plt.plot(second_target_xs, np.interp(second_target_xs, target_xs, interpolated))
plt.title("Second comparison")
plt.show()

covariances = second_interpolator.get_matrix()

print("Condition number", scipy.linalg.norm(covariances) * scipy.linalg.norm(np.linalg.pinv(covariances)))
sign, slogdet = np.linalg.slogdet(covariances)
print("Sign, slogdet", sign, slogdet)
ln_L = -0.5 * (second_interpolated.T.dot(np.linalg.pinv(covariances).dot(second_interpolated)) + sign * slogdet + len(covariances) * np.log(2*np.pi))

print(second_interpolated.T.dot(scipy.linalg.solve(covariances, second_interpolated)),
      second_interpolated.T.dot(np.linalg.pinv(covariances).dot(second_interpolated)), ln_L)

#second_interpolated = second_interpolator.apply_transform(second_transform)
#second_interpolated_alt = interpolator.apply_transform(combined)
#interpolator.get_matrix(combined)
#print(second_interpolated)
#print(second_interpolated_alt)

plt.plot(xs, data)
#plt.plot(target_xs, interpolated)
plt.plot(second_target_xs, second_interpolated)
#plt.plot(second_target_xs, second_interpolated_alt)
plt.show()
'''
