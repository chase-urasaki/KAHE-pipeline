"""
This script reduces the images by doing the following: 
0) Opens the masterflat file and reads the data and mask
1) Registers pairs of science images for pair subtraction
2) Removes read noise
3) Estimates variance and background from GAIN
4) Performs pairwise subtraction



Notes: 
- The script is designed to be run from the command line with the following arguments: 
    python make_masterdark.py *darks_list*.txt

"""
import sys
import shutil
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from scripts.utils import remove_outliers, interpolate_missing, remove_outliers_square
import os
# from scripts.nirspec_constants import READ_NOISE, GAIN, N, OVERSCAN_WIDTH
from scripts.corrections import subtract_repetitive, subtract_crosstalk
from redspec_badpix import fixpix
from scripts.utils import read_config_file
import argparse
import glob
import importlib

# Import Constants 
nirspec_constants = importlib.import_module("scripts.nirspec_constants")
READ_NOISE = nirspec_constants.READ_NOISE
GAIN = nirspec_constants.GAIN
N = nirspec_constants.N
OVERSCAN_WIDTH = nirspec_constants.OVERSCAN_WIDTH


def make_reduced_directories(): 
    # Create calibration directory if it doesn't exist
    reduced_fits = f"reduced/{NAME}_{DATE}/fits"
    reduced_jpgs = f"reduced/{NAME}_{DATE}/jpgs"
    if not os.path.exists(reduced_fits):
        os.makedirs(reduced_fits)
    else: 
        shutil.rmtree(reduced_fits)
        os.makedirs(reduced_fits)
    
    if not os.path.exists(reduced_jpgs): 
        os.makedirs(reduced_jpgs)
    else: 
        shutil.rmtree(reduced_jpgs)
        os.makedirs(reduced_jpgs)

    return reduced_fits, reduced_jpgs
        
def zero_inter_order_regions(image, edges, margin=1):
    zeroed_image = np.zeros(image.shape)
    for o in range(len(edges)):
        plt.plot(edges[o][0], color='r')
        plt.plot(edges[o][1], color='g')
        for c in range(N):
            min_y = int(round(edges[o][0][c]) + 1)
            max_y = int(round(edges[o][1][c]) - 1)
            zeroed_image[min_y : max_y, c] = image[min_y : max_y, c]
    #plt.imshow(zeroed_image, vmin=-100, vmax=100)
    #plt.show()
    return zeroed_image

if __name__ == "__main__":
    print("Getting flats and bad pixel mask...")

    parser = argparse.ArgumentParser(description="Perform CCD processing using the master flat. Optionally specify a path to a file")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--masterflat", type=str, default=None,
                        help="Path to an existing master dark FITS file (optional).")
    args = parser.parse_args()
    config = read_config_file(args.config)

    NAME = config['Target']['name']
    DATE = config['Target']['date']
    COSMIC_RAYS_SIGMA  = int(config['Pixel Masking']['cosmic_rays_sigma'])

    if args.masterflat:
        print(f"Using existing master dark from: {args.masterflat}")
        with fits.open(args.masterflat) as hdul:
            masterflat = hdul[0].data
        masterflat_hdul = fits.open(args.masterflat)
        masterflat = masterflat_hdul[0].data
            #std = hdul["STD"].data if "STD" in hdul else np.zeros_like(masterdark)
            #mask = hdul["MASK"].data.astype(bool) if "MASK" in hdul else np.zeros_like(masterdark, dtype=bool)
    else:
        masterflat_hdul = fits.open(f'./calibrations/{NAME}_{DATE}/{DATE}_masterflat.fits')
        masterflat = masterflat_hdul[0].data
        # with fits.open(f'calibrations/{NAME}_{DATE}/masterflat.fits') as hdul:
        #     masterflat = hdul[0].data
        std = masterflat_hdul["STD"].data if "STD" in masterflat_hdul else np.zeros_like(masterflat)
        mask = masterflat_hdul["MASK"].data.astype(bool) if "MASK" in masterflat_hdul else np.zeros_like(masterflat, dtype=bool)
    
    #masterflat_hdul = fits.open(sys.argv[1])
    #masterflat = masterflat_hdul[0].data
    masterflat[masterflat == 0] = 1

    masterflat_mask = np.zeros_like(masterflat, dtype=bool)
    flat_bad_pixels = np.zeros_like(masterflat, dtype=bool)
    
    if "MASK" in masterflat_hdul:
        masterflat_mask = np.array(masterflat_hdul["MASK"].data, dtype=bool)
    else: 
        print("No mask found in masterflat. Please double check masterflat fits. Aborting...")
        sys.exit(1)

    if "BADPIX" in masterflat_hdul:
        flat_bad_pixels = np.array(masterflat_hdul["BADPIX"].data, dtype=bool)
    else: 
        print("No bad pixel map found in masterflat. Please double check masterflat fits. Aborting...")
        sys.exit(1)

    science_filenames = [line.strip() for line in open(f'{NAME}_{DATE}_science.txt')]

    print('Making directories...')
    red_fits_dir, red_jpg_dir = make_reduced_directories()

    for pair_num in range(int(len(science_filenames)/2)):
        i = 2 * pair_num    
        print("Processing {} and {}".format(
            science_filenames[i], science_filenames[i+1]))
            
        with fits.open(science_filenames[i]) as hdul:        
            first_image = np.rot90(hdul[0].data, 3)
                    
            first_mjd = hdul[0].header["MJD"]
            echelle_angle = hdul[0].header["ECHLPOS"]
            disp_angle = hdul[0].header["DISPPOS"]
            num_reads = hdul[0].header["NUMREADS"]
            num_coadds = hdul[0].header["COADDS"]
                    
        with fits.open(science_filenames[i+1]) as hdul:
            second_image = np.rot90(hdul[0].data, 3)
            second_mjd = hdul[0].header["MJD"]
            assert(hdul[0].header["NUMREADS"] == num_reads)
            assert(hdul[0].header["COADDS"] == num_coadds)


        diff_image = subtract_crosstalk(first_image - second_image)
        diff_image = subtract_repetitive(diff_image, 128, 0)
        diff_image = subtract_repetitive(diff_image, 128, 1)
        diff_image = subtract_repetitive(diff_image, 128, 64)

        # Remove cosmic rays

        diff_image = remove_outliers_square(diff_image, binwidth=11, nsigma=COSMIC_RAYS_SIGMA)
        
        read_noise = READ_NOISE / np.sqrt(num_reads) * np.sqrt(num_coadds) * np.sqrt(2)

        diff_image = diff_image / (masterflat) * GAIN 
        diff_image = interpolate_missing(diff_image, flat_bad_pixels)
        diff_image = zero_inter_order_regions(diff_image, masterflat_hdul["EDGES"].data)

        # #Uncomment below to use only first order
        # ymin = int(min(masterflat_hdul["EDGES"].data[0][0]))
        # ymax = int(max(masterflat_hdul["EDGES"].data[0][1]))
        # print("ymin, ymax", ymin, ymax)
        # diff_image, _ = fixpix(diff_image[ymin:ymax])
        
        image_hdu = fits.PrimaryHDU(diff_image)
        image_hdu.header["MJD"] = first_mjd
        image_hdu.header["ECHLPOS"] = echelle_angle
        image_hdu.header["DISPPOS"] = disp_angle
        image_hdu.header["RNOISE"] = read_noise
        bad_pixels_hdu = fits.ImageHDU(np.array(flat_bad_pixels, dtype=int), name="BADPIX")

        print('Computing variance...')
        var_estimate = GAIN * (np.abs(first_image) + np.abs(second_image)) /masterflat**2 + read_noise**2
        var_estimate = zero_inter_order_regions(var_estimate, masterflat_hdul["EDGES"].data)
        var_estimate = interpolate_missing(var_estimate, flat_bad_pixels)
        variance_hdu = fits.ImageHDU(var_estimate, name="VAR")

        print('Computing background...')
        bkd_estimate = GAIN * ((first_image + second_image) - np.abs(first_image - second_image)) / masterflat
        bkd_estimate = zero_inter_order_regions(bkd_estimate, masterflat_hdul["EDGES"].data)
        bkd_estimate, _ = remove_outliers(bkd_estimate, flat_bad_pixels)
        bkd_hdu = fits.ImageHDU(bkd_estimate, name="BKD")
        
        print(f'Writing fits for {science_filenames[i]} and {science_filenames[i+1]}...')
        hdul = fits.HDUList([image_hdu, variance_hdu, bkd_hdu, bad_pixels_hdu] + list(masterflat_hdul[2:]))
        hdul.writeto(os.path.join(red_fits_dir, "red_" + os.path.basename(science_filenames[i])), overwrite
        =True)
        plt.clf()
        plt.imshow(image_hdu.data, origin = 'lower',vmin = -5_000, vmax = 5_000)
        plt.title(str(science_filenames[i]))
        plt.colorbar()
        #plt.savefig(os.path.join(red_jpg_dir, "red_" + os.path.basename(science_filenames[i]))[:-5] + ".jpg")
        first_basename, _ = os.path.splitext(os.path.basename(science_filenames[i]))
        plt.savefig(os.path.join(red_jpg_dir, f'red_{first_basename}.jpg'))
        plt.draw()
        plt.show(block=False)
        # Optional: add a small sleep if you want the plot to stay visible for a moment
        import time
        time.sleep(2)  # Non-blocking equivalent to plt.pause(2)
        plt.close()
        
        image_hdu.data *= -1
        image_hdu.header["MJD"] = second_mjd
        hdul.writeto(os.path.join(red_fits_dir, "red_" + os.path.basename(science_filenames[i+1])), overwrite=True)
        plt.clf()
        plt.imshow(image_hdu.data, origin='lower', vmin=-5_000, vmax=5_000)
        plt.title(str(science_filenames[i+1]))
        plt.colorbar()
        #plt.savefig(os.path.join(red_jpg_dir, "red_" + os.path.basename(science_filenames[i+1]))[:-5] + ".jpg")
        second_basename, _ = os.path.splitext(os.path.basename(science_filenames[i+1]))
        plt.savefig(os.path.join(red_jpg_dir, f'red_{second_basename}.jpg'))
        plt.pause(2)
        plt.close()
        hdul.close()
        
    masterflat_hdul.close()

    print("Done!")    
