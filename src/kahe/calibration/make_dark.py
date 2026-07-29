"""
This script generates the master dark (for only the flats)by doing the following: 
0) Making a calibrations directory 
1) Median combines darks in the input list of filenames
2) Masks out pixels that are too bright/dark or too variable
3) Writes the masterdark, std, and mask to a fits file

Notes: 
- The script is designed to be run from the command line with the following arguments: 
    python make_masterdark.py

"""
#%%
import sys
import astropy.io.fits as fits
import numpy as np
import os
import matplotlib.pyplot as plt
import astropy.stats
import argparse

# # Read in config file from argument line (specified in pipeline)
# parser = argparse.ArgumentParser()
# parser.add_argument("--config", type=str, required=True, help="Path to the config file")
# parser.add_argument("--masterdark", type=str, help="Path to the master dark file")
# args = parser.parse_args()
# config = read_config_file(args.config)

SIGMA_CLIP = 5


# def make_calibration_directory(): 
#     # Create calibration directory if it doesn't exist
#     calibration_dir = f"calibrations/{NAME}_{DATE}"
#     os.makedirs(calibration_dir, exist_ok=True)
#     return calibration_dir

def combine_darks(filenames, std_threshold=SIGMA_CLIP):
    all_masks = 0.
    data = []

    for filename in filenames:
        with fits.open(filename) as hdulist:
            dark = np.array(hdulist[0].data, dtype=float) / hdulist[0].header["COADDS"]
            data.append(np.rot90(dark, 3))
            #print(data[-1][1470,311])
            plt.imshow(data[-1], vmin=0, vmax=30)
            plt.plot([311], [1470], 'x', color='red')
            plt.ylim(1460,1480)
            plt.xlim(300,320)
            plt.show()
            all_masks += astropy.stats.sigma_clip(data[-1], 5).mask

    masterdark = np.median(data, axis=0)
    std = np.std(data, axis=0)

    master_mask = all_masks > len(filenames)/2
    #Uncomment below to include variable pixels too
    #master_mask = np.logical_or(all_masks > len(filenames)/2, std > std_threshold)
    print("Too bright/dark", np.sum(all_masks > len(filenames)/2))
    print("Too variable", np.sum(std > std_threshold))
    plt.imshow(master_mask, origin='lower', cmap='gray')
    plt.title("Master Mask")
    plt.colorbar()
    plt.plot([311], [1470], 'x', color='red')
    plt.ylim(1460,1480)
    plt.xlim(300,320)
    plt.figure()
    return masterdark, std, master_mask

def combine_darks_robust(filenames, std_threshold=SIGMA_CLIP, min_frames=3, relaxed=False):
    if len(filenames) < min_frames:
        msg = f"⚠️ Only {len(filenames)} dark frames provided (min recommended: {min_frames})."
        if relaxed:
            print(msg + " Proceeding anyway due to relaxed mode.")
        else:
            raise ValueError(msg + " Use relaxed=True to override. \n"
            "Suggestion: Either collect more dark frames or use an existing master dark.")

    all_masks = 0.
    data = []

    for filename in filenames:
        with fits.open(filename) as hdulist:
            dark = np.array(hdulist[0].data, dtype=float) / hdulist[0].header["COADDS"]
            data.append(np.rot90(dark, 3))
            all_masks += astropy.stats.sigma_clip(data[-1], 5).mask

    data = np.array(data)
    masterdark = np.median(data, axis=0)
    std = np.std(data, axis=0)

    # Flags for bad pixels
    brightness_mask = all_masks > len(filenames)/2
    variability_mask = std > std_threshold
    master_mask = brightness_mask  # Optionally include variability_mask

    # Log pixel stats
    n_bright = np.sum(brightness_mask)
    n_variable = np.sum(variability_mask)

    print(f"🧼 Bright/dark masked pixels: {n_bright}")
    print(f"📉 Variable pixels: {n_variable} (σ > {std_threshold})")

    # Heuristic warning
    if n_variable > 0.1 * np.prod(std.shape) and not relaxed:
        print("⚠️ More than 10% of pixels flagged as too variable.")
        print("   Consider relaxing the std threshold or enabling relaxed mode.")

    # Show mask image
    plt.imshow(master_mask, origin='lower', cmap='gray')
    plt.title("Master Mask")
    plt.colorbar()
    plt.show()

    return masterdark, std, master_mask
#%%
if __name__ == "__main__":
    #calibration_dir = make_calibration_directory()
    # Specifiy data directory and list of darks
    data_dir = 'FITS/240704'
    list_of_darks = '240704_darks_list.txt'

    filenames = [line.strip() for line in open(list_of_darks)]
    # prepend the directory to each filename
    full_filenames = [os.path.join(data_dir, filename) for filename in filenames]
    # masterdark, std, mask = combine_darks(filenames)
    masterdark, std, mask = combine_darks_robust(full_filenames, std_threshold=SIGMA_CLIP, min_frames=3)
    median_mask = astropy.stats.sigma_clip(masterdark, SIGMA_CLIP).mask

    # plot the masterdark
    plt.imshow(masterdark, origin='lower', cmap='gray')
    plt.title("Master Dark")
    plt.colorbar()
    plt.show()

    # plot the std
    plt.imshow(std, origin='lower', cmap='gray')
    plt.title("Standard Deviation")
    plt.colorbar()
    plt.show()

    # plot the mask
    plt.imshow(median_mask, origin='lower', cmap='gray')
    plt.title("Median Mask")
    plt.colorbar()
    plt.show()

    hdul = fits.HDUList([fits.PrimaryHDU(masterdark),
                        fits.ImageHDU(std, name="STD"),
                        fits.ImageHDU(np.array(mask, dtype=int), name="MASK")])

    hdul.writeto(os.path.join('./','240704_masterdark.fits'), overwrite=True)
    #%%