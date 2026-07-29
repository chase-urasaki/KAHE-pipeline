"""
Generate master dark frames for the KAHE pipeline or general calibration
"""
#%%
import sys
import os
import argparse 
import configparser
from pathlib import Path 
from typing import Tuple, List, Optional
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import astropy.stats

# Read in config file
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

def get_dark_params_from_config(config: configparser.ConfigParser) -> Tuple[str, float, str, str]:
    """ Extract dark frame parameters from the configuration.
    Arguments:
        config (configparser.ConfigParser): Parsed configuration object.
        
    Returns:
        data_dir (str): Directory containing dark frames.
        sigma_clip (float): Sigma clipping threshold for dark frame combination.
        target_name (str): Name of the target object.
        obs_date (str): Observation date in YYMMDD format.
    """
    try:
        data_dir = str(config["File Lists"]["darks"])
        SIGMA_CLIP = float(config["Pixel Masking"]["dark_sigma"])
        target_name = str(config["Target"]["name"])
        obs_date = str(config["Target"]["date"])


    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        raise ValueError(f"Missing required configuration parameter: {e}")
    
    return data_dir, SIGMA_CLIP, target_name, obs_date

def combine_darks(darks_dir, std_threshold, min_frames=3, relaxed=False, show_plots=True):
    """
    Combine multiple dark frames into a master dark frame, while identifying and masking bad pixels.

    Arguments:
        darks_dir (str): Directory containing dark frames.
        std_threshold (float): Threshold for standard deviation to flag variable pixels.
        min_frames (int): Minimum number of dark frames required to proceed.
        relaxed (bool): If True, proceed even if fewer than min_frames are provided.
        show_plots (bool): If True, display plots of the master dark, standard deviation, and mask.

    Returns:
        masterdark (np.ndarray): Combined master dark frame.
        std (np.ndarray): Standard deviation of the dark frames.
        master_mask (np.ndarray): Mask indicating bad pixels.
    """
    filenames = [os.path.join(darks_dir, f) for f in os.listdir(darks_dir) if f.endswith('.fits')]

    if len(filenames) < min_frames:
        msg = f"Only {len(filenames)} dark frames provided (min recommended: {min_frames})."
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

    print(f"Bright/dark masked pixels: {n_bright}")
    print(f"Variable pixels: {n_variable} (σ > {std_threshold})")

    # Heuristic warning
    if n_variable > 0.1 * np.prod(std.shape) and not relaxed:
        print("More than 10% of pixels flagged as too variable.")
        print("Consider relaxing the std threshold or enabling relaxed mode.")

    # Show mask image
    plt.imshow(master_mask, origin='lower', cmap='gray')
    plt.title("Master Mask")
    plt.colorbar()
    plt.show()

    return masterdark, std, master_mask

def save_master_dark(masterdark, std, mask, target_name, date):
    """ Save the master dark frame, standard deviation, and mask to a FITS file.
    
    Arguments:
        masterdark (np.ndarray): Combined master dark frame.
        std (np.ndarray): Standard deviation of the dark frames.
        mask (np.ndarray): Mask indicating bad pixels.
        target_name (str): Name of the target object.
        date (str): Observation date.

    Returns: 
        None
    """
    # Construct output path
    output_dir = f"{target_name}_{date}/cals"
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{target_name}_{date}_masterdark.fits")
    
    hdul = fits.HDUList([
        fits.PrimaryHDU(masterdark),
        fits.ImageHDU(std, name="STD"),
        fits.ImageHDU(np.array(mask, dtype=int), name="MASK")
    ])
    hdul.writeto(output_path, overwrite=True)
    print(f"✅ Master dark saved to {output_path}")

def main():
    """Main entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        description="Generate master dark frame for KAHE pipeline"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to pipeline config file"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        help="Override sigma threshold from config"
    )
    parser.add_argument(
        "--relaxed",
        action="store_true",
        help="Allow fewer than minimum frames"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip diagnostic plots"
    )
    
    args = parser.parse_args()
    
    # Read config
    try:
        config = read_config_file(args.config)
        data_dir, sigma, target_name, date = get_dark_params_from_config(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Config error: {e}")
        return 1
    
    # CLI args can override sigma
    if args.sigma:
        sigma = args.sigma
    
    # Process darks
    try:
        masterdark, std, mask = combine_darks(
            data_dir,
            std_threshold=sigma,
            min_frames=3,
            relaxed=args.relaxed,
            show_plots=not args.no_plots
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1
    
    # Save result
    save_master_dark(masterdark, std, mask, target_name, date)
    print("Master dark generation complete.")
    return 0


#%%
if __name__ == "__main__":
    sys.exit(main())