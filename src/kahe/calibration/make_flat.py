"""
Generate master flat frame for KAHE pipeline
"""
import sys
import os
import argparse
import configparser
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.linalg
import scipy.ndimage
import astropy.io.fits as fits

from kahe.utils.helper_functions import trace_edge, interpolate_missing, subtract_crosstalk, subtract_overscan
import kahe.instruments.nirspec as nirspec



# === CONSTANTS ===
N = nirspec.N
RIGHT_MARGIN = nirspec.RIGHT_MARGIN
LEFT_MARGIN = nirspec.LEFT_MARGIN 
FLAT_MIN = nirspec.FLAT_MIN
FLAT_MAX = nirspec.FLAT_MAX 
MIN_ORDER_SEPARATION = nirspec.MIN_ORDER_SEPARATION
OVERSCAN_WIDTH = nirspec.OVERSCAN_WIDTH
FILTER = nirspec.NIRSPEC1_70
MIN_FRAMES = 3

# === CONFIG FUNCTIONS ===

def read_config_file(config_path: str) -> configparser.ConfigParser:
    """Read and parse the pipeline configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = configparser.ConfigParser()
    config.read(config_path)
    print(f"Read config: {config_path}")
    return config


def get_flat_params_from_config(config: configparser.ConfigParser) -> Tuple[str, str, float, str, str]:
    """Extract flat frame parameters from the configuration.
    
    Arguments:
        config: Parsed configuration object
        
    Returns:
        flats_dir: Directory containing flat frames
        master_dark_path: Path to master dark FITS file
        sigma: Sigma clipping threshold
        target_name: Name of the target object
        obs_date: Observation date in YYMMDD format
    """
    try:
        flats_dir = str(config["File Lists"]["flats"])
        master_dark_path = str(config["Calibrations"]["master_dark_path"])
        sigma = float(config["Pixel Masking"]["flat_sigma"])
        target_name = str(config["Target"]["name"])
        obs_date = str(config["Target"]["date"])
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        raise ValueError(f"Missing required configuration parameter: {e}")
    
    return flats_dir, master_dark_path, sigma, target_name, obs_date


# === CORE ALGORITHMS ===

def combine_flats(flats_dir: str, masterdark: np.ndarray, 
                  min_frames: int = MIN_FRAMES, relaxed: bool = False, subtract_crosstalk_: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Combine multiple flat frames into a master flat frame.
    
    Arguments:
        flats_dir: Directory containing flat frames
        masterdark: Master dark frame to subtract
        min_frames: Minimum number of frames required
        relaxed: If True, proceed with fewer frames
        
    Returns:
        masterflat: Combined master flat frame
        std: Standard deviation of the flat frames
    """
    filenames = [os.path.join(flats_dir, f) for f in os.listdir(flats_dir) if f.endswith('.fits')]
    
    if len(filenames) < min_frames:
        msg = f"Only {len(filenames)} flat frames provided (min: {min_frames})"
        if relaxed:
            print(f"{msg}. Proceeding in relaxed mode.")
        else:
            raise ValueError(f"{msg}. Use relaxed=True to override.")
    
    print(f"Processing {len(filenames)} flat frames...")
    data = np.zeros((len(filenames), N, N))
    medians = np.zeros(len(filenames))
    
    for i, filename in enumerate(filenames, 1):
        with fits.open(filename) as hdulist:
            flat = np.rot90(hdulist[0].data, 3) / hdulist[0].header["COADDS"]
            
            # Optional crosstalk correction
            if subtract_crosstalk_:
                flat = subtract_crosstalk(flat)
                if np.any(np.isnan(flat)):
                    print(f"Warning: {Path(filename).name} contains NaN values after crosstalk correction")
            
            # Overscan correction
            flat = subtract_overscan(flat)
            
            # Subtract master dark
            flat = flat - masterdark
            
            # Normalize
            medians[i-1] = np.median(flat)
            flat /= np.median(flat)
            data[i-1] = flat
        
        print(f"   [{i}/{len(filenames)}] {Path(filename).name}")
    
    masterflat = np.median(data, axis=0)
    std = np.std(data, axis=0)
    
    print(f"Combined {len(filenames)} flat frames")
    return masterflat, std


def fit_and_divide(masterflat, top_edges, bot_edges, X_ORD=5, Y_ORD=3):
    print('Fitting and dividing...')
    normalization = np.ones(masterflat.shape) * np.inf
    for o in range(len(top_edges)):
        top_edge = top_edges[o]-10
        bot_edge = bot_edges[o]
        #if o == 0: top_edge += 10 #Hack
                    
        all_scaled_xs = []
        all_scaled_ys = []
        all_values = []
        all_xs = []
        all_ys = []

        for c in range(LEFT_MARGIN, masterflat.shape[1] - RIGHT_MARGIN):
            y_min = int(round(top_edge[c])) + 1
            y_max = int(round(bot_edge[c])) - 1
            rows = np.arange(y_min, y_max)
            scaled_ys = (rows - y_min) / (y_max - y_min)
            all_scaled_xs += [c / masterflat.shape[1]] * (y_max - y_min)
            all_scaled_ys += list(scaled_ys)
            all_values += list(masterflat[y_min : y_max, c])
            all_xs += [c] * (y_max - y_min)
            all_ys += list(rows)

        all_scaled_xs = np.array(all_scaled_xs)
        all_scaled_ys = np.array(all_scaled_ys)
        all_values = np.array(all_values)
        all_xs = np.array(all_xs)
        all_ys = np.array(all_ys)
        
        A = []
        for i in range(X_ORD):
            for j in range(Y_ORD):
                A.append(all_scaled_xs**i * all_scaled_ys**j)
        A = np.array(A).T
        coeffs, _, _, _ = scipy.linalg.lstsq(A, all_values)
        predicted = A.dot(coeffs)
        normalization[all_ys, all_xs] = predicted
        #plt.scatter(all_scaled_xs, all_values)
        #plt.plot(all_scaled_xs, predicted, color='r')
        #plt.show()
        
            
    return masterflat/normalization, normalization


def identify_anomalous_gains(masterflat, bad_pixel_map, top_edges, bottom_edges, margin=1):
    """ Identify pixels with anomalous gains in the master flat and update the bad pixel map.

    Arguments:
        masterflat: The combined master flat frame
        bad_pixel_map: Current bad pixel mask
        top_edges: Array of top edge traces for each order
        bottom_edges: Array of bottom edge traces for each order
        margin: Margin in pixels to avoid near the edges

    Returns:
        Updated bad pixel map with anomalous gains flagged
    """

    print('Identifying anomalous gains...')
    bad_pixel_map = np.copy(bad_pixel_map)
    for o in range(len(top_edges)):
        for c in range(N):
            min_y = int(np.round(top_edges[o][c])) + margin
            max_y = int(np.round(bottom_edges[o][c])) - margin
            bad_gains = np.logical_or(masterflat[min_y : max_y, c] < FLAT_MIN,
                                      masterflat[min_y : max_y, c] > FLAT_MAX)
            bad_pixel_map[min_y : max_y, c][bad_gains] = True
    return bad_pixel_map



def trace_edges(masterflat, FILTER=None, window=10, percentile=98, 
                show_plots=True, orders=None):
    """Trace spectral order edges in the master flat.
    
    Arguments:
        masterflat: Master flat frame
        FILTER: Optional filter bounds tuple (upper, lower)
        window: Window size for edge detection
        percentile: Percentile threshold for peak finding
        show_plots: If True, display diagnostic plots
        orders: Which orders to trace. Options:
            - None (default): trace all detected orders
            - int: trace single order by index (e.g., -3 for third from end)
            - list of ints: trace specific orders by indices (e.g., [0, 2, 5])
            - "all": explicitly trace all orders
        
    Returns:
        upper_edges: Array of upper edge traces
        lower_edges: Array of lower edge traces
    """
    print('Tracing edges...')
    
    if FILTER is None:
        # Detect edges along the vertical (spatial) axis
        edges = scipy.ndimage.sobel(masterflat, axis=0)
        edges[0:OVERSCAN_WIDTH + 2] = 0  # Remove overscan rows

        # Compute the vertical profile at the center columns
        midsection = np.median(edges[:, int(N/2 - window) : int(N/2 + window)], axis=1)
        #quarter_section_left = np.median(edges[:, int(N/4 - window) : int(N/4 + window)], axis=1)
        # quarter_section_right = np.median(edges[:, int(N*3/4 - window) : int(N*3/4 + window)], axis=1)

        upper_positions = scipy.signal.find_peaks(midsection,
                                                  height=np.percentile(midsection, percentile),
                                                  distance=MIN_ORDER_SEPARATION)[0]
        lower_positions = scipy.signal.find_peaks(-midsection,
                                                  height=np.percentile(-midsection, percentile),
                                                  distance=MIN_ORDER_SEPARATION)[0]

        print(f"  Found {len(upper_positions)} upper positions: {upper_positions}")
        print(f"  Found {len(lower_positions)} lower positions: {lower_positions}")
        
        if show_plots:
            plt.figure(figsize=(12, 6))
            plt.imshow(masterflat, origin='lower', aspect='auto', cmap='gray')
            plt.title("Master Flat with Edge Traces")
            plt.colorbar(label='Counts')

            for pos in upper_positions:
                plt.axhline(pos, color='lime', linestyle='--', lw=1)
            for pos in lower_positions:
                plt.axhline(pos, color='magenta', linestyle='--', lw=1)

            plt.xlabel("X (columns)")
            plt.ylabel("Y (rows)")
            plt.tight_layout()
            plt.show()

    else:
        # Use filter bounds if provided
        bounds = FILTER
        if len(bounds) == 2:
            upper_positions = np.array([bounds[0]])
            lower_positions = np.array([bounds[1]])

    # Determine which orders to trace
    num_orders = min(len(upper_positions), len(lower_positions))
    
    if orders is None or orders == "all":
        # Trace all detected orders
        order_indices = list(range(num_orders))
    elif isinstance(orders, int):
        # Single order by index
        order_indices = [orders if orders >= 0 else num_orders + orders]
    elif isinstance(orders, (list, tuple)):
        # Specific orders by indices
        order_indices = [idx if idx >= 0 else num_orders + idx for idx in orders]
    else:
        raise ValueError(f"Invalid orders parameter: {orders}")
    
    # Validate indices
    for idx in order_indices:
        if idx < 0 or idx >= num_orders:
            raise ValueError(f"Order index {idx} out of range [0, {num_orders-1}]")
    
    print(f"  Tracing {len(order_indices)} order(s): {order_indices}")
    
    # Trace the selected orders
    upper_edges = []
    lower_edges = []
    
    for o in order_indices:
        print(f"    Tracing order {o}...")
        upper_edges.append(trace_edge(masterflat, upper_positions[o], False))
        lower_edges.append(trace_edge(masterflat, lower_positions[o], True))

    return np.array(upper_edges), np.array(lower_edges)


def save_master_flat(masterflat: np.ndarray, bad_pixels: np.ndarray, 
                     edges: Tuple[np.ndarray, np.ndarray], mask: np.ndarray,
                     raw_masterflat: np.ndarray, std: np.ndarray, 
                     normalization: np.ndarray, target_name: str, date: str,
                     output_dir: str = "./cals") -> None:
    """Save master flat to FITS file with all extensions.
        optional save to a png for quick viewing.
    
    Arguments:
        masterflat: Normalized master flat frame
        bad_pixels: Bad pixel mask
        edges: Tuple of (top_edges, bottom_edges)
        mask: Final combined mask
        raw_masterflat: Raw (unnormalized) master flat
        std: Standard deviation
        normalization: Normalization frame
        target_name: Name of target object
        date: Observation date
        output_dir: Output directory for FITS file
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{target_name}_{date}_masterflat.fits")
    
    top_edges, bottom_edges = edges
    
    # Create HDUs
    image_hdu = fits.PrimaryHDU(masterflat)
    bad_pixels_hdu = fits.ImageHDU(np.array(bad_pixels, dtype=int), name="BADPIX")
    
    # Create edge traces table
    cols = fits.ColDefs([
        fits.Column(name='Top edges', format='{}D'.format(N), array=np.array(top_edges)),
        fits.Column(name='Bottom edges', format='{}D'.format(N), array=np.array(bottom_edges))
    ])
    edges_hdu = fits.BinTableHDU.from_columns(cols, name="EDGES")
    
    mask_hdu = fits.ImageHDU(np.array(mask, dtype=int), name="MASK")
    
    # Combine all HDUs
    hdul = fits.HDUList([
        image_hdu,
        bad_pixels_hdu,
        edges_hdu,
        mask_hdu,
        fits.ImageHDU(raw_masterflat, name="RAW"),
        fits.ImageHDU(std, name="STD"),
        fits.ImageHDU(normalization, name="NORMALIZATION")
    ])
    
    hdul.writeto(output_path, overwrite=True)
    print(f"Master flat saved to {output_path}")


# === MAIN CLI ===

def main():
    """Main entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        description="Generate master flat frame for KAHE pipeline"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to pipeline config file"
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
    parser.add_argument(
        "--orders",
        type=str,
        default=None,
        help="Orders to trace: 'all', single index (e.g., '-3'), or comma-separated list (e.g., '0,2,5')"
    )
    
    args = parser.parse_args()
    
    # Parse orders argument
    orders_param = None
    if args.orders is not None:
        orders_arg = args.orders.strip()
        if orders_arg.lower() == "all":
            orders_param = "all"
        elif "," in orders_arg:
            # Multiple orders
            orders_param = [int(x.strip()) for x in orders_arg.split(",")]
        else:
            # Single order
            orders_param = int(orders_arg)
    # If not specified, defaults to None (trace all orders)
    
    # Read config
    try:
        config = read_config_file(args.config)
        flats_dir, master_dark_path, sigma, target_name, date = get_flat_params_from_config(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Config error: {e}")
        return 1
    
    # Load master dark
    print(f"Loading master dark from: {master_dark_path}")
    try:
        with fits.open(master_dark_path) as hdul:
            masterdark = hdul[0].data
            dark_std = hdul["STD"].data if "STD" in hdul else np.zeros_like(masterdark)
            dark_mask = hdul["MASK"].data.astype(bool) if "MASK" in hdul else np.zeros_like(masterdark, dtype=bool)
    except FileNotFoundError:
        print(f"Error: Master dark not found at {master_dark_path}")
        print("Generate it first with: python make_dark.py --config <config>")
        return 1
    
    # Combine flats
    try:
        raw_masterflat, raw_std = combine_flats(
            flats_dir,
            masterdark,
            min_frames=MIN_FRAMES,
            relaxed=args.relaxed
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1
    
    # Trace edges
    trace_filter = FILTER
    trace_orders = orders_param

    # For NIRSPEC1_70, order 70 is typically the 3rd order from the end.
    if orders_param is None and FILTER == nirspec.NIRSPEC1_70:
        trace_filter = None
        trace_orders = -3
        print("Auto-selecting order -3 (third from end) for NIRSPEC1_70")

    top_edges, bottom_edges = trace_edges(
        raw_masterflat,
        FILTER=trace_filter,
        show_plots=not args.no_plots,
        orders=trace_orders
    )
    
    # Interpolate bad pixels from dark
    is_bad_pixel = np.array(dark_mask, dtype=bool)
    masterflat = interpolate_missing(raw_masterflat, is_bad_pixel)
    
    # Fit and divide out smooth variations
    print("Fitting and dividing smooth variations...")
    masterflat, normalization = fit_and_divide(masterflat, top_edges, bottom_edges)
    
    # Create final mask
    mask = np.zeros(masterflat.shape, dtype=bool)
    mask[masterflat < FLAT_MIN] = True
    mask[masterflat > FLAT_MAX] = True
    mask[:, -RIGHT_MARGIN:] = True
    mask = np.logical_or(is_bad_pixel, mask)
    
    n_masked = np.sum(mask)
    total = np.prod(mask.shape)
    print(f"Final mask: {n_masked} pixels ({100*n_masked/total:.2f}%)")
    
    # Show final result
    if not args.no_plots:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        im1 = axes[0].imshow(masterflat, aspect='auto', origin='lower', vmin=0, vmax=1.5, cmap='gray')
        axes[0].set_title("Master Flat")
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(normalization, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title("Normalization")
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(mask, aspect='auto', origin='lower', cmap='gray')
        axes[2].set_title("Mask")
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        fig.savefig(f"./{target_name}_{date}/cals/_masterflat_summary.png", dpi=600)
        plt.show()
       
    
    # Save result
    save_master_flat(
        masterflat, is_bad_pixel, (top_edges, bottom_edges),
        mask, raw_masterflat, raw_std, normalization,
        target_name, date,
        output_dir=f"{target_name}_{date}/cals", 
    )
    
    print(" Master flat generation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())