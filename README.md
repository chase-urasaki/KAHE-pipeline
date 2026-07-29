# KAHE Pipeline

Calibration and analysis tools for Keck/NIRSPEC spectroscopy.

## Installation

### 1. Install Package 

#### From Source (development)

```bash
git clone https://github.com/chase-urasaki/KAHE-pipeline.git
cd KAHE-pipeline
pip install -e .
```
#### From PyPI 
```bash
pip install kahe
```
## Quick Start 

### 1. Create a new project
This creates a project directory with the required folder structure and a blank configuration file: 

```bash
kahe-setup {targetname}_{obsdate}
``` 
This generates the following: 
TOI2123_240704/
├── pipeline_config.ini           # Edit this with your parameters
├── cals/                         # Master calibration files which you can reuse for the same night
├── raw/                          # Raw FITS data
├── extracted_spectra/            # 1D extracted spectra
├── wl_calibrated/                # Wavelength-calibrated spectra
├── telluric_correction_inputs/   # Inputs for telluric correction
└── telluric_correction_out/      # Telluric-corrected output

### 2. Organize Raw Data 
raw/
├── darks/
│   ├── dark_0001.fits
│   ├── dark_0002.fits
│   └── ...
├── flats/
│   ├── flat_0001.fits
│   ├── flat_0002.fits
│   └── ...
└── science/
    ├── science_0001.fits
    ├── science_0002.fits
    └── ...
