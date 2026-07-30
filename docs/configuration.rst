Configuration
==============

The ``pipeline_config.ini`` file controls all pipeline parameters.

Target Section
--------------

.. code-block:: ini

    [Target]
    name = TOI2123
    date = 240704
    alt_name = TOI-2123

Stellar Parameters
-------------------

Get these values from SIMBAD:

.. code-block:: ini

    [Stellar Parameters]
    V_SYS = -26.64e3      # Systemic velocity (m/s)
    logg = 4.56           # Surface gravity (cgs)
    Teff = 5432           # Effective temperature (K)
    Z = 0                 # Metallicity

Planet Parameters
------------------

From ExoFOP, NASA Exoplanet Archive, or literature:

.. code-block:: ini

    [Planet Parameters]
    name = b              # a, b, c, etc
    T_c = 59691.31982     # Time of conjunction (MJD)
    T_dur = 3.473         # Transit duration (hours)
    P = 10.3088372        # Orbital period (days)
    K_p = 96.177e3        # Keplerian velocity (m/s)
    
Calibrations
------------
Specify the path of the master calibration files:
If work on the same night, you can reuse the master calibration files from that night, just specifiy the path

.. code-block:: ini

    [Calibrations]
    master_dark_parth =  
    master_flat_path =  

Pixel Masking
-------------

Select the sigma threshold for pixel masking:

.. code-block:: ini

    [Pixel Masking]
    Dark  = 5
    Flat  = 5
    cosmic_ray_sigma = 5

Filters
-------
Select which filter and order to use for the analysis:
Options are: 
- NIRSPEC1_70
    - Order 70 of NIRSPEC1 (old or new y-band)

- NIRSPECHEI_1
    - He narrowband filter 

.. code-block:: ini

    [Filters]
    filter = NIRSPEC1_70 