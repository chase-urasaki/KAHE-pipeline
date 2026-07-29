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
    logg = 4.56           # Surface gravity
    Teff = 5432           # Effective temperature (K)
    Z = 0                 # Metallicity

Planet Parameters
------------------

From ExoFOP or NASA Exoplanet Archive:

.. code-block:: ini

    [Planet Parameters]
    name = b
    T_c = 59691.31982     # Time of conjunction (MJD)
    T_dur = 3.473         # Transit duration (hours)
    P = 10.3088372        # Orbital period (days)
    K_p = 96.177e3        # Keplerian velocity (m/s)