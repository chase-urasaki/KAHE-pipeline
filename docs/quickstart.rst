Quick Start
===========

Create a Project
----------------

Initialize a new project:

.. code-block:: bash

    kahe-setup TOI2123_240704

This creates a project directory with the required structure.

Organize Your Data
------------------

Copy FITS files into the ``raw/`` directory and use the following structure:

.. code-block:: text

    raw/
    ├── darks/
    ├── flats/
    └── science/

Configure the Pipeline
----------------------

Edit ``pipeline_config.ini`` with your observation parameters.
Fill in first the paths to the flats and darks, 

Process Calibrations
--------------------

Create master dark and flat frames:

.. code-block:: bash

    kahe-make-dark pipeline_config.ini

Once the dark is created, you can create the master flat frame:
Specifiy the dark frame path in the ``pipeline_config.ini`` file and run:

.. code-block:: bash

    kahe-make-flat pipeline_config.ini

It may be the case that your flat looks something like this: 

This means that the bounds are no longer set correctly. Perfectly normal, as it happens when the the echelle and cross disperser are not in the same position as before. You can fix this by going into the ``fix_flat`` notebook.
When this happens, you will need to re-run the ``kahe-make-flat`` command after fixing the flat. and change the value in the nirspec.py file 
TODO: fix this so it can be set in the config file.

Extract Spectra
---------------

Extract 1D spectra from 2D frames:

.. code-block:: bash

    kahe-extract pipeline_config.ini