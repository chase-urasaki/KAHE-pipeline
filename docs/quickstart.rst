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

Extract Spectra
---------------

Extract 1D spectra from 2D frames:

.. code-block:: bash

    kahe-extract pipeline_config.ini