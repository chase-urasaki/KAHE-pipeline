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

Copy FITS files into the ``raw/`` directory:

.. code-block:: text

    raw/
    ├── darks/
    ├── flats/
    └── science/

Configure the Pipeline
----------------------

Edit ``pipeline_config.ini`` with your observation parameters.

Process Calibrations
--------------------

Create master dark and flat frames:

.. code-block:: bash

    kahe-make-dark pipeline_config.ini
    kahe-make-flat pipeline_config.ini

Extract Spectra
---------------

Extract 1D spectra from 2D frames:

.. code-block:: bash

    kahe-extract pipeline_config.ini