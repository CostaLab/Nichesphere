.. Nichesphere documentation master file, created by
   sphinx-quickstart on Mon Feb  3 18:13:32 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NicheSphere's documentation!
=======================================

.. image:: _static/logo.png
   :alt: NicheSphere Logo
   :align: right
   :width: 180px
   :class: logo-header


**NicheSphere** is an sc-verse compatible Python library which allows the user to find differential 
co-localization domains / niches based on cell type pair co-localization probabilities in different 
conditions. Cell type pair co-localization probabilities can be obtained in different ways, 
for example, through deconvolution of spatial transcriptomics / PIC-seq data 
(getting the probabilities of finding each cell type in each spot / multiplet) ; 
or counting cell boundaries overlaps for each cell type pair in single cell spatial data (MERFISH , CODEX ...).

It also offers the possibility to look at biological process based differential communication among differential 
co-localization domains based on Ligand-Receptor pairs expression data, such as results from `pyCrossTalkeR <https://pycrosstalker.readthedocs.io/>`_.


.. Left bar navigation structure: 
   Hidden to control sidebar hierarchy and avoid duplicating titles on the home page

.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   tutorials
   api

Installation
------------
.. References generated from custom labels defined in installation.rst
* :ref:`installation_prerequisites`
* :ref:`installation_standard`
* :ref:`installation_github`

Tutorials
---------
.. Bullet points shown on the home page
* `NicheSphere tutorial — Myocardial Infarction (Visium) <notebooks/Nichesphere_tutorial_MIvisium.ipynb>`_
* `NicheSphere × PILOT: niche-based trajectory inference <notebooks/Nichesphere_tutorial_MIvisium_PILOT.ipynb>`_


Docker image and summarized analysis tutorial
---------------------------------------------

We provide access to a Docker image, available at: https://gitlab.com/sysbiobig/ismb-eccb-2025-tutorial-vt3/container_registry. 
The Docker image comes preconfigured with all necessary libraries, tools, and software required to follow the hands-on exercises. 
Additionally, the repository at https://gitlab.com/sysbiobig/ismb-eccb-2025-tutorial-vt3 contains a summarized Nichesphere 
co-localization + communication analysis tutorial.

API Reference
-------------
.. References generated from custom labels defined in coloc.rst, comm.rst, tl.rst, niche_stats and api_sankey_plot`
* :ref:`api_coloc`
* :ref:`api_comm`
* :ref:`api_tl`
* :ref:`api_niche_stats`
* :ref:`api_sankey_plot`



