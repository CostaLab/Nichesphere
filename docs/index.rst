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


Citation
--------

If you use ``NicheSphere`` in your research, please cite our paper:

   **NicheSphere reveals Spp1⁺ macrophages as central hubs coordinating fibrotic remodeling in myeloproliferative neoplasms**

   Hélène F.E. Gleitz, Mayra L. Ruiz Tejada Segura, James S. Nagai, Stijn N.R. Fuchs, Gerjanne Vroeg in de Wei, Inge A.M. Snoeren, Giulia Cesaro, Iris J. Bakker, Marta Gargallo Garasa, Jessica E. Pritchard, Tesa Klenovšek, Stephani Schmitz, Lina Schmidt, Eric Bindels, Twan Lammers, Joost Gribnau, Hind Medyouf, Kai Markus Schneider, Carolin V. Schneider, Rafael Kramann, Ivan G. Costa, Rebekka K. Schneider.
   
   *bioRxiv* 2026.03.16.711605
   
   `https://doi.org/10.64898/2026.03.16.711605 <https://doi.org/10.64898/2026.03.16.711605>`_


.. code-block:: bibtex

   @misc{gleitz_nichesphere_2026,
      title = {{NicheSphere} reveals {Spp1}⁺ macrophages as central hubs coordinating fibrotic remodeling in myeloproliferative neoplasms},
      author = {Gleitz, Hélène F. E. and Segura, Mayra L. Ruiz Tejada and Nagai, James S. and Fuchs, Stijn N. R. and Wei, Gerjanne Vroeg in de and Snoeren, Inge A. M. and Cesaro, Giulia and Bakker, Iris J. and Garasa, Marta Gargallo and Pritchard, Jessica E. and Klenovšek, Tesa and Schmitz, Stephani and Schmidt, Lina and Bindels, Eric and Lammers, Twan and Gribnau, Joost and Medyouf, Hind and Schneider, Kai Markus and Schneider, Carolin V. and Kramann, Rafael and Costa, Ivan G. and Schneider, Rebekka K.},
      publisher = {bioRxiv},
      year = {2026},
      month = mar,
      doi = {10.64898/2026.03.16.711605},
      url = {https://www.biorxiv.org/content/10.64898/2026.03.16.711605v1}
   }

Tutorials
---------
.. Bullet points shown on the home page

* :doc:`NicheSphere tutorial — Myocardial Infarction (Visium) </notebooks/Nichesphere_tutorial_MIvisium>`
* :doc:`NicheSphere × PILOT: niche-based trajectory inference </notebooks/Nichesphere_tutorial_MIvisium_PILOT>`



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