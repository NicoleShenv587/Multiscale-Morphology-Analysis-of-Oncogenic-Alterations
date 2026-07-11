**Multiscale Morphology Analysis of Oncogenic Alterations**

This repository contains the computational pipeline used to analyze three-dimensional expansion microscopy data for quantitative multiscale structural phenotyping of hepatocytes. The workflow includes cell and mitochondrial segmentation, global cell morphology analysis, membrane curvature quantification, mitochondrial morphology and skeleton analysis, and multiscale feature integration for oncogenic phenotype classification.
If you use this code, please cite our paper:
**Deciphering the Multiscale Morphology of Somatic Oncogenic Alterations in Hepatocellular Carcinoma**

Detailed documentation, installation instructions, example datasets, and parameter descriptions are available in the project documentation.

As a quick overview:

**Cell Segmentation**

Contains the workflow for three-dimensional single-cell segmentation from expansion microscopy images. 

**Mitochondria Segmentation**

Contains the workflow for segmentation of individual mitochondria from volumetric fluorescence images.

**Global Cell Morphology**

Contains code for quantifying global three-dimensional cell morphology, including cell volume, surface area, elongation, sphericity, and other morphometric descriptors. 

**Curvature Analysis**

Contains code for generating cell-surface meshes and quantifying membrane-curvature features, including vertex-wise mean curvature and high-curvature statistics. 

**Mitochondria Analysis**

Contains code for extracting three-dimensional mitochondrial morphology features, including size, shape, elongation measurements, mitochondrial skeletonization, and network topology quantification.

**Classification**

Contains code for feature preprocessing, pseudo-bulk feature integration, cross-validation, and multiscale classification of oncogenic phenotypes.

**Visualization**

Contains scripts for figure generation used in the manuscript.

**System**

The pipeline has been tested on Linux and Windows using Python 3.10. Required software dependencies are listed in *requirements.txt* and *environment.yml*.

***Additional Dependencies***

The segmentation and curvature analysis rely on:

- u-Segment3D
- u-Unwrap3D

Please install these packages following the instructions provided in their official repositories before running the corresponding analysis scripts.

No specialized hardware is required for feature extraction and analysis, although GPU acceleration may improve the performance of segmentation workflows on large volumetric datasets.


<img width="880" height="286" alt="image" src="https://github.com/user-attachments/assets/6d35f518-2d09-4842-a436-cdfa94e804c8" />


