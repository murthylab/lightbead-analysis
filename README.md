Data analysis for Gauthey et al. 2025 [1]. The data can be found at ......
This repository contains the code to reproduce the figures in [1] and a signal extraction pipeline.

# Dependencies
To run the analysis code you will need scipy, numpy, pandas, scikit-learn.
For the signal extraction pipeline you will need ANTsPy (https://github.com/ANTsX/ANTsPy.git).
For visualizing ROIs on slices you will need the packages included in roi_plotting_requirements.txt.

# Signal extraction pipeline
To extract calcium activity from ROIs you need to follow these steps:
1) Run Sliding_window.m on raw TIFF files to have the data in the right ordering
2) Run batch_tiff_to_dff_mean_brain_RigE.sh to create a mean brain
3) Run batch_tiff_to_dff_mc_RigE.sh to compute motion correction
4) Run batch_tiff_to_dff_concat_RigE.sh
5) Run batch_signal_to_supervoxel_roi_RigE.sh to extract signal

# Reference
[1] Gauthey W., Lin A., Ahmed O.M., Leifer A.M., Murthy M., Thiberge S.Y., 2025

# Credits
Albert Lin, Wayan Gauthey, Sama Ahmed
