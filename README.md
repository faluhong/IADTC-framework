# IADTC-framework

The demo code of the IADTC framework used to generate the global spatiotemporally seamless daily mean land surface temperature. 

The major steps of the IADTC framework include:   
(1) using the multi-type ATC model to reconstruct the under-cloud LST for each MODIS overpass time.   
(2) using the linear interpolation to fill the NaN values in the overpass time series.   
(3) estimating the daily mean land surface with the DTC model.   

Here is the stand-alone version developed based on python 3.8 to display the workflow of the IADTC framework. The global spatiotemporally seamless daily mean land surface temperature product from 2003 to 2019 is publicly available at: https://doi.org/10.5281/zenodo.6287052.

You may refer to the following papers for the details.

[1] Hong, F., Zhan, W., Göttsche, F.M., Liu, Z., Dong, P., Fu, H., Huang, F., & Zhang, X. (2022). A global dataset of spatiotemporally seamless daily mean land surface temperatures: generation, validation, and analysis. Earth System Science Data, 14, 3091-3113. https://essd.copernicus.org/articles/14/3091/2022/. 

[2] Hong, F., Zhan, W., Göttsche, F.-M., Lai, J., Liu, Z., Hu, L., Fu, P., Huang, F., Li, J., Li, H., & Wu, H. (2021). A simple yet robust framework to estimate accurate daily mean land surface temperature from thermal observations of tandem polar orbiters. Remote Sensing of Environment, 264, 112612. https://www.sciencedirect.com/science/article/pii/S0034425721003321.

If you have any questions, feel free to contact me (hongfalu@foxmail.com). 
