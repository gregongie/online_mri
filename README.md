# Online Dynamic MRI reconstruction with low-rank plus sparse model

MATLAB code to implement the RUFFed GROUSE algorithm described in the paper:

> Ongie, G., Dewangan, S., Fessier, J. A., & Balzano, L. (2017). 
> Online dynamic MRI reconstruction via robust subspace tracking. 
> Proceedings of IEEE GlobalSIP (pp. 1180-1184).

The main file is `demo.m`, which performs online (frame-by-frame) reconstruction on a
phantom dataset using the GROUSE and RUFFed GROUSE algorithms.

Requires the Michigan Iterative Reconstruction Toolbox (MIRT) as a dependency:
https://web.eecs.umich.edu/~fessler/code/

## Version history
* Version 0.1, Updated 1/30/2019

## Author
Greg Ongie ([website](https://gregongie.github.io)) 

## Acknowledgments
The dataset used in the demo is derived from the MRXCAT phantom 
http://www.biomed.ee.ethz.ch/mrxcat.html
