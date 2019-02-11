# ENABLE - ENhancement of Automated Blood fLow Estimates 

ENABLE is a technique to evaulate image quality in multi-PLD ASL data. 

*Shirzadi et al, "Enhancement of Automated Blood Flow Estimates (ENABLE) From
Arterial Spin-Labelled MRI" J Magn Reson Imaging. 2017 Jul 6. doi: 10.1002/jmri.25807*

ENABLE performs the following steps:

1. A **preprocessing** step, consisting of tag/control differencing, brain 
extraction, T1 co-registration and segmentation

2. A **sorting** step in which quality measures are calculated for each 
PLD image, namely:
  - Contrast:Noise ratio in GM (CNR)
  - Detectibility metric (DETECT, proportion of voxels with significantly greater than zero signal in GM)
  - Coefficient of variation of difference data (COV, spatial standard deviation divided by spatial mean in GM)
  - Temporal SNR (TSNR, spatial mean of ASL image divided by temporal STD image in GM)

3. A **selection** step in which a combined quality measure is calculated and the best image is identified:
  - 0.1*CNR + 1.8*DETECT - COV - TSNR
 
4. A *CBF generation* step in which a CBF image is generated from the best image

## Usage

    oxasl_enable -i <ASL input file> -t1 <T1 image> -n <Noise ROI image> -o <Output dir>

oxasl_enable can also be used as a plug-in for the OXASL ASL processing pipeline - see documentation
for this tool at:

https://oxasl.readthedocs.io/

### Options

    --version      show program's version number and exit
    -h, --help     show this help message and exit
    -i INFILE      ASL data file
    --t1=T1        T1 map
    -n NOISE       Noise ROI
    -o OUTPUT      Output dir
    --debug=DEBUG  Debug mode
