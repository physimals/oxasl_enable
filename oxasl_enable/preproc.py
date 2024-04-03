#!/usr/bin/env python
"""
ASL_PREPROC: Preprocessing of ASL images tog et data into correct form for Oxford_ASL and BASIL

Michael Chappell & Brad MacIntosh, FMRIB Image Analysis & Physics Groups

Copyright (c) 2018 University of Nottingham
"""

import os, sys
import shutil
from optparse import OptionParser, OptionGroup

import numpy as np

import fslhelpers as fsl

usage = """ASL_PREPROC"

Assembles Multi-TI ASL images into correct form for Oxford_ASL and BASIL"
"""

def preproc(infile, nrp, nti, outfile, smooth=False, fwhm=6, mc=False, log=sys.stdout):
    log.write("Preprocessing with multi-TI ASL data\n")

    # Total number of TC pairs and volumes
    npairs=nti*nrp
    input_nvols = fsl.Image(infile).shape[3]
    
    log.write("Working on data %s\n" % infile)
    log.write("Number of TIs                 : %i\n" % nti)
    log.write("Number of repeats             : %i\n" % nrp)
    log.write("Total number of TC pairs      : %i\n" % npairs)
    log.write("Input data number of volumes  : %i\n" % input_nvols)

    fsl.mkdir("asl_preproc_temp")
    fsl.imcp(infile, "asl_preproc_temp/stacked_asaq")
    
    ntc = 2
    if npairs == input_nvols:
        log.write("Data appears to be differenced already - will not perform TC subtraction\n")
        ntc = 1
    elif npairs*2 > input_nvols:
        raise Exception("Not enough volumes in input data (%i - required %i)\n" % (input_nvols, npairs*2))
    elif npairs*2 < input_nvols:
        log.write("Warning: Input data contained more volumes than required (%i - required %i) - some will be ignored" % (input_nvols, npairs*2))

    if mc:
        log.write("Warning: Motion correction is untested - check your results carefuly!\n")
        fsl.Prog("mcflirt").run("-in asl_preproc_temp/stacked_asaq -out asl_preproc_temp/stacked_asaq -cost mutualinfo")

    stacked_nii = fsl.Image("asl_preproc_temp/stacked_asaq")
    stacked_data = stacked_nii.data()
    input_nvols = stacked_data.shape[3]

    log.write("Assembling data for each TI and differencing\n")

    shape_3d = list(stacked_data.shape[:3])
    diffs_stacked = np.zeros(shape_3d + [npairs])
    for ti in range(nti):
        diffs_ti = np.zeros(shape_3d + [nrp])
        for rp in range(nrp):
            v = ntc*rp*nti + ntc*ti
            if ntc == 2:
                tag = stacked_data[:,:,:,v]
                ctrl = stacked_data[:,:,:,v+1]
                diffs_ti[:,:,:,rp] = tag - ctrl
            else:
                diffs_ti[:,:,:,rp] = stacked_data[:,:,:,v]

        diffs_stacked[:,:,:,ti*nrp:(ti+1)*nrp] = diffs_ti
    
    # take the mean across the repeats
    diffs_mean = np.zeros(shape_3d)
    diffs_mean[:,:,:] = np.mean(diffs_stacked, 3)

    log.write("Assembling stacked data file\n")

    stacked_nii.new_nifti(diffs_stacked).to_filename("%s.nii.gz" % outfile)
    stacked_nii.new_nifti(diffs_mean).to_filename("%s_mean.nii.gz" % outfile)

    log.write("ASL data file is: %s\n" % outfile)
    log.write("ASL mean data file is: %s_mean\n" % outfile)

    if smooth:
        # Do spatial smoothing
        sigma = round(fwhm/2.355, 2)
        log.write("Performing spatial smoothing with FWHM: %f (sigma=%f)\n" % (fwhm, sigma))
        fsl.maths.run("%s -kernel gauss %f -fmean %s_smooth" % (outfile, sigma, outfile))
        fsl.maths.run("%s_mean -kernel gauss %f -fmean %s_smooth_mean" % (outfile, sigma, outfile))

        log.write("Smoothed ASL data file is: %s_smooth" % outfile)
        log.write("Smoothed ASL mean data file is: %s_smooth_mean" % outfile)

    shutil.rmtree("asl_preproc_temp")
    log.write("DONE\n")

if __name__ == "__main__":
    try:
        p = OptionParser(usage=usage, version="@VERSION@")
        p.add_option("-i", dest="infile", help="Name of (stacked) ASL data file")
        p.add_option("--nrp", dest="nrp", help="Number of repeats", type="int")
        p.add_option("--nti", dest="nti", help="Number of TIs in data", type="int")
        p.add_option("-o", dest="output", help="Output name", default="asldata")
        p.add_option("-s", dest="smooth", help="Spatially smooth data", action="store_true")
        p.add_option("--fwhm", dest="fwhm", help="FWHM for spatial filter kernel", type="float", default=6)
        p.add_option("-m", dest="mc", help="Motion correct data", action="store_true", default=False)
        options, args = p.parse_args(sys.argv)

        preproc(options.infile, options.nrp, options.nti, options.outfile, optinos.smooth, optinos.fwhm, options.mc)
    except Exception as e:
        print("ERROR: " + str(e) + "\n")


