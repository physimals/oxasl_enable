#!/usr/bin/env python
"""
OXASL_ENABLE
============

Enhancement of Automated Blooed Estimates for ASL-MRI

ENABLE is a tool designed to assess quality of images in multi-repeat
ASL data and discard repeats of low quality with the aim of increasing
the overall quality of the data set.

OXASL_ENABLE is a Python implementation working within the OXASL
framework, and is hence based around the AslImage and Workspace classes.

A command line tool is provided (``oxasl_enable``), also the API may be
used directly. The main method is ``enable`` - see this method's 
documentation for the workspace attributes that should be set.

Copyright (c) 2018 University of Oxford
"""
from __future__ import print_function

import os
import sys
import math

import numpy as np
import pandas as pd
import scipy.stats

from fsl.data.image import Image

from oxasl import Workspace, reg, struc, image
from oxasl.options import AslOptionParser, OptionCategory, IgnorableOptionGroup, GenericOptions
from oxasl.reporting import Report

from ._version import __version__, __timestamp__

def get_rois(wsp):
    """
    Generate ROIs for GM and noise
    
    Existing ROIs may be provided. If provided in structural space, the ASL data is registered
    to the structural image and the ROIS will be transformed to ASL space

    If (either) of the noise or GM ROIs are not provided they are automatically generated as follows

     - Grey matter: From segmentation of structural image (via FAST or FSL_ANAT output if provided)
     - Noise: By inverting brain mask from brain-extracted structural image

    A structural image is therefore required if either ROI needs to be generated, or if either ROI
    is supplied in structural space. 

    Optional workspace attributes
    -----------------------------

     - ``regfrom`` : Reference image for registration of ASL to structural data
     - ``asldata`` : Single TI ASL data. Middle volume will be used as reference if ``regfrom`` is required and not set
     - ``gm_roi`` : Grey matter ROI in ASL or structural space
     - ``noise_roi`` : Noise ROI in ASL or structural space
     - ``noise_from_struc`` : If True, existing ``noise_roi`` image is in structural space
     - ``gm_from_struc`` : If True, existing ``gm_roi`` image is in structural space
     - ``struc`` : Structural image
     
    Workspace attributes set
    ------------------------

     - ``gm_roi`` : Grey matter ROI in ASL space
     - ``noise_roi`` : Noise ROI in ASL space
    """
    wsp.log.write("Generating ROIs...\n")

    if wsp.regfrom is None:
        wsp.log.write(" - Reference image for registration not provided - using ASL data middle volume\n")
        middle = int(wsp.asldata.shape[3]/2)
        wsp.regfrom = Image(wsp.asldata.data[:, :, :, middle], header=wsp.asldata.header)
        
    struc.init(wsp)
    if (wsp.gm_roi is None or wsp.noise_roi is None or wsp.noise_from_struc or wsp.gm_from_struc) and wsp.structural.struc is None:
        raise RuntimeError("Need to specify a structural image unless you provide both noise and GM ROIs in ASL space")

    if wsp.gm_roi is None:
        struc.segment(wsp)
        wsp.log.write("Taking GM ROI from segmentation of structural image\n")
        wsp.gm_roi = Image((wsp.structural.gm_pv.data > 0).astype(np.int), header=wsp.structural.struc.header)
        wsp.gm_from_struc = True

    if wsp.noise_roi is None:
        wsp.log.write("Generating noise ROI by inverting T1 brain mask\n")
        wsp.noise_roi = Image(1-wsp.structural.brain_mask.data, header=wsp.structural.struc.header)
        wsp.noise_from_struc = True

    if wsp.noise_from_struc or wsp.gm_from_struc:
        # Need struc->ASL registration space so we can apply to noise and/or GM ROIs
        wsp.do_flirt = True
        wsp.do_bbr = False
        reg.reg_asl2struc(wsp)

    if wsp.noise_from_struc:
        wsp.log.write(" - Registering noise ROI to ASL space since it was defined in structural space\n\n")
        wsp.noise_roi_struc = wsp.noise_roi
        wsp.noise_roi = reg.struc2asl(wsp, wsp.noise_roi_struc, interp="nn")

    if wsp.gm_from_struc:
        wsp.log.write(" - Registering GM ROI to ASL space since it was defined in structural space\n\n")
        wsp.gm_roi_struc = wsp.gm_roi
        wsp.gm_roi = reg.struc2asl(wsp, wsp.gm_roi_struc, interp="nn")

    wsp.log.write("DONE\n\n")

def tsf(df, t):
    """ 
    Survival function (1-CDF) of the t-distribution

    Heuristics to agree with FSL ttologp but not properly justified
    """
    if t < 0:
        return 1-tsf(df, -t)
    elif t > 700:
        return scipy.stats.t.sf(t, df)
    else:
        return 1-scipy.special.stdtr(df, t)

def calculate_cnr(wsp, gm_roi, noise_roi):
    """
    Calculate CNR (Contrast:Noise ratio) for each repeat in single-TI ASL data

    CNR is defined as mean difference signal divided by nonbrain standard deviation

    :param gm_roi: Grey-matter mask
    :param noise_roi: Noise mask (typically all non-brain parts of the image)
    
    Required workspace attributes
    -----------------------------

     - ``asldata`` : Single TI ASL data
     
    Workspace attributes set
    ------------------------

     - ``cnrs`` : Sequence of CNR values, one for each repeat
    """
    wsp.log.write("Calculating CNR for each repeat...")
    
    tdim = wsp.asldata.shape[3]
    wsp.cnrs = []
    for i in range(tdim):
        vol_data = wsp.asldata.data[:, :, :, i].astype(np.float32)
        meanGM = np.mean(vol_data[gm_roi.data > 0])
        noisestd = np.std(vol_data[noise_roi.data > 0])
        cnr = meanGM/noisestd
        if cnr < 0:
            wsp.log.write("WARNING: CNR was negative - are your tag-control pairs the right way round?")
        wsp.cnrs.append(cnr)
    wsp.log.write("DONE\n\n")

def sort_cnr(wsp):
    """
    Sort ASL data by decreasing CNR
    
    Required workspace attributes
    -----------------------------

     - ``asldata`` : Single TI ASL data
     - ``cnrs`` : Sequence of CNR value for each repeat
     
    Workspace attributes set
    ------------------------

     - ``asldata_sorted`` : Single TI ASL data, with repeats sorted by CNR
     - ``cnrs_sorted`` : Sorted sequence of CNR tuples: (source repeat index, CNR)
    """
    wsp.log.write("Sorting repeats by CNR\n\n")
    wsp.cnrs_sorted = sorted(enumerate(wsp.cnrs), key=lambda x: x[1], reverse=True)

    # Create re-ordered data array
    wsp.log.write("Repeat number\tCNR\n")
    sorted_data = np.zeros(wsp.asldata.shape)
    for idx, cnr in enumerate(wsp.cnrs_sorted):
        sorted_data[:, :, :, idx] = wsp.asldata.data[:, :, :, cnr[0]].astype(np.float32)
        wsp.log.write("%i\t%.3f\n" % (cnr[0], cnr[1]))
        
    page = wsp.report.page("cnrs")
    page.heading("Repeats ordered by CNR")
    page.table(wsp.cnrs_sorted, headers=["Repeat number", "Contrast:Noise ratio"])
    
    wsp.asldata_sorted = wsp.asldata.derived(sorted_data, suffix="_sorted")
    wsp.log.write("\nDONE\n\n")

def calculate_quality_measures(wsp, gm_roi, noise_roi):
    """
    Calculate quality measures on the data subset obtained by
    cumulatively including repeats from single-TI ASL data.
    
    :param gm_roi: Grey matter ROI in ASL space
    :param noise_roi: Noise ROI in ASL space

    Required workspace attributes
    -----------------------------

     - ``asldata_sorted`` : Single TI ASL data, with repeats sorted by CNR
     - ``min_nvols`` : Minimum number of repeats to include
     
    Workspace attributes set
    ------------------------

     - ``qms`` : Quality measures obtained by cumulatively including each
                 repeat sequentially. Mapping from measure name to
                 sequence of values.
    """
    wsp.log.write("Calculating quality measures...\n")
    if wsp.min_nvols < 2:
        raise RuntimeError("Need to keep at least 2 repeats to calculate quality measures")

    tdim = wsp.asldata_sorted.shape[3]
    gm_roi = gm_roi.data
    noise_roi = noise_roi.data
    num_gm_voxels = np.count_nonzero(gm_roi)

    wsp.log.write("Repeats\ttCNR\tDETECT\tCOV\ttSNR\n")
    qms = {"tcnr" : [], "detect" : [], "cov" : [], "tsnr" : []}
    report_table = []

    for i in range(wsp.min_nvols, tdim+1, 1):
        temp_data = wsp.asldata_sorted.data[:, :, :, :i]

        mean = np.mean(temp_data, 3)
        std = np.std(temp_data, 3, ddof=1)

        # STD = 0 means constant data across volumes, do something sane
        std[std == 0] = 1
        mean[std == 0] = 0

        snr = mean / std
        serr = std / math.sqrt(float(i))
        tstats = mean / serr

        # Annoyingly this is slower than using ttologp. scipy.special.stdtr
        # is fast but not accurate enough. Need to understand exactly what 
        # this is doing, however, because it seems to rely on 'anything below
        # float32 minimum == 0'
        calc_p = np.vectorize(lambda x, vol=i: tsf(vol, x))
        p1 = calc_p(tstats).astype(np.float32)
        p1[p1 > 0.05] = 0
        sigvox2 = p1
        sigvoxGM2 = np.count_nonzero(sigvox2[gm_roi > 0])

        DetectGM = float(sigvoxGM2)/num_gm_voxels
        tSNRGM = np.mean(snr[gm_roi > 0])
        meanGM = np.mean(mean[gm_roi > 0])
        stdGM = np.std(mean[gm_roi > 0], ddof=1)
        CoVGM = 100*float(stdGM)/float(meanGM)

        noisestd = np.std(mean[noise_roi > 0], ddof=1)
        SNRGM = float(meanGM)/float(noisestd)
        qms["tcnr"].append(SNRGM)
        qms["detect"].append(DetectGM)
        qms["cov"].append(CoVGM)
        qms["tsnr"].append(tSNRGM)
        wsp.log.write("%i\t%.3f\t%.3f\t%.3f\t%.3f\n" % (i, SNRGM, DetectGM, CoVGM, tSNRGM))
        report_table.append([i, SNRGM, DetectGM, CoVGM, tSNRGM])
    wsp.qms = qms
    wsp.log.write("DONE\n\n")  
        
    page = wsp.report.page("qms")
    page.heading("Cumulative Quality measures by included repeats")
    page.table(report_table, headers=["Number of repeats", "SNRGM", "DetectGM", "CoVGM", "tSNRGM"])

def get_combined_quality(wsp, ti, b0="3T"):
    """
    Calculate combined quality scores of subsets of single-TI ASL data
    obtained by cumulatively including repeats.
    
    :param ti: TI value
    :param b0: Field strength: '3T' or '1.5T'

    Required workspace attributes
    -----------------------------

     - ``qms`` : Individual quality measures for subsets of data obtained by cumulatively
                 including repeats. In form of a dict mapping measure name to a sequence 
                 of values.

    Workspace attributes set
    ------------------------

     - ``quality`` : Sequence giving the total quality achieved by cumulatively including 
                     each repeat
    """
    # Weightings from the ENABLE paper, these vary by PLD
    sampling_plds = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]
    coef = {
        "3T" : {
            "tcnr" : [0.4, 0.3, 0.7, 0.1, 0.5, 0.3],
            "detect" : [1.6, 1.7, 1.0, 1.8, 1.4, 1.8],
            "cov" : [-0.9, -0.9, -0.3, -1.0, -0.8, -0.6],
            "tsnr" : [-1.1, -1.0, -1.1, -1.0, -1.0, -0.8],
        },
        "1.5T" : {
            "tcnr" : [0.5, 0.7, 0.7, 0.8, 0.2, 0.1],
            "detect" : [1.3, 1.2, 1.0, 0.8, 1.5, 1.6],
            "cov" : [-0.5, -0.7, -0.3, -0.4, -0.6, -0.7],
            "tsnr" : [-1.2, -1.0, -1.1, -1.0, -1.0, -1.0],
        }
    }

    if b0 not in coef:
        raise RuntimeError("We don't have data for B0=%s" % b0)
    coeffs = coef[b0]

    num_meas = len(wsp.qms["detect"])
    wsp.quality = np.zeros([num_meas], dtype=np.float)
    for meas, vals in wsp.qms.items():
        c = np.interp([ti], sampling_plds, coeffs[meas])

        # Try to ensure numerical errors do not affect the result
        vals[vals == np.inf] = 0
        vals[vals == -np.inf] = 0
        vals = np.nan_to_num(vals)

        normed = np.array(vals, dtype=np.float) / max(vals)
        wsp.quality += c * normed

    wsp.best_num_vols = int(np.argmax(wsp.quality) + wsp.min_nvols)
    wsp.maxqual = max(wsp.quality)

    wsp.log.write("Repeats\tOverall Quality\n")
    for idx, q in enumerate(wsp.quality):
        wsp.log.write("%i\t%.3f\n" % (idx, q))
        wsp.results[idx+wsp.min_nvols-1]["qual"] = q        
    wsp.log.write("Maximum quality %.3f with %i repeats\n" % (wsp.maxqual, wsp.best_num_vols))
        
    page = wsp.report.page("qual")
    page.heading("Cumulative overall quality by included repeats")
    page.text("Maximum overall quality achieved using %i repeats (quality score: %.3f)" % (wsp.best_num_vols, wsp.maxqual))
    page.table([(idx+wsp.min_nvols, qual) for idx, qual in enumerate(wsp.quality)], headers=["Number of repeats", "Overall Quality"])
   
def enable(wsp):
    """
    Remove volumes from a multi-repeat ASL data set to improve overall quality
    
    Required workspace attributes
    -----------------------------

     - ``asldata`` : Multi-repeat AslImage. Can be multi-TI data. TI value must
                     be provided (not just number)
     - ``gm_roi`` : Grey matter ROI in ASL space
     - ``noise_roi`` : Noise matter ROI in ASL space
     - ``min_nvols`` : Minimum number of repeats to include from each TI
     
    Workspace attributes set
    ------------------------

     - ``asldata_enable`` : AslImage with volumes potentially removed
     - ``enable_results`` : Sequence of results for each TI.
                            Each TI results object is itself a sequence of dictionaries
                            one for each repeat at this TI and sorted by contrast:noise ratio
                            (not in original repeat order). Each contains the keys
                            ``ti``, ``rpt`` (TI value and original repeat index), ``cnr`` Contrast:Noise ratio,
                            ``tcnr`` : ?, ``detect`` : ?, ``cov`` : ?, ``tsnr`` : Signal:Noise ratio,
                            ``qual`` : Overall quality of data set formed by including all repeats
                            up to and including this one. ``selected`` : True if the best quality data set
                            for this TI includes this repeat.
     - ``enable_ti_<val>`` : Sub-workspace containing output from each TI individually
    """
    get_rois(wsp)

    wsp.asldata_diff = wsp.asldata.diff().reorder("rt")
    wsp.enable_results = []
    ti_data = []

    # Process each TI in turn
    for ti_idx, ti in enumerate(wsp.asldata_diff.tis):
        wsp.log.write("\nProcessing TI: %i (%s)\n" % (ti_idx, str(ti)))
        
        # Create workspace for TI
        wsp_ti = wsp.sub("enable_ti_%s" % (ti_idx+1), report=Report())
        wsp_ti.asldata = wsp.asldata_diff.single_ti(ti_idx)
        wsp_ti.min_nvols = wsp.min_nvols
        wsp_ti.report.title = "Quality assessment for TI %i (%.2f)" % (ti_idx+1, ti)
        wsp_ti.results = []

        # Write out the mean differenced image for comparison
        wsp_ti.asldata_mean = wsp_ti.asldata.mean_across_repeats()

        # Sorting and calculation of quality measures
        calculate_cnr(wsp_ti, wsp.gm_roi, wsp.noise_roi)
        sort_cnr(wsp_ti)
        for orig_vol, cnr in wsp_ti.cnrs_sorted:
            wsp_ti.results.append(
                {"ti" : ti, "ti_idx" : ti_idx, "rpt" : orig_vol, "cnr" : cnr,
                 "tcnr" : 0.0, "detect" : 0.0, "cov" : 0.0, "tsnr" : 0.0, "qual" : 0.0}
            )

        calculate_quality_measures(wsp_ti, wsp.gm_roi, wsp.noise_roi)
        for meas in "tcnr", "detect", "cov", "tsnr":
            for idx, val in enumerate(wsp_ti.qms[meas]):
                wsp_ti.results[idx+wsp_ti.min_nvols-1][meas] = val
           
        # Get the image at which the combined quality measure has its maximum
        get_combined_quality(wsp_ti, ti)

        # Generate data subset with maximum quality
        for idx, result in enumerate(wsp_ti.results):
            result["selected"] = idx < wsp_ti.best_num_vols
            
        wsp_ti.asldata.summary(wsp.log)
        wsp_ti.asldata_enable = wsp_ti.asldata.derived(wsp_ti.asldata_sorted.data[:, :, :, :wsp_ti.best_num_vols], 
                                                       rpts=wsp_ti.best_num_vols)
        wsp_ti.asldata_enable.summary(wsp.log)

        wsp_ti.asldata_enable_mean = wsp_ti.asldata_enable.mean_across_repeats()

        ti_data.append(wsp_ti.asldata_enable)
        wsp.enable_results += wsp_ti.results
        wsp.report.add("ti_%i" % (ti_idx+1), wsp_ti.report)

    # Create combined data set
    rpts = [img.rpts[0] for img in ti_data]
    combined_data = np.zeros(list(wsp.asldata.shape[:3]) + [sum(rpts),])
    start = 0
    for nrpts, img in zip(rpts, ti_data):
        combined_data[:, :, :, start:start+nrpts] = img.data
        start += nrpts
    wsp.asldata_enable = wsp.asldata_diff.derived(combined_data, order="rt", tis=wsp.asldata.tis, rpts=rpts)
    wsp.log.write("\nCombined data has %i volumes (repeats at each TI: %s)\n" % (sum(rpts), str(rpts)))
    
    # Pandas can cleverly convert list of dicts to data frame!
    wsp.enable_results = pd.DataFrame(wsp.enable_results)
            
    page = wsp.report.page("summary")
    page.heading("Summary report")
    table = [(ti, orig_rpts, img.rpts[0]) for ti, orig_rpts, img in zip(wsp.asldata_diff.tis, wsp.asldata_diff.rpts, ti_data)]
    page.table(table, headers=["TI", "Original number of repeats", "Number of included repeats"])
    page.heading("BASIL options:", level=1)
    page.text("Using the ENABLE output data, the repeat specification should be::")
    page.text("    %s" % " ".join(["--rpt%i=%i" % (idx+1, rpt) for idx, rpt in enumerate(rpts)]))
    
class EnableOptions(OptionCategory):
    """
    ENABLE option category
    """

    def __init__(self, **kwargs):
        OptionCategory.__init__(self, "enable", **kwargs)

    def groups(self, parser):
        group = IgnorableOptionGroup(parser, "ENABLE options", ignore=self.ignore)
        group.add_option("--noise", "-n", dest="noise_roi", help="Noise ROI. If not specified, will run BET on structural image and invert the brain mask", type="image")
        group.add_option("--noise-from-struc", help="If specified, noise ROI is assumed to be in structural image space and will be registered to ASL space", action="store_true", default=False)
        group.add_option("--gm", dest="gm_roi", help="Grey matter ROI. If not specified, FAST will be run on the structural image", type="image")
        group.add_option("--gm-from-struc", help="If specified, GM ROI is assumed to be in T1 image space and will be registered to ASL space", action="store_true", default=False)
        group.add_option("--regfrom", help="Reference image in ASL space for registration and motion correction. If not specified will use middle volume of ASL data", type="image")
        group.add_option("--min-nvols", help="Minimum number of repeats to keep for each TI", type="int", default=6)

        return [group]

def main():
    """
    Entry point for ENABLE command line application
    """
    try:
        parser = AslOptionParser(usage="oxasl_enable -i <ASL input file> [options...]", version=__version__)
        parser.add_category(image.AslImageOptions())
        parser.add_category(EnableOptions())
        parser.add_category(struc.StructuralImageOptions())
        parser.add_category(GenericOptions())
        
        options, _ = parser.parse_args()
        if not options.output:
            options.output = "enable_output"
        if options.debug:
            options.save_all = True

        wsp = Workspace(savedir=options.output, auto_asldata=True, **vars(options))
        wsp.report.title = "ENABLE processing report"
        print("ASL_ENABLE %s (%s)" % (__version__, __timestamp__))
        
        wsp.asldata.summary()
        print("")

        # Preprocessing (TC subtraction, optional MoCo/smoothing)
        #options.diff = True
        #preproc = preprocess(asldata, options, ref=ref)

        enable(wsp)
        print(wsp.enable_results)
        
        print("\nTo run BASIL use input data %s" % wsp.asldata_enable.name)
        print("and %s" % " ".join(["--rpt%i=%i" % (idx+1, rpt) for idx, rpt in enumerate(wsp.asldata_enable.rpts)]))
    
        wsp.report.generate_html(os.path.join(wsp.output, "report"), "report_build")

    except RuntimeError as e:
        print("ERROR: " + str(e) + "\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
