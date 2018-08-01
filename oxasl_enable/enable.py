#!/usr/bin/env python
# ENABLE package for ASL

import os, sys
import glob
import shutil
import math
from optparse import OptionParser, OptionGroup

import nibabel as nib
import numpy as np
import scipy.stats

from oxasl import AslImage, Workspace, AslOptionParser, OptionCategory, IgnorableOptionGroup, GenericOptions
from oxasl.struc import StructuralImageOptions, preproc_struc, segment

from ._version import __version__, __timestamp__

def get_rois(wsp):
    wsp.log.write("Generating ROIs...\n")

    # Structural image is required if we need to derive grey matter or noise ROIs
    if (wsp.gm_roi is None or wsp.noise_roi is None or wsp.noise_from_struc or wsp.gm_from_struc) and wsp.struc is None:
        raise RuntimeError("Need to specify a structural image if not providing both noise and GM ROIs or if either are in structural space")

    if wsp.gm_roi is None:
        # If we need a GM ROI, segment structural image 
        segment(wsp)
        wsp.log.write("Taking GM ROI from segmentation of structural image\n")
        wsp.gm_roi = Image((wsp.gm_pv_struc.data > 0).astype(np.int), header=wsp.struc.header)
        options.gm_from_struc = True

    if wsp.noise_roi is None:
        # If we need a noise ROI, invert the structural brain mask
        preproc_struc(wsp)
        wsp.log.write("Generating noise ROI by inverting T1 brain mask\n")
        wsp.noise_roi = Image(1-wsp.struc_brain_mask.data, header=wsp.struc.header)
        wsp.noise_from_struc = True

    if wsp.noise_from_struc or wsp.gm_from_struc:
        # Need to register T1 to ASL space so we can apply the transform the the 
        # either or both of the noise/GM ROIs
        wsp.regfrom = wsp.reg
        wsp.do_flirt = True
        wsp.do_bbr = False
        reg_asl2struc(wsp)

    if wsp.noise_from_struc:
        wsp.log.write(" - Registering noise ROI to ASL space since it was defined in structural space\n\n")
        wsp.noise_roi_asl = fsl.applyxfm(wsp.noise_roi, wsp.regfrom, wsp.struc2asl, out=fsl.LOAD, interp="nearestneighbour", log=wsp.fsllog)["out"]
    else:
        wsp.noise_roi_asl = wsp.noise_roi

    if options.gm_from_t1:
        wsp.log.write(" - Registering GM ROI to ASL space since it was defined in structural space\n\n")
        wsp.gm_roi_asl = fsl.applyxfm(wsp.gm_roi, wsp.regfrom, wsp.struc2asl, out=fsl.LOAD, interp="nearestneighbour", log=wsp.fsllog)["out"]
    else:
        wsp.gm_roi_asl = wsp.gm_roi

    wsp.log.write("DONE\n\n")
    return noise_roi, gm_roi

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

def calculate_cnr(asl_data, gm_roi, noise_roi, log=sys.stdout):
    """
    Sort ASL images based on CNR (Contrast:Noise ratio). 

    CNR is defined as mean difference signal divided by nonbrain standard deviation

    :param asl_data: Differenced single-TI ASL data
    :param gm_roi: Grey-matter mask
    :param noise_roi: Noise mask (typically all non-brain parts of the image)
    
    :returns: 4D image in which volumes have been sorted from highest to lowest CNR
    """
    wsp.log.write("Sort ASL-diff images based on CNR...\n")

    tdim = asl_data.shape[3]
    cnrs = []
    for i in range(tdim):
        vol_data = asl_data.data()[:,:,:,i].astype(np.float32)
        meanGM = np.mean(vol_data[gm_roi.data() > 0])
        noisestd = np.std(vol_data[noise_roi.data() > 0])
        cnr = meanGM/noisestd
        if cnr < 0:
            wsp.log.write("WARNING: CNR was negative - are your tag-control pairs the right way round?")
        cnrs.append(cnr)

    return cnrs

def sort_cnr(asl_data, cnrs, log=sys.stdout):
    # Sort by decreasing CNR
    sorted_cnrs = sorted(enumerate(cnrs), key=lambda x: x[1], reverse=True)

    # Create re-ordered data array
    wsp.log.write("Images sorted by CNR\n\n")
    wsp.log.write("Volume\tCNR\n")
    sorted_data = np.zeros(asl_data.shape)
    for idx, cnr in enumerate(sorted_cnrs):
        vol_data = asl_data.data()[:,:,:,cnr[0]].astype(np.float32)
        sorted_data[:,:,:,idx] = vol_data
        wsp.log.write("%i\t%.3f\n" % (cnr[0], cnr[1]))

    wsp.log.write("DONE\n\n")

    sorted_cnrs = [(idx, cnr[0], cnr[1]) for idx, cnr in enumerate(sorted_cnrs)]
    return asl_data.derived(sorted_data, suffix="_sorted"), sorted_cnrs

def calculate_quality_measures(asl_data, gm_roi, noise_roi, min_nvols, log=sys.stdout):
    """
    Calculate quality measures for ASL data, sorted by CNR
    """
    wsp.log.write("Calculate quality measures...\n")
    if min_nvols < 2:
        raise RuntimeError("Need to keep at least 2 volumes to calculate quality measures")

    tdim = asl_data.shape[3]
    gm_roi = gm_roi.data()
    noise_roi = noise_roi.data()
    num_gm_voxels = np.count_nonzero(gm_roi)

    wsp.log.write("Volumes\ttCNR\tDETECT\tCOV\ttSNR\n")
    qms = {"tcnr" : [], "detect" : [], "cov" : [], "tsnr" : []}
    
    for i in range(min_nvols, tdim+1, 1):
        temp_data = asl_data.data()[:,:,:,:i]

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
        calc_p = np.vectorize(lambda x: tsf(i, x))
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

    wsp.log.write("DONE\n\n")  
    return qms

def get_combined_quality(qms, ti, b0="3T", log=sys.stdout):

    # Weightings from the ENABLE paper, these vary by PLD
    sampling_plds = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]
    coef={
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

    num_meas = len(qms["detect"])
    qual = np.zeros([num_meas], dtype=np.float)
    for meas, vals in qms.items():
        c = np.interp([ti], sampling_plds, coeffs[meas])

        # Try to ensure numerical errors do not affect the result
        vals[vals == np.inf] = 0
        vals[vals == -np.inf] = 0
        vals = np.nan_to_num(vals)

        normed = np.array(vals, dtype=np.float) / max(vals)
        qual += c * normed
    
    return qual

def enable(wsp):
    wsp.asldata_diff = wsp.asldata.diff().reorder("rt")
    results = []
    ti_data = []

    # Process each TI in turn
    for idx, ti in enumerate(wsp.asldata_diff.tis):
        wsp.log.write("\nProcessing TI: %i (%s)\n" % (idx, str(ti)))
        
        # Create workspace for TI
        wsp_ti = wsp.sub("ti_%s" % str(ti))
        wsp_ti.asldata = wsp.asldata_diff.single_ti(idx)
        wsp_ti.results = []

        # Write out the mean differenced image for comparison
        wsp_ti.asldata_mean = wsp_ti.asldata.mean_across_repeats()

        # Sorting and calculation of quality measures
        wsp_ti.cnrs = calculate_cnr(wsp_ti.asldata, wsp.gm_roi, wsp.noise_roi)

        wsp_ti.sorted_data, wsp_ti.sorted_cnrs = sort_cnr(wsp_ti.asldata, wsp_ti.cnrs)
        for sorted_order, orig_vol, cnr in wsp_ti.sorted_cnrs:
            wsp_ti.results.append(
                {"ti" : ti, "rpt" : orig_vol, "cnr" : cnr,
                 "tcnr" : 0.0, "detect" : 0.0, "cov" : 0.0, "tsnr" : 0.0, "qual" : 0.0}
            )

        wsp_ti.qms = calculate_quality_measures(wsp_ti.sorted_data, wsp.gm_roi, wsp.noise_roi, min_nvols=wsp.min_nvols)
        for meas in "tcnr", "detect", "cov", "tsnr":
            for idx, val in enumerate(wsp_ti.qms[meas]):
                wsp_ti.results[idx+min_nvols-1][meas] = val
           
        # Get the image at which the combined quality measure has its maximum
        wsp_ti.combined_q = get_combined_quality(wsp_ti.qms, ti)
        wsp.log.write("Volumes\tOverall Quality\n")
        for idx, q in enumerate(combined_q):
            wsp.log.write("%i\t%.3f\n" % (idx, q))
            ti_results[idx+min_nvols-1]["qual"] = q
            
        # Generate data subset with maximum quality
        wsp_ti.best_num_vols = np.argmax(wsp_ti.combined_q) + wsp.min_nvols
        maxqual = wsp_ti.results[wsp_ti.best_num_vols-1]["qual"]
        wsp.log.write("Maximum quality %.3f with %i volumes\n" % (maxqual, best_num_vols))

        for idx, result in enumerate(wsp_ti.results):
            result["selected"] = idx < wsp_ti.best_num_vols
            
        wsp_ti.asldata_enable = AslImage(wsp_ti.sorted_data.data[:,:,:,:wsp_ti.best_num_vols], 
                                         order=wsp_ti.sorted_data.order, ntis=1, nrpts=best_num_vols,
                                         header=wsp_ti.sorted_data.header)

        wsp_ti.asldata_enable_mean = wsp_ti.asldata_enable.mean_across_repeats()

        ti_data.append(wsp_ti.asldata_enable)
        results += wsp_ti.results

    # Create combined data set
    rpts = [img.rpts[0] for img in ti_data]
    combined_data = np.zeros(list(wsp.asldata.shape[:3]) + [sum(rpts),])
    start = 0
    for nrpts, img in zip(rpts, ti_data):
        combined_data[:,:,:,start:start+nrpts] = img.data
        start += nrpts
    wsp.log.write("\nCombined data has %i volumes (repeats: %s)\n" % (sum(rpts), str(rpts)))
    wsp.asldata_enable = AslImage(combined_data,
                                  order="rt", tis=wsp.asldata.tis, rpts=rpts,
                                  header=wsp.asldata.header)
    return wsp.asldata_enable, results

class EnableOptions(OptionCategory):
    """
    ENABLE option category
    """

    def __init__(self, **kwargs):
        OptionCategory.__init__(self, "enable", **kwargs)

    def groups(self, parser):
        groups = []
        
        group = IgnorableOptionGroup(parser, "ENABLE options", ignore=self.ignore)
        group.add_option("--noise", "-n", dest="noise", help="Noise ROI. If not specified, will run BET on structural image and invert the brain mask", type="image")
        group.add_option("--noise-from-struc", help="If specified, noise ROI is assumed to be in structural image space and will be registered to ASL space", action="store_true", default=False)
        group.add_option("--gm", help="Grey matter ROI. If not specified, FAST will be run on the structural image", type="image")
        group.add_option("--gm-from-struc", help="If specified, GM ROI is assumed to be in T1 image space and will be registered to ASL space", action="store_true", default=False)
        group.add_option("--ref", help="Reference image in ASL space for registration and motion correction. If not specified will use middle volume of ASL data",  type="image")
        group.add_option("--min-nvols", help="Minimum number of volumes to keep for each TI", type="int", default=6)
        groups.append(group)

        return groups

def main():
    """
    Entry point for ENABLE command line application
    """
    try:
        parser = AslOptionParser(usage="oxasl_enable -i <ASL input file> [options...]", version=__version__)
        parser.add_category(AslImageOptions())
        parser.add_category(EnableOptions())
        parser.add_category(StructuralImageOptions())
        parser.add_category(GenericOptions())
        
        options, _ = parser.parse_args(sys.argv)
        if not options.output:
            options.output = "enable_output"

        if not options.asldata:
            sys.stderr.write("Input ASL data not specified\n")
            parser.print_help()
            sys.exit(1)
                
        print("ASL_ENABLE %s (%s)" % (__version__, __timestamp__))
        asldata = AslImage(options.asldata, **parser.filter(options, "image"))
        wsp = Workspace(savedir=options.output, **vars(options))
        wsp.asldata = asldata
        
        asldata.summary()
        print("")

        # Reference image for registration/MoCo
        if wsp.ref is None:
            # Use middle volume of ASL data if not given
            middle = int(asldata.shape[3]/2)
            wsp.ref = Image(asldata.data[:,:,:,middle], header=asldata.header)
            
        # Preprocessing (TC subtraction, optional MoCo/smoothing)
        #options.diff = True
        #preproc = wsp.preprocess(asldata, options, ref=ref)

        enable(wsp)
        for result in wsp.enable_results:
            print("Ti=%.3f, Repeat %i, CNR=%.3f, Q=%.3f, selected=%s" % (result["ti"], result["rpt"], result["cnr"], result["qual"], result["selected"]))
        
        print("\nTo run BASIL use input data %s" % wsp.enable_asldata.name)
        print("and %s" % " ".join(["--rpt%i=%i" % (idx+1, rpt) for idx, rpt in enumerate(wsp.enable_asldata.rpts)]))
    
    except RuntimeError as e:
        print("ERROR: " + str(e) + "\n")
        p.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
