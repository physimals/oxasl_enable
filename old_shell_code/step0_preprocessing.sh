
#!/bin/bash

#codedir=/Volumes/zshirzadi/codes/
codedir=$HOME/build/ENABLE/

# name convention: ASL: ASL_${subID} & T1:  T1_${subID}

subID=$1
cd ${subID}/

echo "ASL preprocessing"
echo ""

tdim=`fslval ASL_${subID} dim4`
echo "TDIM=$tdim"
tdimminus1=$(($tdim - 1))

fslroi ASL_${subID} ASL_${subID}_ref 0 1
fslroi ASL_${subID} ASL_${subID}_raw 1 $tdimminus1

bash ${codedir}asl_preproc.sh -i ASL_${subID}_raw --nrp $tdimminus1 --nti 1 -o ASL_${subID}_diff -s --fwhm 5 -m

bet ASL_${subID}_ref ASL_${subID}_ref_bet
# this is done to avoid the contrast enhanced rim resulting from low intensity ref image
thr_ref=`fslstats ASL_${subID}_ref_bet -P 10`
fslmaths ASL_${subID}_ref_bet -thr $thr_ref ASL_${subID}_ref_bet

echo "T1 segmentation and co-registration"
echo ""

bet T1_${subID} T1_${subID}_bet -B -f 0.3

flirt -dof 7 -in ASL_${subID}_ref_bet -ref T1_${subID}_bet -out ASL_2T1 -omat ASL_2T1.mat
convert_xfm -omat T1_2ASL.mat -inverse ASL_2T1.mat

mkdir fast
fast -p -o fast/T1_${subID}_fast T1_${subID}_bet

flirt -in fast/T1_${subID}_fast_seg -ref ASL_${subID}_ref_bet -applyxfm -init T1_2ASL.mat -interp nearestneighbour -out fast/T1_${subID}_fast_seg_2asl

fslmaths fast/T1_${subID}_fast_seg_2asl -thr 1.5 -uthr 2.5 fast/T1_${subID}_fast_seg_2asl_GM

cd ../









