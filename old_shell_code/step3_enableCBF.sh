
#!/bin/bash
# script to generate CBFraw & CBFenable images
codedir=$HOME/build/ENABLE/

# name convention: ASL: ASL_${subID} & T1:  T1_${subID}

subID=$1

lambda=0.9
alpha=0.98
TI1=0.7
TI=1.9
T1blood=1.65
TE=0.012

cd ${subID}

    numIPIs=`cat ../group_qualityAssess/optIndex.csv | head -n ${subID} |tail -n 1`
    fslroi ASL_${subID}_diff_smooth_sorted ASL_${subID}_diff_smooth_sorted_enable 0 $numIPIs
    fslmaths ASL_${subID}_diff_smooth_sorted_enable -Tmean ASL_${subID}_diff_smooth_sorted_enable_mean
    fslmaths ASL_${subID}_diff_smooth_sorted_enable_mean -thr -150 -uthr 150 ASL_${subID}_diff_smooth_sorted_enable_mean
    fslmaths ASL_${subID}_diff_smooth_mean -thr -150 -uthr 150 ASL_${subID}_diff_smooth_conv_mean

    mkdir temp
    for z in `seq 10 33`; do
        zz=$[$z -9]
        TIeff="`echo "scale=10; ($TI+$TE*3.6*($zz-1))" | bc -l`"
        calibration="`echo "scale=15; 6000*${lambda}*e($TIeff/$T1blood)/(2*$alpha*$TI1)" | bc -l`"
        fslroi ASL_${subID}_diff_smooth_conv_mean temp/diffmean_z$z 0 -1 0 -1 $zz 1
        fslmaths temp/diffmean_z$z -mul $calibration temp/CBF_z$z
    done
    fslmerge -z ASL_${subID}_CBF_conv1 temp/CBF_*
    fslmaths ASL_${subID}_CBF_conv1 -div ASL_${subID}_ref_bet ASL_${subID}_CBF_conv
    #rm -rf temp/

    #mkdir temp
    for z in `seq 10 33`; do
        zz=$[$z -9]
        TIeff="`echo "scale=10; ($TI+$TE*3.6*($zz-1))" | bc -l`"
        calibration="`echo "scale=15; 6000*${lambda}*e($TIeff/$T1blood)/(2*$alpha*$TI1)" | bc -l`"
        fslroi ASL_${subID}_diff_smooth_sorted_enable_mean temp/diffmean_z$z 0 -1 0 -1 $zz 1
        fslmaths temp/diffmean_z$z -mul $calibration temp/CBF_z$z
    done
    fslmerge -z ASL_${subID}_CBF_enable temp/CBF_*
    fslmaths ASL_${subID}_CBF_enable -div ASL_${subID}_ref_bet ASL_${subID}_CBF_enable
    #rm -rf temp/

cd ../










