
#!/bin/bash
# script to run enable to sort and calculate ASL quality measures

#codedir=/Volumes/zshirzadi/codes/
codedir=$HOME/build/ENABLE/

# name convention: ASL: ASL_${subID} & T1:  T1_${subID}

subID=$1

cd ${subID}/

echo "sort ASL-diff images based on CNR"
echo ""

tdim=`fslval ASL_${subID}_diff_smooth dim4`
tdimminus1=$(($tdim - 1))

time_steps=`seq 0 $tdimminus1`
mkdir temp

for i in ${time_steps[*]};do
    fslroi ASL_${subID}_diff_smooth temp/asldiff_${i} $i 1
    meanGM=`fslstats temp/asldiff_${i} -k fast/T1_${subID}_fast_seg_2asl_GM -m`
    noisestd=`fslstats temp/asldiff_${i} -k ../noise_ROI -s`
    cnrrev1=$(echo " scale=0; $noisestd*100000000/$meanGM" | bc -l)

    if [ "$cnrrev1" -lt "0" ]; then
        cnrrev1=`echo $(echo "${cnrrev1}+1000000000000" | bc -l)`
    fi

    cnrrev=$(echo 000000000${cnrrev1} | tail -c 14)

    mv temp/asldiff_${i}.nii.gz temp/asldiff_${cnrrev}.nii.gz
done

fslmerge -t ASL_${subID}_diff_smooth_sorted temp/asldiff_*.nii.gz
rm -rf temp/



echo "calculate quality measures"
echo ""
allVoxels_GM=`fslstats fast/T1_${subID}_fast_seg_2asl_GM  -V | cut -d ' ' -f 1`

time_steps=`seq 6 1 $tdimminus1`
fslmaths ASL_${subID}_diff_smooth_mean -mul 0 -add 1 oneimage

for i in ${time_steps[*]}; do

    fslroi ASL_${subID}_diff_smooth_sorted temp_output 0 $i
    fslmaths temp_output.nii.gz -Tmean imagemean_${i}
    fslmaths temp_output.nii.gz -Tstd imagestd_${i}
    fslmaths imagemean_${i} -div imagestd_${i} imagesnr_${i}
    rooti=$(echo "sqrt ( $i )" | bc -l)
    fslmaths imagestd_${i} -div ${rooti} imageserr_${i}
    fslmaths imagemean_${i} -div imageserr_${i} imagetstats_${i}
    ttologp -logpout logp1 oneimage imagetstats_${i} ${i}
    fslmaths logp1 -exp p1
    fslmaths p1 -uthr .05 p1_threshold

    sigvoxGM=`fslstats p1_threshold -k fast/T1_${subID}_fast_seg_2asl_GM  -V | cut -d ' ' -f 1`
    DetectGM=$(echo "$sigvoxGM/$allVoxels_GM" | bc -l)
    tSNRGM=`fslstats imagesnr_${i} -k fast/T1_${subID}_fast_seg_2asl_GM  -m`
    meanGM=`fslstats imagemean_${i} -k fast/T1_${subID}_fast_seg_2asl_GM  -m`
    stdGM=`fslstats imagemean_${i} -k fast/T1_${subID}_fast_seg_2asl_GM  -s`
    CoVGM=$(echo "$stdGM/$meanGM*100" | bc -l)
    noisestd=`fslstats imagemean_${i} -k ../noise_ROI -s`
    SNRGM=$(echo "$meanGM/$noisestd" | bc -l)
    echo  ${SNRGM} "," ${DetectGM} "," ${CoVGM} ","  ${tSNRGM} >> asldiff_qualitymeasures.csv

    rm -rf image*
    rm -rf logp1*
    rm -rf p1*
    rm -rf temp_*
done

rm -rf oneimage*



cd ../



mkdir group_qualityAssess/


cat ${subID}/asldiff_qualitymeasures.csv |cut -d ',' -f 1  >> group_qualityAssess/SNR_${subID}.csv
cat ${subID}/asldiff_qualitymeasures.csv |cut -d ',' -f 2  >> group_qualityAssess/Detect_${subID}.csv
cat ${subID}/asldiff_qualitymeasures.csv |cut -d ',' -f 3  >> group_qualityAssess/CoV_${subID}.csv
cat ${subID}/asldiff_qualitymeasures.csv |cut -d ',' -f 4 >> group_qualityAssess/tSNR_${subID}.csv

paste -s  group_qualityAssess/SNR_*.csv >> group_qualityAssess/SNRallsubjects.csv
paste -s group_qualityAssess/Detect_*.csv   >> group_qualityAssess/Detectallsubjects.csv
paste -s group_qualityAssess/CoV_*.csv    >> group_qualityAssess/CoVallsubjects.csv
paste -s group_qualityAssess/tSNR_*.csv    >> group_qualityAssess/tSNRallsubjects.csv


##### go to matlab and get the enable indecies 




