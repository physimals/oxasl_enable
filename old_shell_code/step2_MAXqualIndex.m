clc; 
clear all;
%pkg load statistics;

SNR=csvread('SNRallsubjects.csv')';
Detect=csvread('Detectallsubjects.csv')'; 
CoV=csvread('CoVallsubjects.csv')';
tSNR=csvread('tSNRallsubjects.csv')';
coef=[ 0.1    1.8   -1.0   -1.0]; %  weightings for PLD=1300, 3T data, these changes depending on PLD and B0 

NumSubjects=1
for i=1:NumSubjects
Detect_norm(:,i)=Detect(:,i)/max(Detect(:,i));
SNR_norm(:,i)=SNR(:,i)/max(SNR(:,i));
CoV_norm(:,i)=CoV(:,i)/max(CoV(:,i));
tSNR_norm(:,i)=tSNR(:,i)/max(tSNR(:,i));
end

qual=coef(1).*SNR_norm + coef(2).*Detect_norm + coef(3).*CoV_norm +coef(4).*tSNR_norm;
[maxqual, maxindex]=max(qual);
optIndex=maxindex+5;
csvwrite('optIndex.csv',optIndex')
%figure
%boxplot(optIndex)

