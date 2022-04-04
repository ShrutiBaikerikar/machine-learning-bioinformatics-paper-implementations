# Classification and Selection of Genes as Cancer Biomarkers with Random Forest method

## Table of Contents

- [Abstract](#abstract)
- [Dataset](#dataset)
- [Implementation details](#implementation-details)
- [Results](#results)
- [Citation](#citation)

## Abstract <a name="abstract"></a>

The research paper focusses on microarray and next-generation sequencing technologies to identify to informative molecular signals or patterns in diseases.

Analysis of gene expression data from microarray experiments reveal which genes could be significantly involved in pathogenesis of a disease. Here the authors have utilised this principle in combination with machine learning techniques to identify cancer biomarkers.

Random forest is a machine learning technique which ,in this case, is used to classify samples as 'Cancer' or 'Normal' based on the gene expression data. Random Forest is a supervised learning algorithm which builds a 'forest' of multiple decision trees and merges the results to give an accurate prediction.


## Dataset <a name="dataset"></a>

The original datasets used in this study were obtained from the public database GEO (Gene Expression Omnibus) [https://www.ncbi.nlm.nih.gov/geo/] (https://www.ncbi.nlm.nih.gov/geo/)

-Leukemia [GSE9476](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE9476): This dataset contained 64 samples and 22283 genes that were divided into 2 groups of 38 control and 26 cancer samples.

-Colon cancer [GSE44861](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE44861): This data set included 111 samples and 22 277 genes that were divided into 2 groups of 55 control and 56 cancer samples.

-Prostate cancer [GSE71783](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE71783): This data set consisted of 30 samples and 17881 genes that were divided into 2 groups of 15 control and 15 cancer samples.

Preprocessed data, which can be directly utilised for ML, has been provided in the Dataset folder. The datasets are:

 - Leukemia(GSE9476) : Leukemia_RFData.csv
 - Prostate Cancer(GSE71783): ProstateCancer_RFData.csv
 - Colon Cancer(GSE44861): ColonCancer_RFData.csv

## Implementation details <a name="implementation-details"></a>

R software 4.0.3 was used for pre-processing of the data, statiscal analysis and machine learning implementation.

Limma package was used to analyze and preprocess the microarray gene expression datasets. Differentially expressed genes were identified and top 2000 genes with a p-value < 0.05 were retained for further study.

This data comprising of 2000 gene expression values for given number of samples was trained by Random Forest method. The importance of each gene was calculated and ranked. These steps were perfomed using Random Forest package.

Further to obtain the smallest set of genes that could predict whether samples were 'Normal' or 'Cancer/Tumor', varSelRF package was used. This package iteratively fits random forest on the given data. At each step of iteration, it discards a certain percentage of genes that are least important in the prediction. 
Here the percentage was 0.2 which is the default value; it means that 20% of the genes are dropped at each iteration.

Variable importance was not recalculated at each step to avoid overfitting. The authors defined the smallest set of genes as the same set such that the minimum SD of error of the entire forest is zero.


## Results <a name="results"></a>

The scripts on Random Forest applications for each of the dataset are as follows:
- Leukemia(GSE9476) : Leukemia_RF.R
- Prostate Cancer(GSE71783): ProstateCancer_RF.R
- Colon Cancer(GSE44861): ColonCancer_RF.R

Random Forest method extracted small set of genes for every dataset that could predict whether a sample was ' Normal' or 'Cancer/Tumor'. The findings were as follows:
-Leukemia        : ALDH1A1, BAG4, GPX1, JAG1, PLXNC1      Accuracy:95.31  Precision:96.00
-Colon Cancer    : CA1, CA7, DIEXF, GUCA2A, GUCA2B, IGH   Accuracy:87.39  Precision:86.21
-Prostate Cancer : ACP5, CENPBD1, MT1A, PROM1, QTRT1      Accuracy:70.00  Precision:71.43

Most of the genes identified by Random Forest method have biological relevance in the respsective cancer pathology and diagnosis and some have also been identified as clinical biomarkers via wet-lab and other experimental techniques. The detailed results along with their inferences can be found in the file Results.pdf in the Results Folder. 

##### Note: Differences in Results obtained by scripts in this repository versus those obtained by authors of the research papers could be attributed to revisions in genome annotation, differences in preprocessing of genomic data and the heuristic nature of algorithms implemented.


## Citation <a name="citation"></a>

Ram, M., Najafi, A., Shakeri, M. (2017). Classification and Biomarker Genes Selection for Cancer Gene Expression Data Using Random Forest. Iranian Journal of Pathology, 12(4), 339-347. doi: 10.30699/ijp.2017.27990
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5844678/ (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5844678/)
