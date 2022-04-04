# Using Support Vector Machine Classifiers to Predict Osteosarcoma Metastasis

## Table of Contents

- [Abstract](#abstract)
- [Dataset](#dataset)
- [Implementation details](#implementation-details)
- [Results](#results)
- [Citation](#citation)

## Abstract <a name="abstract"></a>

Osteosarcoma is a very common type of bone cancer; it is the 8th most common type of cancer in children. Despite the success with chemotherapy, metastasis in osteosarcoma leads to poor prognosis and survival rates.
This research paper explores the gene expression profile of osteosarcoma patients via meta-analysis of microarrays to identify key genes associated with metastasis.

These differentially expressed genes are further utilised to construct a protein-protein interaction network and top featured genes from this network are used to train a Support Vector Machine Classifier.

Support Vector Machine is a supervised machine learning technique which ,in this case, is used to classify samples as 'Metastasis' or 'Non-Metastasis' based on the gene expression data. SVM classifiers plots data in a n-dimensional space and tries to find a hyperplane that could separate the data points into their respective classes.
This hyperplane could be linear or non-linear depending on the kernel used such as Linear-SVM, Polynomial-SVM, Gaussian Radial Basis Function - SVM, Sigmoid-SVM etc.

## Dataset <a name="dataset"></a>

The original datasets used in this study were obtained from the public database GEO (Gene Expression Omnibus) [https://www.ncbi.nlm.nih.gov/geo/] (https://www.ncbi.nlm.nih.gov/geo/)
All the datasets contained gene expression data on osteosarcoma and information regarding metastasis.

- GSE32981 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32981): This dataset contained 23 samples that were divided into 2 groups of 18 metastasis and 5 non-metastasis samples.

- GSE21257 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE21257): This data set included 53 samples that were divided into 2 groups of 34 metastasis and 19 non-metastasis samples.

- GSE14359 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE14359): This data set consisted of 20 samples, but only 18 were included in the study and these were divided into 2 groups of 8 metastasis and 10 non-metastasis samples.

- GSE14827 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE14827): This data set consisted of 27 samples that were divided into 2 groups of 9 metastasis and 18 non-metastasis samples.

- GSE9508 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE9508): This data set consisted of 39 samples, but only 34 were included in the study and these were divided into 2 groups of 21 metastasis and 13 non-metastasis samples.

Preprocessed data, which can be directly utilised for ML, has been provided in the Datasets folder. 
Data obtained after meta-analysis and preprocessing is made available as 'Metaanalysis_dataset.csv' in Datasets folder


## Implementation details <a name="implementation-details"></a>

#### Note: This implementation is slightly different from what has been followed in the original manuscript but the end goal is the same i.e. 'developing an SVM classifier that predicts Osteosarcoma Metastasis'.

R software 4.0.3 was used from pre-processing of the data, statiscal analysis and machine learning implementation.

Datasets GSE14359 and GSE14827 were based on Affymetrix platform. Background correction and normalization of these datasets was performed with 'affy' package. The other three datasets were acquired using GEOquery package. 
Probes corresponding to the same gene were averaged as the final expression value of the gene for all dataset.

MetaQC package was used to perform quality control of the datasets prior to meta-analysis. MetaDE package was used to perform meta-analysis of the datasets.

ind.cal.ES function from MetaDE package was first used to calculate effect sizes and sampling variances. Further MetaDE.ES function was used to screen the differentially expressed genes (DEGs).
The selection criteria for DEGs was: tau2=0, Qpval>0.05 and P<0.05

These DEGs were then used to construct a protein-protein interaction network using Cytoscape. Protein products of the DEGs was mapped to the StringDB database and the PPI network was created. Betweeness Centrality score were calculated for all genes.

Genes were ranked on Betweeness Centrality scores and top 100 genes were selected for further analyisis. Since the sample size of each dataset was relatively small, gene expression data from all datasets were merged and then split into train and test data for further analysis.

'e1071' package was used to develop an SVM classifier. Further Recursive Feature Elimination (RFE-CV) was performed using 'caret' package to identify the most important genes that could aid in prediction of metastasis. A new SVM classifier was developed based on these important genes.

Finally web-based tools [WebGestalt] (http://www.webgestalt.org/) and [ToppGene] (https://toppgene.cchmc.org/) were used to perform Geneset Enrichment Analysis.


## Results <a name="results"></a>

Detailed results (including the plots and the graphs) have been included in the Results.pdf file in the Results Folder.

### A] Differentially Expressed Genes

Quality control of the five gene expression datasets was conducted. Principal Component Analysis revealed that GSE9508 deviated from the other 4 datasets and hence was excluded from further analysis.

Total 298 DEGs were identified from the four datasets. The top 10 genes ranked according to p-value are:

- TREX2: Three Prime Repair Exonuclease 2
- IL2RA: Interleukin 2 Receptor Subunit Alpha
- NCOA3: Nuclear Receptor Coactivator 3
- PARD6A: Par-6 Family Cell Polarity Regulator Alpha
- PGRMC1: Progesterone Receptor Membrane Component 1
- WAC:    WW Domain Containing Adaptor With Coiled-Coil
- TP53I3: Tumor Protein P53 Inducible Protein 3
- PP3CB:  Protein Phosphatase 3 Catalytic Subunit Beta
- SYNJ1:  Synaptojanin 1
- ACADSB: Acyl-CoA Dehydrogenase Short/Branched Chain

### B] Protein-Protein Interaction Network

The DEGs along with a few non-DEGS (with p-value < 0.05) were utilised to create a PPI network. The network comprised of 437 nodes and 3262 edges. 

Betweeness Centrality was calculated for each node and the top 10 genes ranked by Betweeness Centrality score were:
- TFRC   Transferrin Receptor
- PTEN   Phosphatase and tensin homolog
- ATM    ATM serine/threonine kinase
- LDLR   Low density lipoprotein receptor
- CYP2B6 Cytochrome P450 2B6
- PIK3C3 Phosphatidylinositol 3-Kinase Catalytic Subunit Type 3
- PLCB4  Phospholipase C Beta 4
- GNG10  G Protein Subunit Gamma 10
- SYNJ1  Synaptojanin 1
- HDAC6  Histone Deacetylase 6

Top 100 genes ranked on their Betweeness Centrality Score were extracted for further analysis.

### C] Support Vector Machine Classifier

The script involving SVM Classifier on Osteosarcoma dataset is ' Metaanalysis_Osteosarcoma_SVM.R ', available in the Scripts Folder.

Data pertaining to the top 100 genes (ranked on Betweeness Centrality score) was extracted from each gene expression datasets. Since the sample sizes were small, data from each gene expression dataset was merged to form a new dataset
that comprised of 121 samples and 100 genes. A 70-30% train - test split was applied.

SVM Classifier with RBF kernel was used to classify the data into Metastasis and Non-Metastasis samples. Train accuracy of 93.02% and test accuracy of 71.43% was obtained.

### D] Gene Set Enrichment Analysis

The pathways identified in the Over Representation Analysis using WebGestalt and Gene Ontology database are:
- GO:0008152	metabolic process	
- GO:0065007	biological regulation	
- GO:0050896	response to stimulus	
- GO:0032501	multicellular organismal process	
- GO:0007154	cell communication	
- GO:0032502	developmental process	
- GO:0016043	cellular component organization	
- GO:0051179	localization	
- GO:0051704	multi-organism process	
- GO:0008283	cell proliferation	
- GO:0040007	growth	
- GO:0000003	reproduction	 

The top 10 pathways identified by ToppGene and KEGG database are:
- 83061	        Wnt signaling pathway
- 373901	HTLV-I infection
- 749777	Hippo signaling pathway
- 83105	        Pathways in cancer
- 193328	mRNA surveillance pathway
- 102279	Endocytosis
- 83052	        Phosphatidylinositol signaling system
- 868086	Rap1 signaling pathway
- 83092	        Melanogenesis
- 868085	Ras signaling pathway
 


#### Note: Differences in Results obtained by scripts in this repository versus those obtained by authors of the research papers could be attributed to revisions in genome annotation, differences in preprocessing of genomic data and the heuristic nature of algorithms implemented.


## Citation <a name="citation"></a>

He, Y., Ma, J., & Ye, X. (2017). A support vector machine classifier for the prediction of osteosarcoma metastasis with high accuracy. International journal of molecular medicine, 40(5), 1357–1364. https://doi.org/10.3892/ijmm.2017.3126
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5627885/ (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5627885/)
