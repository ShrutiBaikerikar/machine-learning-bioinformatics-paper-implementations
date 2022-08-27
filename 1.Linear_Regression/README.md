# Prediction of Protein Folding Rates From Amino Acid Sequence with Linear Regression

## Table of Contents

- [Abstract](#abstract)
- [Dataset](#dataset)
- [Implementation details](#implementation-details)
- [Results](#results)
- [Citation](#citation)

## Abstract <a name="abstract"></a>

The research paper focusses on the use of amino acid sequence of protein to predict protein-folding rate using linear regression.

The amino acid sequence of proteins has been used to predict three dimensional structure of the protein. Similarly, the author has made use of protein sequence in combination with physical-chemical, energetic
and conformational properties of amino acid residues to predict protein-folding rate. Protein-folding is the process by which protein chains fold to attain its 3D structure that makes the protein biologically functional.

Multiple Linear regression is a model that assumes a linear relationship between multiple input variables and single output variable. The author developed linear regression models that predicted protein folding rates from properties based on amino acid sequences of the protein.
It was also noted that classifying the proteins based on their structure (alpha, beta or mixed) helps improve the correlation between amino acid properties and protein folding rates.

## Dataset <a name="dataset"></a>

The original dataset provided by the author comprised of normalized and raw values of generic properties of amino acids. The normalized values of amino acid properties were further used to prepare the final dataset for machine learning based on the amino acid sequence of each protein.
The final dataset (AA_train.csv) comprises of 50 proteins and 51 features.

The dataset was further split into mini-datasets for further analysis. These datasets have been provided in Datasets folder.


## Implementation details <a name="implementation-details"></a>

Python 3.8 was used for the machine learning implementation. The implementation in this script is different from the author's research as the author's research is rooted in statistical methods while the aim of this script and this repository is to examine Machine Learning in Bioinformatics.

First, amino acid sequences were obtained from Protein Data Bank based on PDB IDs given by the author. Python scripts were written to calculate features for each protein utilising the amino acid sequences, formulae given by the author and scaled values for properties for each amino acid as given by the author.
The final dataset prepared was named as AA_train.csv

This dataset was further split into three datasets: alphaproteins.csv , betaproteins.csv and mixedproteins.csv . The author hypothesized that classification of proteins into different
structural classes helped identify a good correlation between amino acid properties and protein folding rates.

Scikit-Learn package was used to implement linear regression. Therefore linear regression was applied separately to each of the three classes of the proteins and prediction capability was observed.

Finally, linear regression was applied to the whole dataset, to a subset of the dataset including features that were identified as informative by the author and further analysis was done to identify informative features based on current machine learning techniques.


## Results <a name="results"></a>

### A] Linear Regression Implemented on Individual Structural Classes of Proteins

The scripts for these procedures are AA_PF_Alpha.py , AA_PF_Beta.py and AA_PF_Mixed.py . Linear Regression was implemented on each of the datasets separately (alpha, beta and mixed proteins) and the results (coefficients) obtained were very similar to author's calculation. The results are:

- Alpha Proteins

  Equation: ln(kf) = -33.900986 aC + 20.44688311688312
  RMSE: 1.024642930552026

  For Alpha-proteins, parameter aC or 'Power to be at the C-terminal of alpha helix' was found to have a negative correlation.

- Beta Proteins

  Equation: ln(kf) =  -80.568311 K0 + 142.197815 Pb + 73.838257 Ra -112.007958 dASA -10.420315381053525
  RMSE: 1.3876346992227506

  In the equation for beta proteins, K0, Pb, Ra and dASA  are respectively, compressibility, alpha-strand tendency, reduction in solvent accessibility and solvent accessible surface area for protein unfolding.

- Mixed Proteins

  Equation: ln(kf) = -92.555168 K0 -124.162228 Ra + 163.929428 dASA + 113.959323 GhD -59.13100873834357
  RMSE: 2.3252135439259027

  In the equation for mixed proteins, K0, Ra, dASA and GhD are compressibility, reduction in solvent accessibility and solvent accessible surface area for protein unfolding and Gibbs free energy change of hydration for denatured protein. 


### B] Linear Regression Implemented on Entire Dataset

The script for this procedure is AA_PF_LR.py . 

#### - Applying Linear Regression on all Features of the Complete Dataset

  Linear Regression with Stochastic Gradient Descent was applied to the entire dataset. The results were:
  
  Intercept: 0.51474962
       RMSE: 2.841733087647818
  
  Coefficient with highest value: GhN (Gibbs free energy change of hydration for native protein) : 0.383378
  Coefficient with lowest value : Structure : -1.738907


#### - Applying Linear Regression on Informative Features Identified by the Author

  Linear Regression with Stochastic Gradient Descent was applied to the dataset but only on features that were identified as informative by the author. These features were 'Structure','K0','Pb','aC','Ra','dASA','GhD'.
  The results were:

  Intercept: 2.86571834
       RMSE: 2.9068070847907865
  
  Coefficients:

  |  Variable  | Coefficient |
  | ---------  | ----------- |
  | Structure  |  -1.616233  |
  | K0         |   1.210805  |
  | Pb         |   1.350066  |
  | aC         |   0.891756  |
  | Ra         |   1.075924  |
  | dASA       |   1.265247  |
  | GhD        |   1.929174  |


#### - Identifying Important Features using current ML techniques

  Linear Regression with Stochastic Gradient Descent was applied to the entire dataset to identify the top 10 informative features based on correlation. These features were:
  - Structure: Structure class of the protein (Alpha, Beta, Mixed)
  - pHi : Isoelectric point
  - pK : Equilibrium constant with reference to the ionization property of COOH group
  - Mu : Refractive index
  - Pa : Alpha-helical tendency
  - Pt : Turn tendency
  - Pc : Coil tendency
  - F : Mean rms fluctuational displacement
  - aM : Power to be at the middle of alpha helix
  - Nm : Average medium-range contacts
  
  The Root Mean Squared Error obtained was 2.7891654237496137.   


Note: The difference in results obtained from these scripts and those found by the author are due to differences in protein sequence (which is regularly updated) and implementation technique. The implementation technique used in the scripts focuses on the use of Machine Learning as opposed to the author's use of statistical techniques.

## Citation <a name="citation"></a>

Gromiha MM. A statistical model for predicting protein folding rates from amino acid sequence with structural class information. J Chem Inf Model. 2005 Mar-Apr;45(2):494-501. doi: 10.1021/ci049757q. PMID:15807515.
https://pubmed.ncbi.nlm.nih.gov/15807515/ (https://pubmed.ncbi.nlm.nih.gov/15807515/)
