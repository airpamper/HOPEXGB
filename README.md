# HOPEXGB
This is a public code for predicting the associations of miRNAs and lncRNAs with diseases

'data' directory

Contain the associations between lncRNAs and diseases, the associations between miRNAs and diseases, the interactions between lncRNAs and miRNAs, the heterogeneous network DML.

code

python preprocess.py--To create the heterogeneous disease-miRNA-lncRNA (DML) information network


python HOPEXGB.py--To predict the associations of miRNAs/lncRNAs with diseases

Requirements
HOPEXGB is tested to work under Python 3.8.0+

The required dependencies for HOPEXGB are sklearn, numpy, pandas, scipy, itertools, xgboost and time.
