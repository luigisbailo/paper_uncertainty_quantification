 Supplementary material to reproduce results presented in the manuscript: "Uncertainty Quantification in Deep Neural Networks through Statistical Inference on Latent Space".
 
 ===
 This directory contains a jupyter notebook 'reproduce_results.ipynb' that reproduces results presented in the manuscript, and two python modules 'confidence_eval.py' and 'nn.py' that are imported in the notebook.
 
 The notebook requires to be run in an environment that contains a number of packages. The setup of the python environment can be done using anaconda with:
 
 conda create -n uq_results python numpy scipy scikit-learn matplotlib jupyter torch torchvision