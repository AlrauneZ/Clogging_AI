# Overview

This project provides all python scripts to reproduce the results of the manuscript
"Prediction of Pore-scale Clogging Using Artificial Intelligence Algorithms" 
by Chao Lei, Mandana Samari-Kermani , Hamed Aslannejad, and Alraune Zech.

It provides the implementations of the workflows for training and testing the 
five AI algorithms:
    + Artificial Neural Network (ANN)
    + Decision Tree (DT)
    + Random Forest (FR)
    + Linear Regression (LR)
    + Support Vector Regression (SVR)  

It further provides a summary of the algorithm results and python 
scripts to reproduce all figures (in the manuscript and Supporting Information) 
based on the input data and results.

## Structure

The project is organized as follows:

- `README.md` - descrition of the project and its structure
- `LICENSE` - the default license is MIT

- `data/` - folder containing input data:
  + `LBM_results.xlsx` - LBM simulation results for favorable and unfavorable 
      conditions sorted in two sheets
  + `Results_Testing_unfav.scv` - summary of perfomance measures NSE(=R2), MSE and MAE 
      for the four output values for all five algorithms 
      for unfavorable conditions
  + `Results_Testing_fav.scv` - summary of perfomance measures NSE(=R2), MSE and MAE 
      for the four output values for all five algorithms 
      for favorable conditions

- `results/` - here we store computed results and plots displayed in the manuscript and Supporting Information
  + `ANN_Test_results.txt` - Summary of performance Testing for ANN
  + `DT_Test_results.txt` - Summary of performance Testing for DT
  + `RF_Test_results.txt` - Summary of performance Testing for RF
  + `LR_Test_results.txt` - Summary of performance Testing for LR
  + `SVR_Test_results.txt` - Summary of performance Testing for SVR
  + `Fig01_ANN_Hyper.pdf` - Figure 1 from the manuscript on results of hyperparameter testing for ANN
  + `Fig02_DT_Hyper_mss.pdf` - Figure 2 from the manuscript on results of hyperparameter testing for DT
  + `Fig03_Results_Testing.pdf` - Figure 3 from the manuscript on results of all algorithms on test data set
  + `FigS01_ANN_Hyper_Full.pdf` - Figure S1 from the supporting information (SI) on results of hyperparameter testing for ANN
  + `FigS02_DT_Hyper_fav.pdf` - Figure S2 (part on favorable conditions) from SI  on results of hyperparameter testing for DT
  + `FigS02_DT_Hyper_unfav.pdf` - Figure S2 (part on unfavorable conditions) from SI on results of hyperparameter testing for DT
  + `FigS03_RF_Hyper.pdf` - Figure S3 from SI on results of hyperparameter testing for  RF
  + `FigS04_LR_Hyper.pdf` - Figure S3 from SI on results of hyperparameter testing for LR
  + `FigS05_SVR_Hyper_fav.pdf` - Figure S5 (part on favorable conditions) from SI on results of hyperparameter testing for SVR
  + `FigS05_SVR_Hyper_unfav.pdf` - Figure S5 (part on unfavorable conditions) from SI on results of hyperparameter testing for SVR
  + `FigS06_Results_Testing_MSE.pdf` - Figure S6 from SI on results of algorithms on test data set using MSE as performance evaluation
  + `FigS07_Hyper_Clogging.pdf` - Figure S7 from SI on results of hyperparameter testing for clogging

- `src/` - here we place your python/matlab scripts 

  + `01_Training_DT.py` - run the training for the DT
  + `02_Testing_DT.py` - run the performance test for the DT
  + `03_Training_RF.py` - run the training for the RF
  + `04_Testing_DT.py` - run the performance test for the RF
  + `05_Training_DT.py` - run the training for the LR
  + `06_Testing_DT.py` - run the performance test for the LR
  + `07_Training_DT.py` - run the training for the SVR
  + `08_Testing_DT.py` - run the performance test for the SVR
  + `09_Training_DT.py` - run the training for the ANN
  + `10_Testing_DT.py` - run the performance test for the ANN
  + `11_Training_DT.py` - run the training for clogging data set (ANN, DT)
  + `12_Testing_DT.py` - run the performance test for clogging data set (ANN, DT, RF)
  + `Fig01_Results_ANN.py` - script to create Figure 1 from the manuscript
  + `Fig02_Results_DT.py` - script to create Figure 2 from the manuscript
  + `Fig03_Results_Testing.py` - script to create Figure 3 from the manuscript
  + `SF01_Results_ANN.py` - script to create Figure S1 from the supporting information (SI)
  + `SF02_Results_DT.py` -  script to create Figure S2 from SI
  + `SF03_Results_RF.py` - script to create Figure S3 from SI
  + `SF04_Results_LR.py` - script to create Figure S4 from SI
  + `SF05_Results_SVR.py` - script to create Figure S5 from SI
  + `SF06_Results_Testing_MSE.py` - script to create Figure S6 from SI
  + `SF07_Results_Clogging.py` - script to create Figure S7 from SI


## Python environment

To make the example reproducible, we provide the following files:
- `requirements.txt` - requirements for [pip](https://pip.pypa.io/en/stable/user_guide/#requirements-files) to install all needed packages

## Contact

You can contact us via <a.zech@uu.nl>.

## License

MIT © 2023
