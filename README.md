# Overview

To be added: Project description.

## Structure

The project is organized as follows:


- `README.md` - describe your example in the readme, potentially showing results
- `LICENSE` - the default license is MIT, we can use another one if we want
- `data/` - here we place your input data 
  + `LBM_results_fav.scv` - LBM simulation results for favorable conditions
  + `LBM_results_unfav.scv` - LBM simulation results for unfavorable conditions
  + to be added: descritions of other files containing IA evaluation results (as input for results plotting)
- `results/` - here we store computed results and plots in the paper
  + to be added: performance results of each IA algorithm applied
  + to be added: figures presented in manuscript
- `src/` - here we place your python/matlab scripts 
  + to be added: one script for running each IA, e.g.
  +`01_Training_DT.py` - run the training for the Decision Tree algorithm
  + to be added: one script for perfomance testing of each IA, e.g.
  +`01_Testing_DT.py` - run the performance test for the Decision Tree algorithm
      run an Artificial Neuron Network algorithm
  + to be added: one script per additional analysis
  + scripts to reproduce the presented results (figures/tables) in manuscript
  + `F01_Results_ANN.py` - reproducing Figure 1 of the manuscript on results of hyperparameter testing for ANN algorithm
  + `F02_Results_RBFNN.py` - reproducing Figure 2 of the manuscript on results of hyperparameter testing for RBFNN algorithm
  + `F03_Results_Testing.py` - reproducing Figure 3 on comparison on performance testing of all applied algorithms
  + `F04_Results_ANN_Clogging.py` - reproducing Figure 4 of the manuscript on results of ANN algorithm performance for predicting clogging

## Python environment

To make the example reproducible, we provide the following files:
- `requirements.txt` - requirements for [pip](https://pip.pypa.io/en/stable/user_guide/#requirements-files) to install all needed packages

## Workflow

After finalizing our work, we tag the repository with a version like `v1.0`.

Then, a [Zenodo](https://zenodo.org/) release will be created, so we can cite the repository in our publication.

The `master` branch should always be kept in line with the latest release.
For further development we use the `develop` branch and update `master` with pull-requests.


## Contact

You can contact us via <a.zech@uu.nl>.


## License

MIT © 2021
