# H-tau-tau

This is an analysis of the Higgs boson decay into two tau leptons, performed using data from the ATLAS experiment at CERN. The final goal of the analysis was
to use a neural network in order to differantiate between the CP-odd and CP-even hypotheses for the Higgs boson, based on the four-momenta of its decay products.

### Data

The original .root files used for this analysis were created using Monte Carlo simulations. They contain records of the Higgs boson decaying into two tau leptons, and 
subsequently the leptons decaying into other charged/neutral products. The data includes information like: four-momenta of the decay products, decay modes of the tau leptons,
generated event weights for different CP scenarios (CP-even - 0&deg;, CP-odd - 90&deg; and combinations of those two for other angle values). The original data is only available for CERN users. The theory of this decay channel can be found in the article: [https://arxiv.org/pdf/2212.05833](https://arxiv.org/pdf/2212.05833).

### Files

| File                                          | Description                                                                                                                           |
|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| prepare.py  
preselection.py  
mc_to_csv.py       | Files meant for data preparation from .root files into data frames or .csv files as well as filtering the data.                        |
| plots.py  
distr.py                             | Various plots illustrating the data inside the files.                                                                                 |
| boost_rotation.py                             | Relativistic boost of the four-momenta of the leptons into the CoM frame + rotation so that the momenta are along the z-axis.                                                           |
| boost_rotation_categories.py  
nn_categories.py | Non-relativistic boost of the four-momenta into the CoM frame and a neural network learning to predict most probable CP angle for a given event.                                 |
| C123.py                                       | Calculating coefficients describing the relation between the weight and the CP angle using systems of linear equations for each event. |
