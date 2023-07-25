# Persistence-diagram-guided-neural-network
The repository is part of the work on the PhD topic that focuses on Topology Feature Extraction to guide the classificaiton neural network.


## Installation

The code is written in Python 3.10. In order to reproduce the results, you need to install an virtual enviorment defined in the `environment.yml` file. This can be done by running the following command:

```bash
conda env create -f environment.yml
```

## Contents of the repository

The the code contains the following files:
* `main_PDGNN.py` - The main file that runs the code
* `plot_results.py` - The file that plots the results
* `PDGNN_model.py` - The file that contains the PDGNN class
* `model.py` - The file that contains the base neural network class
* `utils.py` - The file that contains the utility functions
* `/notebooks/` - The folder that contains the notebooks used for the development of the code

## Running the code

In order to run the code, you need to activate the virtual enviorment and then run the code. The code can be run by running the following command:

```bash
conda activate PDGNN
python main_PDGNN.py # To run the PDGNN classification
python main.py # To run the base neural network
```

## Results

The results of the can be viewed with `plot_results.py` script.

## Authors

* **Blaž Dobravec** - *Implementation and design work*
* **Jure Žabkar** - *Supervision - mentorship*
