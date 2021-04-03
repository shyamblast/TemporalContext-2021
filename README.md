# TemporalContext [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4661549.svg)](https://doi.org/10.5281/zenodo.4661549)
Implementation of algorithm described in

> Shyam Madhusudhana et al. **(2021)**. "Improve detection of call sequences with temporal context." (_under review_)

for improving a CNN's ability to detect animal call sequences by attaching a recurrent network to it.

The user interface is made available as a set of jupyter notebooks to accomplish the different aspects of the workflow. A subset of the data used in the research paper is available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4661494.svg)](https://doi.org/10.5281/zenodo.4661494). Download and extract the contents of the archive on your local storage.

#### Setting up

* After cloning the repository, install the required python packages listed in [requirements.txt](requirements.txt) (`pip3 install -r requirements.txt`). Based on your preference for using _tensorflow_ or _tensorflow-gpu_ package, make sure you uncomment the appropriate line in [requirements.txt](requirements.txt) before running 'pip install ...'.
* Update `raw_data_root` in [temporalcontext/settings.py](temporalcontext/settings.py) to point to the local directory where you extracted the dataset.
* Change `project_root` in [temporalcontext/settings.py](temporalcontext/settings.py) to point to a local directory where generated intermediate files, models and results will be placed.

#### Training and Testing 

This stage requires you to run the jupyter notebooks in the following order.

* Run [input_prep-cnn.ipynb](input_prep-cnn.ipynb) to process the audio data and save spectrograms of fixed-length segments for training.
* Run [train_eval-cnn.ipynb](train_eval-cnn.ipynb) to train the base CNN models.
* Run [input_prep-hybrid.ipynb](input_prep-hybrid.ipynb). It uses the trained base CNN models to generate training and validation data for all three hybrid variants.
* The code in [train_eval-hybrid.ipynb](train_eval-hybrid.ipynb) will train the hybrid models. You need to change the variable `lstm_type` in the notebook and run it once per hybrid model type.
* Run [test.ipynb]() to make predictions on the test data (for each fold) with all of the trained models.

#### Generating results

The testing code in the above step only saves the scores output by the trained models. To produce the Precision-Recall curves and F1-score vs. Threshold curves (as shown in the article) run:
```shell
$> python3 generate_results.py
```
All the result figures will be produced in the `perf_results_dir` subdirectory set under `project_root`.
