
## Introduction

This repository contains simulation code from the paper Nurminen et al (2023). The code is written in Python. It heavily depends on the [mne-python](https://mne.tools/stable/index.html) library.

## Installation

The dependencies are easiest to handle using the Anaconda distribution, so you should first install that. An Anaconda environment specification file `environment.yml` is included with the repository. Once you have Anaconda working, clone the repository, change to the repository directory and run `conda env create`. This should use the specification file to create a conda environment called `spfreq`. Activate it using `conda activate sqfreq`.

## Configuration

Edit `paths.py` in the repository root. Change `DATA_PATH` to point to the directory where you cloned the repository. This is needed so that the code can find the necessary data files. Any generated figures will also be placed under the same directory.

Several of the figures require the ImageMagick package, specifically the `montage` command. On Linux systems, ImageMagick is often already installed. Check if the `montage` command is accessible by typing `montage` into a console. On Windows, install ImageMagick if necessary, and then edit the `MONTAGE_PATH` variable in `paths.py`.

## Running the code

 Before attempting to run the code, make sure to activate the right Anaconda environment, as described above.

The main simulation code is contained in `main.py`. The code is organized in code cells. Many IDEs such as Spyder and Visual Studio Code support running individual cells. You can run individual cells (in top to bottom order, as they depend on each other). Alternatively, you can run the entire `main.py` in a Python console, in which case it will attempt to generate all the manuscript figures.

