
## Introduction

This repository contains simulation code from the paper Nurminen et al (2023). The code is written in Python and should be runnable on both Windows and Linux systems.
It heavily depends on the [mne-python](https://mne.tools/stable/index.html) library.

## Installation

The dependencies are easiest to handle using the [Anaconda distribution](https://www.anaconda.com/products/distribution). An Anaconda environment specification file `environment.yml` is included in the repository root.  Once you have Anaconda working, clone the repository and run `conda env create` in the repository directory. This should create a conda environment called `spfreq_sim`. You can activate it using `conda activate sqfreq_sim`.

## Configuration

Edit `paths.py` in the repository root. Change `DATA_PATH` to point to the directory where you cloned the repository, so that the code can find the necessary data files. Any generated figures will also be placed under the same directory.

Several of the figures require the ImageMagick package, specifically the `montage` command. On Linux systems, ImageMagick is often already installed. Check if the `montage` command is accessible by typing `montage` into a console. On Windows, install ImageMagick if necessary, and then edit the `MONTAGE_PATH` variable in `paths.py`.

## Running the code

The main simulation code is contained in `main.py`. The code is organized in Jupyter-style cells. Most IDEs, such as Spyder and Visual Studio Code, support  code cells. You can run individual cells (in top to bottom order, as they depend on each other). Alternatively, you can run the entire `main.py` in a Python console, in which case it will attempt to generate all the manuscript figures.

Before attempting to run the code, make sure to activate the right Anaconda environment first. Precisely how to do this depends on your IDE. If you run the code from a Python console, you can use the `conda activate` command as above.



