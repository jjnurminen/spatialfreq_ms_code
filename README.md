
## Introduction

This repository contains simulation code from the paper Nurminen et al (2023). The code is written in Python. It heavily depends on the mne-python library.

## Installation

The dependencies are easiest to handle using Anaconda (or Miniconda). An Anaconda environment specification file `environment.yml` is included with the repository. Once you have Anaconda working, clone the repository, change to the repository directory and run `conda env create`. This should use the specification file to create a conda environment called `spfreq`. Activate it using `conda activate sqfreq`.

## Configuration

Edit `paths.py` in the repository root. Change `DATA_PATH` to point to the directory where you cloned the repository. This is needed so that the code can find the necessary data files. Any generated figures will also be placed under the same directory.

## Running the code






## Generating figures

Most of the figures require the ImageMagick package. On Linux systems, it is
often already installed. Check if the `montage` command is accessible by typing
`montage` into a console. On Windows, install ImageMagick if necessary, and edit
the path to the montage command in `viz.py`.

