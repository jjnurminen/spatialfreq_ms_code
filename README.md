
## Instructions for use

- Create and activate a suitable Python environment
    - Easiest way is to download `environment.yml` from the package root and run `conda env create`
    - Activate the newly created environment by `conda activate spfreq`
- Clone repository and cd into the directory
- You should now be able to run the code. The simulation code resides in `main.py`.

## Generating figures

Most of the figures require the ImageMagick package. On Linux systems, it is
often already installed. Check if the `montage` command is accessible by typing
`montage` into a console. On Windows, install ImageMagick if necessary, and edit
the path to the montage command in `viz.py`.

