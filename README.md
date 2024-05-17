

# Vital-wave library

## Installing the package

Main dependencies:

* scipy 
  - pip install scipy
* pywt 
  - pip install PyWavelets
* numpy
  - pip install numpy

Key python constraint: 3.10

run pip install wherein the pyproject.toml file is located. Either use flag -e to enable editing of the source. 
The location can also be a github repository

    (base)  pip install -e . 

      * dynamically keeps the module up to date 
    
    (base)  pip install . 

      * requires re-installation as new features are produced 

A successful installation has the following input:

* Obtaining file:///C:/Users/marku/PycharmProjects/pythonProject2
* ...
* Successfully installed vitalwave-0.0.1

[optional] dist

    * python -m build 

[optional] install in virtual environment

    * conda activate

## Documentation

The main documentation is found at: 

* .\docs\\_build\html\index.htm. 

It is a sphinx generated document.

## Sphinx

The repository contains documentation generator, using sphinx.

To run initial run

    * sphinx-quickstart

To generating docs:

    * make.bat clean 
    * make.bat html

To Generating docs config(s) - first time:

    * sphinx-apidoc.exe -o . ..

Hand-maid modifications are needed with this command



## Running the module-code.

The syntax to running the code is, from vitalwave import:

  * basic_algos - collection of standalone functions
  * peak_detectors - standard methods of identifying peaks
  * features - few cyclic based metrics 
  * signal_quality - waveform quality based on Pearson-Coefficient
  * experimental - fiducial points

## to do

This is the working document for the Project.
- check quality indices
- check feature moduuli
- refaktorointi + test notebooks + checking fix comments in the code
- augmentation module
- time series outlier removals 
- respiration algot
- activity algot/models?
- segmentation?

