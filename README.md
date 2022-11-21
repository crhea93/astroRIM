# astroRIM
Recurrent Inference Machine for Astronomy

[![pages-build-deployment](https://github.com/crhea93/astroRIM/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/crhea93/astroRIM/actions/workflows/pages/pages-build-deployment)

**Documentation**: 
https://crhea93.github.io/astroRIM/

## What is `astroRIM`
`astroRIM` is an easy-to-modify python module that enables users to employ the recurrent inference machine (RIM) for 1D and 2D problems in a modular manner. 
Users are provided with a standard set of 1D and 2D RIM configurations and likelihood modules; however, these can readily be modified to fit the user's specific architecture (see documentation for more details). 

## How to install `astroRIM`
We have tried to make the installation of `astroRIM` as smooth and painless as possible; however, if you have suggestions, please reach out to us.

Below are instructions for installing on a linux distribution (only tested on Ubuntu and Pop-OS!).
1. **Clone** this repository. I suggest cloning it in Documents or Applications.
    ```git clone https://github.com/crhea93/astroRIM.git```
2. **Enter repository** wherever you cloned it.
    ```cd astroRIM```
3. **Create** rim environment using the following command: `conda env create -f rim.yml`. Now, whenever you wish to use `astroRIM`, you can load the environment by simply typing the following into your terminal: `conda activate rim`.  

You may run into issues with `hdf5/h5py`. If this is the case, activate the rim environment and install `h5py` with `conda install h5py`.

If you are on a Mac, you will need to change step 3 slightly: 3. Create rim environment with conda create -n rim and then install the requirements with pip install -r requirements.txt.

## Where to find examples
Examples are paramount to the success of any open source code. Therefore, we have tried to make our examples as complete as possible. That said, we surely have forgotten something! If you wish to see an example that does not exist (or for an example to be better explained), please shoot us an email or open up an issue!

All examples can be found in two locations. Read-through examples can be found on our read the docs (https://crhea93.github.io/astroRIM/build/html/index.html) page while jupyter notebooks can be found in the *Notebooks* folder. I suggest starting with https://crhea93.github.io/astroRIM/build/html/Notebooks/RIM-Test-1D-Gaussians.html.


## Contributing
If you wish to contribute, that's awesome! Please shoot me an email at [carter.rhea@umontreal.ca](mailto:carter.rhea@umontreal.ca).
The easiest way to get involved is to make an issue or fork the repo, make your changes, and submit a well-documented pull request.

## Contact
If you have any questions about how to install, use, or modify `astroRIM`, please send an email to [Carter Rhea](mailto:carter.rhea@umontreal.ca).

## Copyright & License
2021 Carter Rhea ([carter.rhea@umontreal.ca](mailto:carter.rhea@umontreal.ca))

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see [https://www.gnu.org/licenses/](https://www.gnu.org/licenses/).


## License

## Citing `astroRIM`
 