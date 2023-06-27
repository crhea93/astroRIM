.. astroRIM documentation master file, created by
   sphinx-quickstart on Fri Nov 11 11:55:53 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to astroRIM's documentation!
====================================


astroRIM is an easy-to-modify python module that enables users to employ the recurrent inference machine (RIM) for 1D and 2D problems in a modular manner. 
Users are provided with a standard set of 1D and 2D RIM configurations and likelihood modules; however, these can readily be modified to fit the user's specific architecture.

In this documentation, you will find the outline of the methodology of the RIM. Additionally, you will be able to see the standard architectures and how
to quickly build your own. Examples include using a RIM to solve a 1D and 2D denconvolution problem.

You can find the API documentation here as well.

Users are strongly encouraged to use `Clusters/run_RIM.py` as a template for training and testing the RIM since it contains all the required calls and creates 
plots for the training vs. validation loss, the learning rate, and an example solution from the test set. We have also included an ipython notebook to
analyse the results of a trained RIM in `Notebooks/Analysis.ipynb`. Other example notebooks can be found in the directory entitled `Notebooks`.

`run_RIM.py` can be called as:
.. code-block::
   python Clusters/run_RIM.py Clusters/example.i

where `Clusters/example.i` contains the network's parameters, the location of the data, and data descriptions. Please read the example (`Clusters/example.i`) to see a complete list of
variables.

Files to Modify 
^^^^^^^^^^^^^^^
If you wish to implement a modified version of the RIM (i.e. not with our standard architecture or not using the standard likelihood function), this can be
easily done by modifying the following two files: `rim_model.py` and `rim_physical.py`. Please see the page: :doc:`modifications`.


.. toctree::
   :maxdepth: 2
   :caption: Python Files:

   rim_sequence
   rim_model 
   rim_physical

Examples
^^^^^^^^
    .. toctree::
       :maxdepth: 2
       :caption: Example Modules:

       Notebooks/RIM-Test-1D-Gaussians
       Notebooks/RIM-Test-2D-Gaussians
       Notebooks/RIM-Deconvolving-Spectra



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
