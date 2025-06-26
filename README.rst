bayes-plasma-tomo
============

Repository for bayesian tomography paper
----------------------------------------

This repository provides open access to all the computational routines implemented by our paper
"Tomography for Plasma Imaging: a Unifying Framework for Bayesian Inference". The paper was recently
submitted for publication. In the meanwhile, the arXiv version of this paper
can be found `here <https://arxiv.org/abs/2506.20232>`_ .

The code here provided can be used to reproduce all the results presented in the paper.

In particular, we provide instructions for:

* **Installation** ⚙️ : simple instructions to get started
* **Phantom generation** 👻 : how to generate the synthetic emissivity profiles used to validate our algorithms
* **Results generation** 📊 : scripts and Jupyter Notebooks allowing to reproduce analyses and plots

The Jupyter Notebooks allow to easily inspect and visualize how the results included in the paper were generated,
even without needing to actually re-run them. 🔎

The pipelines proposed in this paper rely on the open-source computational imaging framework
`Pyxu <https://github.com/pyxu-org/pyxu>`_ and its plugin `pyxu-diffops <https://github.com/dhamm97/pyxu-diffops>`_ ,
both (co-)developed by us: check them out!


Installation: let's get started! 🚀
------------

To clone the repo and install the necessary dependencies:

.. code-block:: bash

   git clone https://github.com/dhamm97/bayes-plasma-tomo.git
   pip install .


Phantom generation
----------------

We provide the instructions necessary to generate the synthetic emission profiles used in this work: feel
free to **reuse** them to validate your own tomographic reconstruction algorithms! ♻️

To generate them, you can navigate to the folder ``src/results/dataset_generation`` and run

.. code-block:: bash

   python generate_sxr_phantoms.py

The soft x-ray profiles will be generated and stored in the folder ``src/results/dataset_generation/sxr_samples``.

Results generation
--------------

Once you have generated the phantoms, we are ready to replicate the paper results.

* The notebook ``dataset_generation_plots.ipynb`` in folder ``src/results/dataset_generation`` shows the generation of the plots included in Section 5.1 of the paper.

* The notebook ``phantom_analysis_results.ipynb`` in folder ``src/results/phantom_anlaysis`` shows the generation of the plots included in Section 5.2 of the paper.
  If you want to reproduce them, see the next bullet point.

* In ``src/results/phantom_analysis/uq_study_results``, the repo only contains the postprocessed aggregate results. If you want to fully reproduce
  the analysis, you can run the python script ``uq_study.py``. Warning ⚠️ : running this script takes a long time. If you have access to a workstation
  featuring multiple cores, run the ``.sh`` version of the script, splitting the computations among (in our case) 10 different cores, which will
  reduce the computation time to a couple hours.

* In ``src/results/sparse_tomography_limits`` you will find the Jupyter Notebook allowing to reproduce the results from Section 5.3.

* In ``src/results/phantom_analysis/hyperparameter_tuning/prior_hyperparameters_tuning`` you will find the scripts necessary to reproduce the
  results already findable in its subfolder ``tuning_data``. The included Jupyter Notebook reports the results included in Appendix C of the paper.

* In ``src/results/phantom_analysis/hyperparameter_tuning/ula_hyperparameters_tuning`` you will find the script ``ula_iteration_number_tuning.py``,
  which can be used to generate the data for Figure 8 in Appendix C. Warning ⚠️ : this script too takes a pretty long time, since it computes
  :math:`10^7` MCMC iterations for 100 different phantoms. If you have access to a workstation featuring multiple cores, run the ``.sh`` version of the script.
  Probably easier to simply check the provided Jupyter Notebook in this case.


Finally, the folder ``src/tomo_fusion`` contains the computational routines and helping tools implemented in our work.

License
-------

Distributed under the terms of the `MIT`_ license,
``pyxu-diffops`` is free and open source software

Issues
------

Hopefully the provided instructions will be enough, but if you encounter any problems feel free to contact us!