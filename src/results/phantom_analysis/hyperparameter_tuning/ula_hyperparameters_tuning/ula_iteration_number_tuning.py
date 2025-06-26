import numpy as np
import os
import copy
import time
import sys
import pyxu.operator as pyxop
from pyxu.experimental.sampler._sampler import ULA
from pyxu.experimental.sampler.statistics import OnlineMoment, OnlineVariance

from src.tomo_fusion.reg_param_est import ProxFuncMoreau

import src.tomo_fusion.tools.helpers as tomo_helps
import src.tomo_fusion.functionals_definition as fct_def

dirname = os.path.dirname(__file__)


def run_ula_checkpoints(reg_param,
            phantom_dir,
            saving_dir,
            phantom_indices,
            burn_in,
            with_pos_constraint=False,
            clip_iterations=None,
            samples=int(1e6),
            writeback_checkpoints=None,
            seed=0):
    # set writeback rate to default value if not provided
    if writeback_checkpoints is None:
        writeback_checkpoints = (10**np.linspace(0, np.log10(samples), int(np.log10(samples))+1)).astype(int)
    # Load phantom data
    psis = np.load(phantom_dir+'/psis.npy')
    sxr_samples = np.load(phantom_dir+'/sxr_samples_with_background.npy')
    alphas = np.load(phantom_dir+'/alpha_random_values.npy')
    trim_val = np.load(phantom_dir+'/trimming_values.npy')

    reconstruction_shape = (120, 40)

    for phantom_id in phantom_indices:
        print("phantom ", phantom_id)
        ground_truth = copy.deepcopy(sxr_samples[phantom_id, :, :].squeeze())
        psi = psis[phantom_id, :, :]
        alpha = alphas[phantom_id]
        trim_val_ = trim_val[phantom_id, :]
        mask_core = tomo_helps.define_core_mask(psi=psi, dim_shape=reconstruction_shape, trim_values_x=trim_val_)

        # anisotropic regularization functional
        reg_fct_type = "anisotropic"
        # Define functionals
        f, g = fct_def.define_loglikelihood_and_logprior(ground_truth=ground_truth, psi=psi,
                                                         sigma_err=sigma_err, reg_fct_type=reg_fct_type,
                                                         alpha=alpha, plot=False,
                                                         seed=phantom_id)

        # define ula objective function
        ula_obj = f + reg_param * g
        pos_constraint = pyxop.PositiveOrthant(dim_shape=f.dim_shape) if with_pos_constraint else None
        if with_pos_constraint:
            # add positivity constraint to ula objective function
            ula_obj += ProxFuncMoreau(pos_constraint, mu=1e-3)
        if clip_iterations == "core":
            clipping_mask = mask_core
        else:
            clipping_mask = None
        # initialize ULA sampler
        gamma = 0.98 / ula_obj.diff_lipschitz
        ula = ULA(f=ula_obj, gamma=gamma)
        x0 = np.zeros(ula_obj.dim_shape[1:])
        rng = np.random.default_rng(seed=seed)
        gen_ula = ula.samples(x0=x0, rng=rng)
        # moments wrt mean
        mean_ula = OnlineMoment(order=1)
        var_ula = OnlineVariance()
        mean_prad_ula_core = OnlineMoment(order=1)
        var_prad_ula_core = OnlineVariance()
        ula_obj_values = np.zeros(int((samples+burn_in)/10), dtype=np.float16)

        # Run ULA
        print("Running {} ULA iterations".format(samples))

        # initialize data
        data = {}
        data["tomo_data"], data["noisy_tomo_data"], data["sigma_err"] = f.tomo_data, f.noisy_tomo_data, f.sigma_err
        data["reg_param"], data["sampling"] = reg_param, g.sampling
        if 'Anis' in g._name:
            data["alpha"] = g.diffusion_coefficient.alpha
        data["with_pos_constraint"], data["clip_iterations"] = with_pos_constraint, clip_iterations
        data["samples"] = samples
        # start timer
        start_time = time.time()

        # perform burn in iterations
        for i in range(burn_in):
            sample = next(gen_ula)
            if clipping_mask is not None:
                ula.x = clipping_mask * sample
            if i % 10 == 0:
                ula_obj_values[int(i/10)] = float(f(sample) + reg_param * np.sum(g.grad(sample) * sample) / 2)

        for i in range(samples):
            sample = next(gen_ula)
            if clipping_mask is not None:
                ula.x = clipping_mask * sample

            if (i+1) % int(1e5) == 0:
                print("iteration ", i+1)

            # update central moments
            mean, var = mean_ula.update(sample), var_ula.update(sample)
            prad_core = tomo_helps.compute_radiated_power(sample, mask_core, g.sampling)
            prad_core = np.array([prad_core])
            mean_prad_core, var_prad_core = mean_prad_ula_core.update(prad_core), var_prad_ula_core.update(prad_core)
            if i % 10 == 0:
                ula_obj_values[int(i/10 + burn_in/10)] = float(f(sample) + reg_param * np.sum(g.grad(sample) * sample) / 2)

            if (i+1) in writeback_checkpoints:
                data["mean_"+str(i+1)], data["var_"+str(i+1)] = copy.deepcopy(mean), copy.deepcopy(var)
                data["mean_prad_" + str(i + 1)], data["var_prad_" + str(i + 1)] = copy.deepcopy(mean_prad_core), copy.deepcopy(var_prad_core)
                data["time_"+str(i+1)] = time.time() - start_time

        data["ula_obj_values"] = ula_obj_values

        np.save(saving_dir+"ula_data_"+str(phantom_id)+".npy", data)

    return


if __name__ == '__main__':

    argv = sys.argv
    if len(argv) == 1:
        phantom_indices = np.arange(900, 910)
    elif len(argv) == 2:
        phantom_indices = int(argv[1])
        print("Running ULA study on phantom-{}".format(int(argv[1])))
    elif len(argv) == 3:
        phantom_indices = np.arange(int(argv[1]), int(argv[2]))
        print("Running ULA study on phantoms {}-{}".format(int(argv[1]), int(argv[2])))
    else:
        raise ValueError("Number of passed arguments must be either 1, 2 or 3")

    # define directory where phantoms are stored
    phantom_dir = '../../../dataset_generation/sxr_samples'

    sigma_err = np.load('../hyperparameter_tuning/prior_hyperparameters_tuning/tuning_data/reg_param_tuning_sigma005/sigma_err.npy')
    reg_param_median = 0.1
    # saving directory
    saving_dir = 'ula_iterations_number_tuning_sigma005/'
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)
    run_ula_checkpoints(reg_param=reg_param_median, phantom_dir=phantom_dir, saving_dir=saving_dir,
                        phantom_indices=phantom_indices,
                        burn_in=int(1e3),
                        with_pos_constraint=True, clip_iterations="core",
                        writeback_checkpoints=np.array([int(1e1), int(2.5e1), int(5e1), int(7.5e1),
                                                        int(1e2), int(2.5e2), int(5e2), int(7.5e2),
                                                        int(1e3), int(2.5e3), int(5e3), int(7.5e3),
                                                        int(1e4),int(2.5e4), int(5e4), int(7.5e4),
                                                        int(1e5),int(2.5e5), int(5e5), int(7.5e5),
                                                        int(1e6),int(2.5e6), int(5e6), int(7.5e6),
                                                        int(1e7)]),
                        samples=int(1e7))
