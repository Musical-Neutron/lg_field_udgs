#!/usr/bin/env python3

# Place import files below
import copy

import h5py
import numpy as np

from common_functions import survey_cone
from process_data import UDGData
from universal_settings import (d_lg, fiducial_lg_mass,
                                generated_data_file_template, sim_ids,
                                sim_n_part, survey_data_file)


def main():
    # Analysis settings
    np.random.seed(10)
    target_lg_masses = np.asarray([7.e12, fiducial_lg_mass, 9.e12])  # Msun

    # Load relevant simulation
    udg_data = UDGData()
    data_list = [
        'simulation_id', 'select_udgs_reff1_mu1', 'select_udgs_reff1_mu2',
        'select_udgs_reff2_mu1', 'select_udgs_reff2_mu2',
        'select_gal_mstar_nstar'
    ]
    (simulation_id, select_udgs_reff1_mu1, select_udgs_reff1_mu2,
     select_udgs_reff2_mu1, select_udgs_reff2_mu2,
     select_gal_mstar_nstar) = udg_data.retrieve_data(data_list)

    udg_selection_criteria = [
        select_udgs_reff1_mu1, select_udgs_reff1_mu2, select_udgs_reff2_mu1,
        select_udgs_reff2_mu2
    ]
    udg_data_selection = [
        'Selection 1', 'Selection 2', 'Selection 3', 'Selection 4'
    ]

    # Load SDSS survey data
    survey_data = np.genfromtxt(survey_data_file)
    sdss_surveyArea, _ = survey_cone(survey_data[:, 0][0])

    # Prepare lists to hold generated data
    nudg_by_sel_tmlg_list = [[[] for _ in np.arange(len(udg_data_selection))]
                             for _ in np.arange(len(target_lg_masses))]
    n_field_by_t_mlg = []
    nudg_s_crit_sim_t_mlg_list = [
        np.empty((len(udg_data_selection), len(sim_ids)))
        for _ in np.arange(len(target_lg_masses))
    ]
    # Iterate over simulations
    for sim_i, sim_id in enumerate(sim_ids):
        # Select the galaxies in the relevant simulation
        galaxies_in_sim = simulation_id == sim_id
        field_in_sim = (galaxies_in_sim * select_gal_mstar_nstar).sum()
        n_udgs_in_sim = np.asarray([(galaxies_in_sim * udg_selection).sum()
                                    for udg_selection in udg_selection_criteria
                                    ])

        # Read in and compile relevant mock observational data with
        # respect to the MW and M31
        n_udgs_mw_list = []
        n_udgs_m31_list = []
        with h5py.File(generated_data_file_template.format(sim_n_part, sim_id),
                       'r') as h5_file:
            for d_selection in udg_data_selection:
                n_udg_sdss_mw = h5_file['MW']['N_UDG,SDSS in SDSS'][
                    d_selection][()]
                n_udg_sdss_m31 = h5_file['M31']['N_UDG,SDSS in SDSS'][
                    d_selection][()]
                n_udgs_mw_list.append(n_udg_sdss_mw)
                n_udgs_m31_list.append(n_udg_sdss_m31)
            rescale_factors = h5_file['LG mass rescale factors'][()]

        # Lists of N_UDG,SDSS wrt. MW and M31 as a function of selection
        # criteria
        n_udgs_mw_list = np.asarray(n_udgs_mw_list)
        n_udgs_m31_list = np.asarray(n_udgs_m31_list)
        mw_medians = np.nanmedian(n_udgs_mw_list, axis=1)
        m31_medians = np.nanmedian(n_udgs_m31_list, axis=1)

        # Select factors to rescale N_UDG as a function of LG mass for
        # the masses in target_lg_masses
        selected_rescale_factors = np.concatenate([
            rescale_factors[:, 1][rescale_factors[:, 0] == target_mass]
            for target_mass in target_lg_masses
        ])

        # Rescale N_field and N_UDG
        rescaled_n_field = np.around(field_in_sim * selected_rescale_factors)
        rescaled_n_udgs = np.around(
            (n_udgs_in_sim * selected_rescale_factors[:, np.newaxis]))
        n_field_by_t_mlg.append(rescaled_n_field)

        # Rescale median N_UDGs(<2.5 Mpc) wrt. MW and M31
        rescaled_n_udgs_mw = round_to_nearest_multiple(
            mw_medians * selected_rescale_factors[:, np.newaxis],
            0.5,
            decimal_precision=1)
        rescaled_n_udgs_m31 = round_to_nearest_multiple(
            m31_medians * selected_rescale_factors[:, np.newaxis],
            0.5,
            decimal_precision=1)

        ################################################################
        # Rescale mock SDSS observations to the target LG masses
        rescaled_nudg_mw_sdss = [[] for _ in np.arange(len(target_lg_masses))]
        rescaled_nudg_m31_sdss = [[] for _ in np.arange(len(target_lg_masses))]
        # Iterate over list of target LG masses
        for t_i, t_mass in enumerate(target_lg_masses):
            nudg_s_crit_sim_t_mlg_list[t_i][:, sim_i] = rescaled_n_udgs[t_i]
            # Iterate over UDG selection criteria
            for s_i in np.arange(len(udg_selection_criteria)):
                t_med_mw = rescaled_n_udgs_mw[t_i][s_i]
                t_med_m31 = rescaled_n_udgs_m31[t_i][s_i]

                # Rescale mock observations
                rescaled_mw_obs = rescale_observations(
                    n_udgs_mw_list[s_i], t_med_mw, sdss_surveyArea,
                    n_udgs_in_sim[s_i] * 0.82)
                rescaled_m31_obs = rescale_observations(
                    n_udgs_m31_list[s_i], t_med_m31, sdss_surveyArea,
                    n_udgs_in_sim[s_i] * 0.82)
                rescaled_nudg_mw_sdss[t_i].append(rescaled_mw_obs)
                rescaled_nudg_m31_sdss[t_i].append(rescaled_m31_obs)

                # Compile data into one list
                nudg_by_sel_tmlg_list[t_i][s_i] = np.concatenate(
                    (nudg_by_sel_tmlg_list[t_i][s_i], rescaled_mw_obs,
                     rescaled_m31_obs))

    # Compile field galaxy observations into one array
    n_field_by_t_mlg = np.asarray(n_field_by_t_mlg)

    ####################################################################
    # Print total number of field galaxies
    print("#" * 50)
    print("Total number of field galaxies")
    for t_i, t_mass in enumerate(target_lg_masses):
        med_field = np.nanmedian(n_field_by_t_mlg[:, t_i])
        field_pm = np.around(np.sqrt(med_field))
        print("M_LG(< {0} Mpc) = {1:.0f} x 10^12 Msun: {2:d}^+{3:d}_-{3:d}".
              format(d_lg, t_mass / 1.e12, int(med_field), int(field_pm)))
    print()

    ####################################################################
    # Print total number of field UDGs
    print("#" * 50)
    print("Total number of field UDGs")
    selection = np.arange(len(udg_selection_criteria)) + 1
    selection_fields = "{:>12}" * len(selection)
    print_fields = "{:>7d}^+{:d}_-{:d}" * len(selection)
    print("M_LG(< {0} Mpc)".format(d_lg) +
          selection_fields.format(*udg_data_selection))
    for t_i, t_mass in enumerate(target_lg_masses):
        med_field = np.nanmedian(nudg_s_crit_sim_t_mlg_list[t_i], axis=1)
        field_pm = np.around(np.sqrt(med_field))
        print_data = np.column_stack(
            (med_field, field_pm, field_pm)).reshape(len(med_field) * 3)
        print("{:.0f} x 10^12 Msun".format(t_mass / 1.e12) +
              print_fields.format(*np.int16(print_data)))
    print()

    ####################################################################
    # Print predicted number of observed field UDGs in SDSS footprint
    print("#" * 50)
    print("Number of field UDGs detectable in SDSS")
    selection = np.arange(len(udg_selection_criteria)) + 1
    selection_fields = "{:>12}" * len(selection)
    print_fields = "{:>6d}^+{:d}_{:d}" * len(selection)
    print("M_LG(< {0} Mpc)".format(d_lg) +
          selection_fields.format(*udg_data_selection))
    for t_i, t_mass in enumerate(target_lg_masses):
        med_field = np.nanmedian(nudg_by_sel_tmlg_list[t_i], axis=1)
        percentiles = np.nanpercentile(nudg_by_sel_tmlg_list[t_i], [84., 16.],
                                       axis=1) - med_field
        print_data = np.column_stack(
            (med_field, *percentiles)).reshape(len(med_field) * 3)
        print("{:.0f} x 10^12 Msun ".format(t_mass / 1.e12) +
              print_fields.format(*np.int16(print_data)))
    print()

    ####################################################################
    # Print chance of finding N or fewer detectable UDGs in the SDSS
    # footprint (fiducial LG mass only)
    print("#" * 50)
    print("Chance of finding N or fewer detectable UDGs in the SDSS " +
          "footprint")
    selection_fields = "{:<7}" * len(selection)

    select_mass = target_lg_masses == fiducial_lg_mass
    t_mass = target_lg_masses[select_mass][0]

    print("-" * 50)
    print("Local Group mass, M_LG(< {} Mpc) = {:.0f} x 10^12 Msun".format(
        d_lg, t_mass / 1.e12))
    max_n = np.nanmax(np.asarray(nudg_by_sel_tmlg_list)[select_mass][0])
    tot_obs = len(np.asarray(nudg_by_sel_tmlg_list)[select_mass][0][0])
    print("Selection: " + selection_fields.format(*selection))
    for n in np.arange(np.nanmax(max_n) + 1):
        n_obs = (np.asarray(nudg_by_sel_tmlg_list)[select_mass][0] <= n).sum(
            axis=1)
        frac = n_obs / tot_obs
        fields = "{:.4f} " * len(frac)
        print("N = {:2d}:    ".format(int(n)) + fields.format(*frac))
    print()
    print("#" * 50)

    ####################################################################
    # # Same as above but for the full list of target LG masses
    # for t_i, t_mass in enumerate(target_lg_masses):
    #     print("-" * 50)
    #     print("Local Group mass, M_LG(< {} Mpc) = {:.0f} x 10^12 Msun".format(
    #         d_lg, t_mass / 1.e12))
    #     max_n = np.nanmax(combined_list[t_i])
    #     tot_obs = len(combined_list[t_i][0])
    #     print("Selection: " + selection_fields.format(*selection))
    #     for n in np.arange(np.nanmax(max_n) + 1):
    #         n_obs = (np.asarray(combined_list[t_i]) <= n).sum(axis=1)
    #         frac = n_obs / tot_obs
    #         fields = "{:.4f} " * len(frac)
    #         print("N = {:2d}:    ".format(int(n)) + fields.format(*frac))
    #     print()
    # print("#" * 50)

    return None


def rescale_observations(observations, target_median, probability, n_udg_sdss):
    """Rescale mock SDSS observations for a different LG mass

    Args:
        observations (arr, len(N)): Array of observations of the total
            UDG population, N_UDG,SDSS.
        target_median (fl): The target median the distribution should
            satisfy.
        probability (fl/arr): When adding additional UDGs to the
            observations, the probability that a given UDG will be in
            the footprint of the survey.
        n_udg_sdss (fl): The total number of UDGs in the simulation that
            are potentially detectable by the survey.

    Returns:
        arr, len(N): Rescaled observations of N_UDG,survey
    """
    new_observations = copy.deepcopy(observations)
    new_n_udg = copy.deepcopy(n_udg_sdss)
    current_median = np.nanmedian(new_observations)
    # Scale observations down
    if current_median > target_median:
        while current_median > target_median:
            # Weight preferentially towards high-N observations that
            # contain a larger fraction of the total number of
            # potentially observable UDGs
            weights = new_observations / new_n_udg
            # Only remove UDGs from observations with N_UDG > 0
            weights[np.isinf(weights)] = -0.01
            new_observations = new_observations - (
                np.ones(len(new_observations)) *
                (np.random.rand(len(new_observations)) <= weights))
            # Ensure that non-physical mock observations are set to 0
            new_observations[new_observations < 0] = 0
            # Recompute median and the total number of UDGs
            current_median = np.nanmedian(new_observations)
            new_n_udg -= 1
    # Scale observations up
    if current_median < target_median:
        while current_median < target_median:
            new_observations = new_observations + (
                np.ones(len(new_observations)) *
                (np.random.rand(len(new_observations)) <= probability))
            current_median = np.nanmedian(new_observations)

    return new_observations


def round_to_nearest_multiple(values,
                              multiple,
                              direction='up',
                              decimal_precision=2):
    """Rounds numbers up/down towards the nearest multiple

    Args:
        values (arr, len(N)): Values to round.
        multiple (fl): Multiple to round values towards.
        direction (str, optional): Round 'up' or round 'down'. Defaults
            to 'up'.
        decimal_precision (int, optional): The number of decimals to
            retain after rounding. Defaults to 2.

    Raises:
        ValueError: If direction is neither 'up' nor 'down'

    Returns:
        arr, len(N): Rounded values
    """
    direction_allowed_values = ['up', 'down']
    if direction not in direction_allowed_values:
        raise ValueError(
            "direction must be one of: {}".format(direction_allowed_values))

    if direction == 'up':
        rounded_values = np.ceil(
            np.round(np.asarray(values) / multiple,
                     decimal_precision)) * multiple
    if direction == 'down':
        rounded_values = np.floor(
            np.round(np.asarray(values) / multiple,
                     decimal_precision)) * multiple

    return rounded_values


if __name__ == "__main__":
    main()
