#!/usr/bin/env python3

# Place import files below
import copy

import h5py
import numpy as np

from process_data import UDGData
from universal_settings import (d_lg, fiducial_lg_mass,
                                generated_data_file_template, sim_ids,
                                sim_n_part, target_lg_masses, zoa_extent_list)


def main():
    # Load relevant simulation
    udg_data = UDGData()
    data_list = [
        'select_udgs_reff1_mu1', 'select_udgs_reff1_mu2',
        'select_udgs_reff2_mu1', 'select_udgs_reff2_mu2'
    ]
    (select_udgs_reff1_mu1, select_udgs_reff1_mu2, select_udgs_reff2_mu1,
     select_udgs_reff2_mu2) = udg_data.retrieve_data(data_list)

    udg_selection_criteria = [
        select_udgs_reff1_mu1, select_udgs_reff1_mu2, select_udgs_reff2_mu1,
        select_udgs_reff2_mu2
    ]
    udg_data_selection = [
        'Selection 1', 'Selection 2', 'Selection 3', 'Selection 4'
    ]

    # Prepare lists to hold generated data
    n_field_by_tmlg = []
    nudg_tot_by_tmlg = []
    nudg_mock_sdss_by_tmlg = [[] for _ in udg_data_selection]
    fudg_in_zoa_by_b = [[] for _ in udg_data_selection]

    # Iterate over simulations
    for sim_i, sim_id in enumerate(sim_ids):
        # Read in and compile relevant mock observational data with
        # respect to the MW and M31
        with h5py.File(generated_data_file_template.format(sim_n_part, sim_id),
                       'r') as h5_file:
            # Total number of UDGs and field galaxies rescaled to the
            # target M_LGs
            n_field_by_tmlg.append(h5_file['Rescaled N_field'][()])
            nudg_tot_by_tmlg.append(h5_file['Rescaled N_UDG,tot'][()])

            for d_i, d_selection in enumerate(udg_data_selection):
                rescaled_nudg_sdss_mw = h5_file['MW']['Rescaled N_UDG,SDSS'][
                    d_selection][()]
                rescaled_nudg_sdss_m31 = h5_file['M31']['Rescaled N_UDG,SDSS'][
                    d_selection][()]
                fudg_zoa_mw = h5_file['MW']['f_UDG in ZoA'][d_selection][()]
                fudg_zoa_m31 = h5_file['M31']['f_UDG in ZoA'][d_selection][()]

                # Compile mock SDSS observations and f_UDG in ZoA
                if sim_i > 0:
                    nudg_mock_sdss_by_tmlg[d_i] = np.concatenate(
                        (nudg_mock_sdss_by_tmlg[d_i],
                         np.concatenate(
                             (rescaled_nudg_sdss_mw, rescaled_nudg_sdss_m31))))
                    fudg_in_zoa_by_b[d_i] = np.concatenate(
                        (fudg_in_zoa_by_b[d_i],
                         np.concatenate((fudg_zoa_mw, fudg_zoa_m31))))
                else:
                    nudg_mock_sdss_by_tmlg[d_i] = np.concatenate(
                        (rescaled_nudg_sdss_mw, rescaled_nudg_sdss_m31))
                    fudg_in_zoa_by_b[d_i] = np.concatenate(
                        (fudg_zoa_mw, fudg_zoa_m31))

    # Compile data into single arrays
    n_field_by_tmlg = np.asarray(n_field_by_tmlg)
    nudg_tot_by_tmlg = np.asarray(nudg_tot_by_tmlg)
    nudg_mock_sdss_by_tmlg = np.asarray(nudg_mock_sdss_by_tmlg)
    fudg_in_zoa_by_b = np.asarray(fudg_in_zoa_by_b)

    ####################################################################
    # Print total number of field galaxies
    print("#" * 50)
    print("Total number of field galaxies")
    for z_i, b_extent in enumerate(target_lg_masses):
        med_field = np.nanmedian(n_field_by_tmlg[:, z_i])
        field_pm = np.around(np.sqrt(med_field))
        print("M_LG(< {0} Mpc) = {1:.0f} x 10^12 Msun: {2:d}^+{3:d}_-{3:d}".
              format(d_lg, b_extent / 1.e12, int(med_field), int(field_pm)))
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
    for z_i, b_extent in enumerate(target_lg_masses):
        med_field = np.nanmedian(nudg_tot_by_tmlg[:, z_i], axis=0)
        field_pm = np.around(np.sqrt(med_field))
        print_data = np.column_stack(
            (med_field, field_pm, field_pm)).reshape(len(med_field) * 3)
        print("{:.0f} x 10^12 Msun".format(b_extent / 1.e12) +
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
    for z_i, b_extent in enumerate(target_lg_masses):
        med_field = np.nanmedian(nudg_mock_sdss_by_tmlg[:, :, z_i], axis=1)
        percentiles = np.nanpercentile(
            nudg_mock_sdss_by_tmlg[:, :, z_i], [84., 16.], axis=1) - med_field
        print_data = np.column_stack(
            (med_field, *percentiles)).reshape(len(med_field) * 3)
        print("{:.0f} x 10^12 Msun ".format(b_extent / 1.e12) +
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

    print("-" * 50)
    print("Local Group mass, M_LG(< {} Mpc) = {:.0f} x 10^12 Msun".format(
        d_lg, fiducial_lg_mass / 1.e12))
    max_n = np.nanmax(nudg_mock_sdss_by_tmlg[:, :, select_mass])
    tot_obs = len(nudg_mock_sdss_by_tmlg[:, :, select_mass][0])
    print("Selection: " + selection_fields.format(*selection))
    for n in np.arange(np.nanmax(max_n) + 1):
        n_obs = (nudg_mock_sdss_by_tmlg[:, :, select_mass] <= n).sum(axis=1)
        frac = np.concatenate(n_obs / tot_obs)
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
    #     max_n = np.nanmax(nudg_mock_sdss_by_tmlg[:, :, t_i])
    #     tot_obs = len(nudg_mock_sdss_by_tmlg[:, :, t_i][0])
    #     print("Selection: " + selection_fields.format(*selection))
    #     for n in np.arange(np.nanmax(max_n) + 1):
    #         n_obs = (nudg_mock_sdss_by_tmlg[:, :, t_i] <= n).sum(axis=1)
    #         frac = n_obs / tot_obs
    #         fields = "{:.4f} " * len(frac)
    #         print("N = {:2d}:    ".format(int(n)) + fields.format(*frac))
    #     print()
    # print("#" * 50)

    ####################################################################
    # Print fraction of UDGs in ZoA
    print("Fraction of field UDGs coincident with ZoA")
    selection = np.arange(len(udg_selection_criteria)) + 1
    selection_fields = "{:>22}" * len(selection)
    print_fields = "{:>8.3f}^+{:.3f}_{:.3f}" * len(selection)
    print("ZoA extent" + " " * 6 +
          selection_fields.format(*udg_data_selection))
    for z_i, b_extent in enumerate(zoa_extent_list):
        med_field = np.nanmedian(fudg_in_zoa_by_b[:, :, z_i], axis=1)
        percentiles = np.nanpercentile(fudg_in_zoa_by_b[:, :, z_i], [84., 16.],
                                       axis=1) - med_field
        print_data = np.column_stack(
            (med_field, *percentiles)).reshape(len(med_field) * 3)
        print("|b| <= {:4.1f} deg ".format(b_extent) +
              print_fields.format(*print_data))
    print()

    return None


if __name__ == "__main__":
    main()
