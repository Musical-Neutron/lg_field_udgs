# Plot scripts for the HESTIA UDG paper

**Last reviewed:** v0.1

A set of scripts and a repository of reduced data to reproduce the plots in the
paper predicting the size of the ultra-diffuse galaxy (UDG) population in the
field of the Local Group using simulations from the HESTIA suite.

This README file is located in the main repository for the plotting scripts.
All plotting scripts should be executed from this directory.

## 1.0 Scripts

There are six scripts that can be executed independently:

* [generate_paper_results.py](/generate_paper_results.py)
  * Generates the data required for some of the figures in the paper.
  Primarily concerns preparing the data for the mock SDSS observations.
  * MUST be run first to ensure that the other scripts operate correctly.
* Fig. 1: [fig_01_plot_selection_criteria.py](/fig_01_plot_selection_criteria.py)
  * Plots effective radius, R<sub>e</sub>, vs. effective surface brightness,
  &mu;<sub>e</sub>, of all candidate UDGs.
* Fig. 3: [fig_03_plot_cumulative_rad_dist.py](/fig_03_plot_cumulative_rad_dist.py)
  * Plots the cumulative number of UDGs as a function of distance from the
  Milky Way (MW) halo analogue. By convention this is the smaller of the two
  primary haloes.
* Fig. 4: [fig_04_plot_luminosity_functions.py](/fig_04_plot_luminosity_functions.py)
  * Plots the luminosity function of the UDG populations as a function of
  _V_-band absolute and apparent magnitudes.
* Fig. 5: [fig_05_plot_mock_sdss_observations.py](/fig_05_plot_mock_sdss_observations.py)
  * Plots the luminosity functions of UDGs that are detectable in SDSS-like
  surveys in each HESTIA high resolution simulation.
* [print_paper_results.py](/print_paper_results.py)
  * Prints information relevant to the paper to stdout.

There is also a master script, [run_all_scripts.py](/run_all_scripts.py),
that will run all of the above scripts when executed. This produces .svg
and .pdf versions of each figure in the paper.

### Supplementary scripts
* [common_functions.py](/common_functions.py)
  * A set of functions common to more than one of the main scripts.
* [process_data.py](/process_data.py)
  * Contains classes and functions to handle basic processing of data.

## 2.0 Data

The [data](/data) directory that contains all files necessary to reproduce the
figures in the paper. There are eight files:

* [8192_09_18_z0_paper_data.hdf5](/data/8192_09_18_z0_paper_data.hdf5)
* [8192_17_11_z0_paper_data.hdf5](/data/8192_17_11_z0_paper_data.hdf5)
* [8192_37_11_z0_paper_data.hdf5](/data/8192_37_11_z0_paper_data.hdf5)
  * All contain the data on which the paper is based
* [8192_lg_2.5Mpc_volume_mass.csv](/data/8192_lg_2.5Mpc_volume_mass.csv)
  * Contains the masses of each LG volume
* [fattahi_2020_N_vs_M.csv](/data/fattahi_2020_N_vs_M.csv)
  * Contains relevant data from [Fattahi et al. (2020)](https://arxiv.org/abs/1907.02463)
* [k08_mag_d_data.csv](/data/k08_mag_d_data.csv)
* [k08_mu_d_data.csv](/data/k08_mu_d_data.csv)
  * Both contain relevant data from [Koposov et al. (2008)](https://arxiv.org/abs/0706.2687)
* [mcconnachie_2012_data.csv](/data/mcconnachie_2012_data.csv)
  * Contains data from [McConnachie et al. (2012)](https://arxiv.org/abs/1204.1562),
  including updates from the [Nearby Dwarf Database](https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/community/nearby/)
* [survey_data.csv](/data/survey_data.csv)
  * Contains data relevant to the SDSS (and other surveys)

## 3.0 Citations

This code and the accompanying data are freely available.

### If you use this code or derivative work

* [O. Newton et al. (2022)](http://arxiv.org/abs/2212.05066)
* Please cite this repository (the up-to-date version will be archived in
Zenodo upon acceptance)

### If you use these data, a derivative work, or results thereof

* [O. Newton et al. (2022)](http://arxiv.org/abs/2212.05066)
* Please cite this repository (the up-to-date version will be archived in
Zenodo upon acceptance)
* [N. Libeskind et al. (2020)](https://doi.org/10.1093/mnras/staa2541)

If you have any questions or would like help in using the scripts, please
email:
> onewton 'at' cft.edu.pl
