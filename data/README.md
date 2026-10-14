# Data

Data from the DTI solvent-based CO₂ capture pilot plant (Danish Technological Institute, Aarhus, Denmark) and the model parameter estimates used in the paper *Neural network-based Grey-box Modeling and Economic Model Predictive Control of a Solvent-based Carbon Capture Pilot Plant*.

The data are published on DTU Data (*link will be added*) and are not part of the code repository. Download them into this folder. `pilot_plant_series/` is not part of the download; driver 01 computes it from the raw logs. The data files were exported from the original project folders with [`../tools/export_data.jl`](../tools/export_data.jl).

```
data/
├── raw_pilot_plant/           Raw logs of the pilot plant from the measurement campaign in Nov-Dec 2025
├── pilot_plant_series/        10 step-response data series used for training and validation (Sections 2 and 6),
│                              computed from the raw logs by drivers/01_process_raw_data.jl
├── parameter_estimates/       Estimated model parameters for 11 NN regularization values (Sections 4 and 6)
└── closed_loop_tests/         Data from the four EMPC tests at the pilot plant (Section 7)
```

## `raw_pilot_plant/`

These are the raw logs of the pilot-plant logger from the measurement campaign in Nov–Dec 2025: five tab-separated text files (`111 - 2025-11-24 Default.txt` … `117 - 2025-12-03 Default.txt`) with one row per sample, about every 20 s. The first columns are the time stamp (`PC time`, local time, e.g. `24-11-2025 08:53:58,1942`) and the logger scan time, followed by all logged signals as `<tag> - [<unit>]`, with decimal commas. The Danish characters in the column names are not UTF-8 encoded, and [`../src/raw_data_processing.jl`](../src/raw_data_processing.jl) fixes them when loading.

## `pilot_plant_series/`

`series_1.csv` … `series_10.csv` are the 10 dynamic step-response data series from the measurement campaign in Nov–Dec 2025 (Table 2 in the paper). The series numbers are the ones used in the paper (e.g. Figure 4). [`../drivers/01_process_raw_data.jl`](../drivers/01_process_raw_data.jl) computes them from the raw logs as follows. It calculates the process variables (gas concentrations from the ideal gas law, flue gas flow from the mass flow, reboiler duty from the accumulated boiler energy, CO₂ mass flows, capture efficiency), adds the set points of the test plan, and cuts the data into 10 windows around the step changes. It is the same processing as in the original data analysis of the pilot plant, and gives identical series. The only exception is a column `ygA_in_SP`, which is left out; the model doesn't use it.

Each file has one row per sample (about 20 s sampling time). The drivers downsample the data to 2 min (every 6th sample).

The model uses these columns (see `load_data_sys_id_DTI` in [`../src/util_diffeq_sys_id.jl`](../src/util_diffeq_sys_id.jl)):

| Column | Description | Unit | Role in the model |
|---|---|---|---|
| `time` | Time stamp | – | Campaign identification |
| `t` | Time | h | Time (shifted to start at 0) |
| `ygA` | CO₂ concentration in the gas at the absorber outlet, $y_g$ | vol% | Output |
| `TlD_HX_out` | Desorber temperature, $T_d$ | °C | Output |
| `TlA_HX_out` | Absorber solvent inlet temperature | °C | Output (not used for campaign 4) |
| `P_reb` | Reboiler duty, $Q$ | kW | Output |
| `Fco` | Solvent flow rate, $F$ | m³/h | Input |
| `TlD_HX_out_SP` | Desorber temperature set point, $T_d^{sp}$ | °C | Input |
| `TlA_HX_out_SP` | Absorber solvent inlet temperature set point | °C | Input (not used for campaign 4) |
| `cginA` | CO₂ concentration in the flue gas at the absorber inlet, $c_{g,in}$ | mol/m³ | Disturbance |
| `FginA` | Flue gas flow rate, $F_{in}$ | m³/h | Disturbance |
| `cgA` | CO₂ concentration in the gas at the absorber outlet | mol/m³ | Used to compute the capture rate $N_a$ |

The remaining columns are additional measured or derived signals and set points from the processing. The model doesn't use them.

## `parameter_estimates/`

These are the parameter estimates of the NN-based grey-box model (4 neurons in the $N_d$ and $Q$ networks), computed on an HPC cluster with the estimation of [`../drivers/02_estimate_parameters.jl`](../drivers/02_estimate_parameters.jl). There is one folder per NN regularization value, $\lambda_{reg} = 10^{-4}, 10^{-3.5}, \ldots, 10^{1}$, named `log10_lambda_<value>`. Each folder contains:

| File | Training data |
|---|---|
| `in_sample.jld2` | All 10 series (in-sample estimate, $\theta^{(is)}$) |
| `oos_series_XX.jld2` | All series except series `XX` (leave-one-out estimate, $\theta^{(-XX)}$) |

[`index.csv`](parameter_estimates/index.csv) lists all 121 files with $\lambda_{reg}$, training series and the name of the original result file (`results/sys_id/config_025/M<model>/<run>.jld2` in the original project).

Each `.jld2` file contains plain Julia types only:

| Variable | Description |
|---|---|
| `p` | Estimated parameters: `p.Nd_params` and `p.Q_params` (NN weights, `layer_1.W`, `layer_1.b`, `layer_2.W`, `layer_2.b`), `p.Na_params` (`k1`, `k2`, `b`) and `p.theta_D` |
| `x0` | Estimated initial states $[c_a, c_d, T_d]$ of each training series |
| `train_series` | Series used for training |
| `lambda_reg` | NN regularization parameter |
| `original_file` | Original result file |

To load a file in Julia:

```julia
using JLD2
f = jldopen("data/parameter_estimates/log10_lambda_-1.0/in_sample.jld2")
p, x0 = f["p"], f["x0"]
close(f)
```

The files are HDF5 files and can also be read with, e.g., `h5py` in Python.

The model used in the EMPC demonstration (Section 7) is the in-sample estimate with $\lambda_{reg} = 0.1$, `log10_lambda_-1.0/in_sample.jld2`.

## `closed_loop_tests/`

These are the data from the four control tests at the pilot plant in April 2026. [`tests.csv`](closed_loop_tests/tests.csv) lists the start time of each test (6 h duration):

| Test | Folder | Scenario |
|---|---|---|
| #1 | `test_1_baseline_open_loop` | Baseline (constant energy price), open-loop |
| #2 | `test_2_flexible_open_loop` | Flexible operation (varying energy price), open-loop |
| #3 | `test_3_baseline_closed_loop` | Baseline, closed-loop EMPC |
| #4 | `test_4_flexible_closed_loop` | Flexible operation, closed-loop EMPC |

Each folder contains:

- **`measurements.csv`**: raw data from the plant historian in long format (`ts`, `sensor`, `unit`, `value`). Time stamps are local time (CEST, UTC+02:00). The file keeps the sensors used in the evaluation, from the start of the logged file to 1 h after the end of the test. The sensors are mapped to variables in `choose_tags_and_reshape_long_df` in [`../src/closed_loop_data.jl`](../src/closed_loop_data.jl):

  | Sensor | Variable | Description |
  |---|---|---|
  | `CO2_abs_in - [%]` | `ygina` | CO₂ concentration in the flue gas at the absorber inlet [vol%] |
  | `Flow_før abs - [kg/h]` | `mgina` | Flue gas mass flow rate at the absorber inlet [kg/h] |
  | `CO2_tør_abs - [%]` | `yga` | CO₂ concentration (dry) in the gas at the absorber outlet [vol%] |
  | `A010_BT02 - [°C]` | `Tla_HX` | Absorber solvent inlet temperature [°C] |
  | `A020_BT01 - [°C]` | `Tgina` | Flue gas temperature at the absorber inlet [°C] |
  | `A020_BT02 - [°C]` | `Tga` | Gas temperature at the absorber outlet [°C] |
  | `A060_BT02 - [°C]` | `Tld_HX` | Desorber temperature [°C] |
  | `Damp_kedel - [kWh]` | `Q_acc` | Accumulated energy of the steam boiler supplying the reboiler [kWh] |
  | `A050_BF01 - [m3/h]` | `F` | Solvent flow rate [m³/h] |
  | `Flue_inlet_temp - [°C]`, `Røg - [°C]`, `Effekt - [kW]` | – | Boiler signals (loaded, not used in the evaluation) |

- **`inputs.csv`**: the applied set points. `time_h` is the time since the start of the test [h], `F_sp` the solvent flow rate [m³/h] and `Td_sp` the desorber temperature set point [°C]. For the open-loop tests, these are the precomputed input trajectories. For the closed-loop tests, they are the first element of the input trajectory computed by the EMPC in each 10 min control interval.
- **`controller_measurements.csv`** (test #4 only): the plant data as the controller received it online during the test, in the same format as `measurements.csv` (only the 9 sensors loaded by the controller). The controller read the plant data from a database that was filled online. For test #3, this data is identical to `measurements.csv`. For test #4, the historian export in `measurements.csv` has a slightly different sampling. The replay of the controller ([`../drivers/09_empc_closed_loop.jl`](../drivers/09_empc_closed_loop.jl)) uses this file, while the evaluation of the tests (Figures 13 and 14) uses `measurements.csv`, as in the paper.
- **`controller_outputs.json`** (closed-loop only): the full optimal input trajectories computed by the EMPC in each control interval, as logged by the controller (`t_hours`, `u_opt = [F trajectory, Td_sp trajectory]`).
- **`solver_stats.json`** (closed-loop only): IPOPT solver statistics for each control interval (solve time, termination status, objective value), as logged by the controller.
