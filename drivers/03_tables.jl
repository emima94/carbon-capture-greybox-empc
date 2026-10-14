# ------------------------------------------------------------------------------------------
# Table 2: Overview of the experimental datasets from the DTI pilot plant
# Table A.5: Normalization parameter values used in the model
#
# Output: results/table_2_datasets.csv, results/table_A5_normalization.csv
# ------------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "..", "src", "setup.jl"))

datasets, _ = load_pilot_plant_data()

## Table 2
cg_tot = 101325 / (8.314 * (35 + 273.15))   # Total gas concentration at 35 °C [mol/m3]
M_CO2 = 44.01                               # Molar mass of CO2 [g/mol]
rho_flue_gas = 1.3                          # Flue gas density [kg/m3]

catall(f) = vcat([f(ds) for ds in datasets]...)
Fgin = catall(ds -> ds.dist[2, :]) .* rho_flue_gas       # Flue gas flow rate [kg/h]
ygin = catall(ds -> ds.dist[1, :]) ./ cg_tot .* 100      # Flue gas CO2 concentration [vol%]
F = catall(ds -> ds.u[1, :])                             # Solvent flow rate [m3/h]
Td_sp = catall(ds -> ds.u[3, :])                         # Desorber temperature set point [°C]
Q = catall(ds -> ds.Y[4, :])                             # Reboiler duty [kW]
m_cap = catall(ds -> ds.misc[1, :]) .* M_CO2 ./ 1000     # Capture rate [kg CO2/h]

total_duration = sum(ds.t[end] for ds in datasets)                          # [h]
sampling_time = mean(mean(diff(ds.t)) * 60 for ds in datasets)              # [min]

table_2 = DataFrame(
    quantity = ["Number of datasets", "Total duration [h]", "Sampling time [min]",
                "Flue gas flow rate [kg/h]", "Flue gas CO2 conc. [vol%]", "Solvent flow rate [m3/h]",
                "Desorber temperature [°C]", "Reboiler duty [kW]", "Capture rate [kg CO2/h]"],
    min = [length(datasets), total_duration, sampling_time,
           minimum(Fgin), minimum(ygin), minimum(F), minimum(Td_sp), minimum(Q), minimum(m_cap)],
    max = [length(datasets), total_duration, sampling_time,
           maximum(Fgin), maximum(ygin), maximum(F), maximum(Td_sp), maximum(Q), maximum(m_cap)])
CSV.write(joinpath(RESULTS_DIR, "table_2_datasets.csv"), table_2)

println("Table 2: Overview of the experimental datasets")
for row in eachrow(table_2)
    if row.min == row.max
        @printf("  %-30s %8.1f\n", row.quantity, row.min)
    else
        @printf("  %-30s %8.1f - %.1f\n", row.quantity, row.min, row.max)
    end
end

## Table A.5: normalization parameters as defined in the model code (get_normalization_params_high_mu)
norm_params = get_normalization_params_high_mu("DTI")
variables = [(:ca, "c_a"), (:cd, "c_d"), (:cga, "c_g"), (:Tld_HX, "T_d"), (:F, "F"),
             (:Na, "N_a"), (:Nd, "N_d"), (:Q, "Q"), (:Tld_HX_SP, "e_Td")]
table_A5 = DataFrame(variable = [v[2] for v in variables],
                     bias_mu = [norm_params[v[1]].mu for v in variables],
                     scale_s = [norm_params[v[1]].sigma for v in variables])
CSV.write(joinpath(RESULTS_DIR, "table_A5_normalization.csv"), table_A5)

println("\nTable A.5: Normalization parameter values used in the model")
for row in eachrow(table_A5)
    @printf("  %-6s  mu = %6.2f   s = %5.2f\n", row.variable, row.bias_mu, row.scale_s)
end
