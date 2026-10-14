# ------------------------------------------------------------------------------------------
# Run all reproduction drivers in order. Each driver runs in its own Julia process.
#
#     julia --project=. run_all.jl
#
# Total runtime: about 10 minutes. The parameter estimation (driver 02) is not included, since
# it took about 50 CPU hours on an HPC cluster; the drivers use the published estimates in
# data/parameter_estimates/. See drivers/02_estimate_parameters.jl for how to run it.
# ------------------------------------------------------------------------------------------

drivers = [
    "01_process_raw_data.jl",       # Raw pilot-plant logs -> data series
    "03_tables.jl",                 # Table 2 and Table A.5
    "04_compute_predictions.jl",    # EKF predictions and RMSEs (needed by 05 and 06)
    "05_fig_time_series.jl",        # Figure 4
    "06_fig_rmse.jl",               # Figures 5 and 6
    "07_fig_nn_output_sweeps.jl",   # Figure 7
    "08_fig_steady_state.jl",       # Figures 8 and 9
    "09_empc_closed_loop.jl",       # Closed-loop EMPC replayed on the logged data of tests #3 and #4
    "10_fig_empc_results.jl",       # Figures 12, 13 and 14, EMPC results
]

julia = Base.julia_cmd()
for driver in drivers
    println("\n", "="^90, "\nRunning drivers/$(driver)\n", "="^90); flush(stdout)
    run(`$julia --project=$(@__DIR__) --threads=auto $(joinpath(@__DIR__, "drivers", driver))`)
end
println("\nAll drivers finished. Figures are in figures/ and results in results/.")
