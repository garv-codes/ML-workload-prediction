# CloudSim Provisioning Simulation

`ProvisioningSimulation.java` replays the hourly forecasts written by `mttd_bridge.py` in a CloudSim data centre and measures what they mean for VM provisioning.

## What it simulates

For every hour of the predictions CSV:

1. **Provision** `V = min(150, max(1, ceil(1.10 × forecast / C)))` VMs from the MTTD forecast (10 % headroom, cap of 150 VMs), with `C = 4,000` requests per VM per hour.
2. **Execute** the hour's real requests (`actual`) as cloudlets of 200 requests each (180,000 MI, 180 s on one 1,000-MIPS VM), assigned round-robin to time-shared VMs.
3. **Check the SLA:** the hour is an SLA violation if the last cloudlet finishes after 3,600 s.

`C` is set by `HITS_PER_VM_PER_HOUR = 4000` (900 MI per request). Setting it to `0` instead switches to automatic calibration (`C = mean(actual) / TARGET_AVG_VMS`); the reported results use 4,000.

| Parameter | Value |
|---|---|
| Data centre | 25 hosts × 8 PEs × 1,000 MIPS, 16 GB RAM, time-shared VM scheduler |
| VM | 1 PE, 1,000 MIPS, 512 MB RAM, time-shared cloudlet scheduler |
| Capacity `C` | 4,000 requests per VM per hour (900 MI per request) |
| Headroom / cap | 1.10 / 150 VMs |
| Time step / SLA | 3,600 s (`INTERVAL_SECONDS`) |
| Horizon | all rows of the CSV (`MAX_HOURS = 0`); a positive value is for quick tests only |

Reported metrics: VM-hours provisioned, VM-hours needed for the exact load, wasted VM-hours (over-provisioning), under-provisioned hours, SLA-violating hours, average completion time.

## Requirements

- JDK (the reported runs used JDK 26)
- **CloudSim 3.0.3** — jar from the [CloudSim 3.0.3 release](https://github.com/Cloudslab/cloudsim/releases/tag/cloudsim-3.0.3) (`cloudsim-3.0.3/jars/cloudsim-3.0.3.jar`)

## Build and run

From this folder (on Windows use `;` instead of `:` in the classpath):

```bash
javac -cp <path-to-cloudsim.jar> -d build ProvisioningSimulation.java

# arguments: <predictions CSV> <output CSV>   (defaults: predictions_retrain.csv cloudsim_results.csv)
java -cp build:<path-to-cloudsim.jar> mttd.ProvisioningSimulation ../results/cricket/predictions_all5.csv   cloudsim_all5.csv
java -cp build:<path-to-cloudsim.jar> mttd.ProvisioningSimulation ../results/cricket/predictions_daily.csv  cloudsim_daily.csv
java -cp build:<path-to-cloudsim.jar> mttd.ProvisioningSimulation ../results/cricket/predictions_hourly.csv cloudsim_hourly.csv
```

## Results (1,080 hours, VM-hours needed for the exact load = 6,981)

| Predictions | VM-hours | Wasted VMh | Under-prov. hours | SLA-violating hours | Avg completion (s) |
|---|---|---|---|---|---|
| Standard MTTD (`predictions_all5.csv`) | 7,475 | 1,077 | 145 | 145 (13.4 %) | 3,036 |
| Daily reselection (`predictions_daily.csv`) | 6,802 | 460 | 263 | 263 (24.4 %) | 3,501 |
| Hourly reselection (`predictions_hourly.csv`) | 7,312 | 702 | 158 | 158 (14.6 %) | 3,324 |

Per-hour outputs (`cloudsim_all5.csv`, `cloudsim_daily.csv`, `cloudsim_hourly.csv`) and console screenshots are in [`../results/cricket/cloudsim/`](../results/cricket/cloudsim/). Running the commands above with CloudSim 3.0.3 reproduces these files exactly.

