# Test Results

Raw thermal-run data (`.npz`) and the analysis notebook (`MPC_Results.ipynb`)
from controller test trials, grouped by test scenario. Each `.npz` is one
recorded run (thermal grid history, heater pose, firing percentage) captured
via `rosbag_reader.py` / `thermal_data_collector.py` from the main pipeline.

Folder naming: `<Policy>_<Part/Scenario>_<Mode>_tests`, where policy is
**Greedy** or **MPC**, part/scenario is the panel geometry or motion pattern
(Honeycomb, Black Composite, Saddle Large/Local/Contour), and mode is how the
heat gun was operated (manual firing vs. maintain/contour tracking).

| Folder | Policy | Scenario | # runs |
|---|---|---|---|
| `Greddy_Local_Saddle_Tests` | Greedy | Saddle, local | 5 |
| `Greedy_Saddle_Contour_Manual_tests` | Greedy | Saddle, contour, manual | 6 |
| `Greedy_Saddle_Large_Tests` | Greedy | Saddle, large | 4 |
| `Greedy_Trails_Black_Composite_Large_Heat_Gun_Manitain_Tests` | Greedy | Black composite, large, maintain | 5 |
| `Greedy_Trails_Black_Composite_Local_Heatgun_Manual_Tests` | Greedy | Black composite, local, manual | 3 |
| `Greedy_trails_Black_composite_heatgun_manual_contour_tests` | Greedy | Black composite, contour, manual | 9 |
| `Greedy_trails_Honeycomb_Contour_heatgunmaintainence_tests` | Greedy | Honeycomb, contour, maintain | 4 |
| `Greedy_trails_Honeycomb_Local_test` | Greedy | Honeycomb, local | 2 |
| `Greedy_trails_honeycomb_Large_test` | Greedy | Honeycomb, large | 3 |
| `Greedy_trails_honeycomb_contour_heatgunmanual_tests` | Greedy | Honeycomb, contour, manual | 6 |
| `MPC_Contour_Saddle_Manual_tests` | MPC | Saddle, contour, manual | 6 |
| `MPC_Local_Saddle_Tests` | MPC | Saddle, local | 5 |
| `MPC_Saddle_Large_Tests` | MPC | Saddle, large | 8 |
| `MPC_Trails_Black_Composite_Large_Heat_gun_manitainence_tests` | MPC | Black composite, large, maintain | 6 |
| `MPC_Trails_Black_Composite_Local_Heatgun_Manual_Tests` | MPC | Black composite, local, manual | 6 |
| `MPC_trails_Black_composite_heatgun_manual_contour_tests` | MPC | Black composite, contour, manual | 7 |
| `MPC_trails_Honeycomb_Contour_Heatgunmaintainence_test` | MPC | Honeycomb, contour, maintain | 3 |
| `MPC_trails_Honeycomb_Contour_Heatgunmanualmode_tests` | MPC | Honeycomb, contour, manual | 6 |
| `MPC_trails_Honeycomb_Large_test` | MPC | Honeycomb, large | 1 |
| `MPC_trails_Honeycomb_Local_Test` | MPC | Honeycomb, local | 1 |

**`MPC_Results.ipynb`** loads these `.npz` files and produces the ramp-up /
steady-state plots referenced in the [paper](../paper/Adaptive_Robotic_Surface_Heating.pdf)
and in the main [README](../README.md#6-controllers-compared-greedy-vs-mpc).

To inspect a single run without the notebook:

```python
import numpy as np
d = np.load("Greedy_Saddle_Large_Tests/05_27_26_saddle_large_test_Greedy_10.npz")
print(d.files)          # array names stored in this run
```
