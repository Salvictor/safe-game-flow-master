# Crazyflie 2.1 Brushless simulation parameter provenance

The closed-loop simulation intentionally separates hardware specifications from
parameters that still require system identification.

| Parameter | Value used | Status | Source/action before hardware claims |
|---|---:|---|---|
| Take-off mass with legs | 0.034 kg | Official specification | Bitcraze product page/datasheet |
| Motor-centre diagonal | 0.100 m | Official specification | Gives 0.050 m centre-to-motor radius |
| Maximum thrust per motor | 30 gf (0.2942 N) | Official specification | Bitcraze product page and motor characterization post |
| Motor | 08028, 10000 KV | Official specification | Bitcraze datasheet |
| Inertia | diag(1.6, 1.6, 2.9)e-5 kg m² | Simulation assumption | Identify using CAD/bifilar-pendulum experiment |
| Motor/thrust time constant | 0.025 s | Simulation assumption | Identify from logged step response |
| Linear drag | 0.015 N/(m/s) | Simulation assumption | Identify from flight data |
| Yaw moment/thrust ratio | 0.006 m | Simulation assumption | Identify on thrust stand or fit from yaw response |

Official references:

- https://www.bitcraze.io/products/crazyflie-2-1-brushless/
- https://www.bitcraze.io/documentation/hardware/crazyflie_2_1_brushless/crazyflie_2_1_brushless-datasheet.pdf
- https://www.bitcraze.io/2024/08/the-optimized-crazyflie-2-1-brushless-motors/

Until the assumed parameters are identified, the 6-DoF results validate the
algorithm under a physically structured plant and controlled uncertainty; they
do not establish high-fidelity reproduction of one particular physical unit.
