"""
Battery Storage Planner for Solar Forecasting
---------------------------------------------
Reads hourly GHI prediction (W/m²), estimates energy (kWh),
and gives battery usage guidance.

Assumptions:
- 1 m² panel
- 18% panel efficiencys
- 5 kWh battery
"""

import pandas as pd

def estimate_energy(ghi_series, panel_efficiency=0.18):
    energy_kwh = ghi_series * 1 * panel_efficiency / 1000
    return energy_kwh.sum()

def battery_decision(total_energy_kwh, battery_capacity_kwh=5):
    if total_energy_kwh >= battery_capacity_kwh:
        return (" Battery Fully Charged  Night irrigation possible", "Full")
    elif total_energy_kwh >= 2:
        return (" Partial Charge  Limit usage to 1 to 2 hours", "Partial")
    else:
        return (" Low Solar  Reschedule irrigation or use grid", "Low")

def run_battery_planner(predicted_ghi):
    print("\n[Battery Storage Planner]")
    total_energy = estimate_energy(predicted_ghi)
    decision, status = battery_decision(total_energy)
    print(f"Predicted Energy (kWh): {total_energy:.2f}")
    print(f"Battery Status: {decision}")
    return status



# Test block (optional)
if __name__ == "__main__":
    ghi_test = pd.Series([600]*5 + [300]*5 + [100]*14)  # Sample 24-hour data
    run_battery_planner(ghi_test)

