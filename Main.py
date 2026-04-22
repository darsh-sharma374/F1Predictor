import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

# Choose the Race you want to predict:
specific_year = 2026 #Change any of these variables
specific_gp = 'Japan'


# Gathering Data and Processing it ---------------------------------------------------------------------------------
fastf1.Cache.enable_cache('cache_directory')
session = fastf1.get_session(specific_year, specific_gp, 'FP3')
session.load(telemetry=False)

all_laps = session.laps.pick_quicklaps().pick_accurate()

top_ten_driver_lap_timings = []
for driver in session.drivers:
    driver_laps = all_laps.pick_drivers(driver)
    fastest_10 = driver_laps.nsmallest(10, 'LapTime')
    top_ten_driver_lap_timings.append(fastest_10)

final_df = pd.concat(top_ten_driver_lap_timings).reset_index(drop=True)
final_df['LapTimeSeconds'] = final_df['LapTime'].dt.total_seconds()
stats = final_df.groupby('Driver')['LapTimeSeconds'].agg(['mean', 'std']).reset_index()
stats = stats.sort_values(by='mean').reset_index(drop=True)


# Stochastic Modelling ---------------------------------------------------------------------------------

points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SEC_PER_POS = 0.20
P_DNF = 0.05
DNF_PENALTY = -10

def get_expected_value(row, rank):
    mu_pos = rank
    sigma_pos = (row['std'] if pd.notna(row['std']) else 1.0) / SEC_PER_POS
    ev = 0
    for pos in range(1, 21):
        prob = norm.cdf(pos + 0.5, mu_pos, sigma_pos) - norm.cdf(pos - 0.5, mu_pos, sigma_pos)
        ev += prob * (1 - P_DNF) * points_system.get(pos, 0)
    ev += (P_DNF * DNF_PENALTY)
    return ev

stats['Expected_Value'] = [get_expected_value(row, i+1) for i, row in stats.iterrows()]

# Knapsack Problem (ILP with real costs) ---------------------------------------------------------------------------------

driver_costs = {
    'ANT': 24.1, 'RUS': 28.3, 'LEC': 23.7, 'HAM': 23.2, 'VER': 28.2,
    'LAW': 7.5,  'GAS': 13.0, 'OCO': 9.1,  'SAI': 12.4, 'BEA': 9.2,
    'NOR': 26.5, 'PER': 7.0,  'COL': 7.6,  'LIN': 7.6,  'PIA': 24.6,
    'HAD': 13.3, 'BOR': 5.8,  'ALB': 10.2, 'HUL': 5.0,  'BOT': 4.1,
    'ALO': 8.2,  'STR': 6.2
}

model = LpProblem(name="F1-Fantasy-Optimizer", sense=LpMaximize)
drivers = stats['Driver'].tolist()
x = {d: LpVariable(name=f"x_{d}", cat="Binary") for d in drivers}

expected_values = dict(zip(stats['Driver'], stats['Expected_Value']))
model += lpSum(expected_values[d] * x[d] for d in drivers)

model += (lpSum(driver_costs[d] * x[d] for d in drivers) <= 100.0, "Budget")
model += (lpSum(x[d] for d in drivers) == 5, "Roster_Size")

model.solve()

# Results and Graphing ---------------------------------------------------------------------------------

selected_drivers = [d for d in drivers if x[d].value() == 1]
total_cost = sum(driver_costs[d] for d in selected_drivers)
total_points = sum(expected_values[d] for d in selected_drivers)

print("OPTIMAL 2026 JAPAN ROSTER :)")
for d in selected_drivers:
    print(f"{d}: ${driver_costs[d]}M ------- E[X]: {expected_values[d]:.2f}")

print("----------------------")
print(f"Total Cost: ${total_cost:.1f}M")
print(f"Total Exp Points: {total_points:.2f}")

# Plotting
plt.figure(figsize=(10, 6))
for d in drivers:
    color = 'red' if d in selected_drivers else 'blue'
    plt.scatter(driver_costs[d], expected_values[d], color=color)
    plt.text(driver_costs[d]+0.3, expected_values[d], d, fontsize=9)

plt.axhline(0, color='black', linewidth=0.5)
plt.title("F1 IA: Cost vs. Expected Points")
plt.xlabel("Cost ($M)")
plt.ylabel("Expected Points E[V]")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()