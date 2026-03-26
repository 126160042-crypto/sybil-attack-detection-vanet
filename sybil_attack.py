import os
import traci
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# SUMO Configuration File
SUMO_CONFIG = "config.sumocfg"

# Speed threshold to turn vehicles into Sybil vehicles
SPEED_THRESHOLD = 10  # Adjust as needed

# Sybil vehicle properties
SYBIL_COLOR = (255, 0, 0)  # Red
NORMAL_COLOR = (0, 0, 255)  # Blue

# Output file
OUTPUT_FILE = "sybil_attack_data.xlsx"

# Initialize Data Storage
data = []
sybil_vehicles = set()

# Function to run the simulation
def run():
    traci.start(["sumo-gui", "-c", SUMO_CONFIG])  # Start SUMO in GUI mode
    step = 0
    
    while step < 100:  # Run simulation for 1000 seconds
        traci.simulationStep()
        timestamp = traci.simulation.getTime()  # Get current simulation time
        
        vehicles = traci.vehicle.getIDList()
        total_vehicles = len(vehicles)
        
        # Determine the number of Sybil vehicles (50% of total vehicles)
        target_sybil_count = total_vehicles // 2
        
        for veh_id in vehicles:
            speed = traci.vehicle.getSpeed(veh_id)
            acceleration = traci.vehicle.getAcceleration(veh_id)
            x, y = traci.vehicle.getPosition(veh_id)
            edge_id = traci.vehicle.getRoadID(veh_id)  # Current road segment
            veh_type = traci.vehicle.getTypeID(veh_id)  # Vehicle type

            # Convert vehicles to Sybil if speed exceeds threshold and we haven't reached 50% yet
            if speed > SPEED_THRESHOLD and len(sybil_vehicles) < target_sybil_count:
                sybil_vehicles.add(veh_id)
                traci.vehicle.setColor(veh_id, SYBIL_COLOR)
                print(f"🚨 Vehicle {veh_id} converted to Sybil at {speed:.2f} m/s!")

            # Ensure Sybil vehicles stay red
            if veh_id in sybil_vehicles:
                traci.vehicle.setColor(veh_id, SYBIL_COLOR)

            # Determine if the vehicle is a Sybil
            is_sybil = "Yes" if veh_id in sybil_vehicles else "No"

            # Calculate minimum distance to Sybil vehicles
            min_distance_sybil = np.inf
            if sybil_vehicles:
                for sybil_id in sybil_vehicles:
                    if sybil_id != veh_id and sybil_id in vehicles:
                        sx, sy = traci.vehicle.getPosition(sybil_id)
                        distance = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
                        min_distance_sybil = min(min_distance_sybil, distance)

            min_distance_sybil = min_distance_sybil if min_distance_sybil != np.inf else "N/A"

            # Calculate distance to the nearest vehicle
            min_distance_vehicle = np.inf
            closest_vehicle = None
            for other_id in vehicles:
                if other_id != veh_id:
                    ox, oy = traci.vehicle.getPosition(other_id)
                    distance = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
                    if distance < min_distance_vehicle:
                        min_distance_vehicle = distance
                        closest_vehicle = other_id

            min_distance_vehicle = min_distance_vehicle if min_distance_vehicle != np.inf else "N/A"

            # Store data
            data.append([
                timestamp, step, veh_id, veh_type, speed, acceleration, 
                x, y, edge_id, is_sybil, min_distance_sybil, min_distance_vehicle, closest_vehicle
            ])

        step += 1  # Increment time step

    traci.close()
    save_data_to_excel()

# Function to save data to an Excel file
def save_data_to_excel():
    df = pd.DataFrame(data, columns=[
        "Timestamp", "Time (s)", "Vehicle ID", "Vehicle Type", "Speed (m/s)", "Acceleration (m/s²)", 
        "Position X", "Position Y", "Edge ID", "Sybil", "Min Distance to Sybil", 
        "Min Distance to Nearby Vehicle", "Closest Vehicle"
    ])
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Data saved to {OUTPUT_FILE}")
    plot_graphs(df)

# Function to plot the results
def plot_graphs(df):
    plt.figure(figsize=(10, 5))

    # Plot Speed Over Time
    for veh_id in df["Vehicle ID"].unique():
        sub_df = df[df["Vehicle ID"] == veh_id]
        plt.plot(sub_df["Time (s)"], sub_df["Speed (m/s)"], label=veh_id)

    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Vehicle Speed Over Time")
    plt.legend()
    plt.show()

    # Plot Distance to Sybil Over Time
    plt.figure(figsize=(10, 5))
    sybil_df = df[df["Min Distance to Sybil"] != "N/A"]
    for veh_id in sybil_df["Vehicle ID"].unique():
        sub_df = sybil_df[sybil_df["Vehicle ID"] == veh_id]
        plt.plot(sub_df["Time (s)"], sub_df["Min Distance to Sybil"], label=veh_id)

    plt.xlabel("Time (s)")
    plt.ylabel("Distance to Nearest Sybil (m)")
    plt.title("Distance to Sybil Vehicles Over Time")
    plt.legend()
    plt.show()

    # Plot Distance to Closest Vehicle Over Time
    plt.figure(figsize=(10, 5))
    vehicle_df = df[df["Min Distance to Nearby Vehicle"] != "N/A"]
    for veh_id in vehicle_df["Vehicle ID"].unique():
        sub_df = vehicle_df[vehicle_df["Vehicle ID"] == veh_id]
        plt.plot(sub_df["Time (s)"], sub_df["Min Distance to Nearby Vehicle"], label=veh_id)

    plt.xlabel("Time (s)")
    plt.ylabel("Distance to Nearest Vehicle (m)")
    plt.title("Distance to Closest Vehicle Over Time")
    plt.legend()
    plt.show()

# Run the script
if __name__ == "__main__":
    run()
