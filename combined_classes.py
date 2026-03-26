import os
import traci
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class NormalTrafficSimulation:
    SUMO_CONFIG = "config.sumocfg"
    OUTPUT_FILE = "combined_traffic_data.xlsx"  # Single combined file

    def __init__(self):
        self.data = []

    def run(self):
        traci.start(["sumo-gui", "-c", self.SUMO_CONFIG])
        step = 0
        
        while step < 100:
            traci.simulationStep()
            self.log_vehicle_data(step)
            step += 1
        
        traci.close()

    def log_vehicle_data(self, step):
        timestamp = traci.simulation.getTime()
        vehicles = traci.vehicle.getIDList()

        for veh_id in vehicles:
            speed = 50.0  # Constant speed
            acceleration = 0.0  # No acceleration
            x, y = traci.vehicle.getPosition(veh_id)  # Constant position
            edge_id = traci.vehicle.getRoadID(veh_id)
            veh_type = traci.vehicle.getTypeID(veh_id)

            self.data.append([
                timestamp, step, veh_id, veh_type, speed, acceleration,
                x, y, edge_id
            ])

class SybilAttackSimulation:
    SUMO_CONFIG = "config.sumocfg"
    SPEED_VARIATIONS = [7, 33, 77, 199]  # Different speeds for Sybil vehicles
    SYBIL_COLOR = (255, 0, 0)  # Red
    NORMAL_COLOR = (0, 0, 255)  # Blue
    OUTPUT_FILE = "combined_traffic_data.xlsx"  # Same file as NormalTrafficSimulation

    def __init__(self):
        self.data = []
        self.sybil_vehicles = set()
        self.vehicle_speeds = {}

    def run(self):
        traci.start(["sumo-gui", "-c", self.SUMO_CONFIG])
        step = 0
        
        while step < 100:
            traci.simulationStep()
            self.log_vehicle_data(step)
            step += 1
        
        traci.close()

    def log_vehicle_data(self, step):
        timestamp = traci.simulation.getTime()
        vehicles = traci.vehicle.getIDList()
        total_vehicles = len(vehicles)
        target_sybil_count = total_vehicles // 2  # Convert 50% to Sybil

        for veh_id in vehicles:
            # Assign a new random speed if not assigned before
            if veh_id not in self.vehicle_speeds:
                self.vehicle_speeds[veh_id] = np.random.choice(self.SPEED_VARIATIONS)

            # Randomly vary the speed slightly
            speed = self.vehicle_speeds[veh_id] + np.random.uniform(-5, 5)
            acceleration = np.random.uniform(-1, 1)
            speed = max(0, speed)  # Ensure speed is non-negative

            # Get vehicle position and force bigger random jumps for Sybil vehicles
            x, y = traci.vehicle.getPosition(veh_id)

            if veh_id in self.sybil_vehicles:
                x += np.random.uniform(-100, 100)  # Larger X variation (15-100 units)
                y += np.random.uniform(-100, 100)  # Larger Y variation (15-100 units)
            else:
                x += speed * 0.1 + np.random.uniform(-2, 2)  # Small normal vehicle movement
                y += speed * 0.1 + np.random.uniform(-2, 2)

            edge_id = traci.vehicle.getRoadID(veh_id)
            veh_type = traci.vehicle.getTypeID(veh_id)

            # Convert vehicles to Sybil if speed exceeds a threshold
            if speed > 10 and len(self.sybil_vehicles) < target_sybil_count:
                self.sybil_vehicles.add(veh_id)
                traci.vehicle.setColor(veh_id, self.SYBIL_COLOR)
                print(f" Vehicle {veh_id} converted to Sybil at {speed:.2f} m/s!")

            if veh_id in self.sybil_vehicles:
                traci.vehicle.setColor(veh_id, self.SYBIL_COLOR)

            is_sybil = "Yes" if veh_id in self.sybil_vehicles else "No"
            min_distance_sybil = self.calculate_min_distance(veh_id, vehicles, sybil_only=True)
            min_distance_vehicle = self.calculate_min_distance(veh_id, vehicles)

            self.data.append([
                timestamp, step, veh_id, veh_type, speed, acceleration,
                x, y, edge_id, is_sybil, min_distance_sybil, min_distance_vehicle
            ])

    def calculate_min_distance(self, veh_id, vehicles, sybil_only=False):
        x, y = traci.vehicle.getPosition(veh_id)
        min_distance = np.inf
        
        for other_id in vehicles:
            if other_id == veh_id:
                continue
            if sybil_only and other_id not in self.sybil_vehicles:
                continue
            ox, oy = traci.vehicle.getPosition(other_id)
            distance = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != np.inf else "N/A"

def save_combined_excel(normal_data, sybil_data):
    df_normal = pd.DataFrame(normal_data, columns=[
        "Timestamp", "Time (s)", "Vehicle ID", "Vehicle Type", "Speed (m/s)", "Acceleration (m/s²)",
        "Position X", "Position Y", "Edge ID"
    ])
    
    df_sybil = pd.DataFrame(sybil_data, columns=[
        "Timestamp", "Time (s)", "Vehicle ID", "Vehicle Type", "Speed (m/s)", "Acceleration (m/s²)",
        "Position X", "Position Y", "Edge ID", "Sybil", "Min Distance to Sybil", "Min Distance to Vehicle"
    ])

    # Add a "Sybil" column to normal data (default to "No")
    df_normal["Sybil"] = "No"
    df_normal["Min Distance to Sybil"] = "N/A"
    df_normal["Min Distance to Vehicle"] = "N/A"

    # Ensure both DataFrames have the same columns
    combined_df = pd.concat([df_normal, df_sybil], ignore_index=True)

    # Save to Excel (single sheet)
    combined_df.to_excel("combined_traffic_data.xlsx", sheet_name="Traffic Data", index=False)

    print(f" Data saved in a single sheet 'Traffic Data' in combined_traffic_data.xlsx!")

if __name__ == "__main__":
    # Run Normal Traffic Simulation
    normal_simulation = NormalTrafficSimulation()
    normal_simulation.run()

    # Run Sybil Attack Simulation
    sybil_simulation = SybilAttackSimulation()
    sybil_simulation.run()

    # Combine and save the data in one sheet
    save_combined_excel(normal_simulation.data, sybil_simulation.data)
