import os
import traci
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class SybilAttackSimulation:
    SUMO_CONFIG = "config.sumocfg"
    SPEED_THRESHOLD = 10  # Speed threshold to turn vehicles into Sybil vehicles
    SYBIL_COLOR = (255, 0, 0)  # Red
    NORMAL_COLOR = (0, 0, 255)  # Blue
    OUTPUT_FILE = "sybil_attack_data.xlsx"

    def __init__(self):
        self.data = []
        self.sybil_vehicles = set()

    def run(self):
        traci.start(["sumo-gui", "-c", self.SUMO_CONFIG])
        step = 0
        
        while step < 100:  # Run simulation for 1000 seconds
            traci.simulationStep()
            self.log_vehicle_data(step)
            step += 1
        
        traci.close()
        self.save_data_to_excel()

    def log_vehicle_data(self, step):
        timestamp = traci.simulation.getTime()
        vehicles = traci.vehicle.getIDList()
        total_vehicles = len(vehicles)
        target_sybil_count = total_vehicles // 2  # Convert 50% to Sybil

        for veh_id in vehicles:
            speed = traci.vehicle.getSpeed(veh_id) + np.random.uniform(-2, 2)  # Varying speed
            acceleration = traci.vehicle.getAcceleration(veh_id)
            x, y = traci.vehicle.getPosition(veh_id)
            x += np.random.uniform(-1, 1)  # Varying position
            y += np.random.uniform(-1, 1)
            edge_id = traci.vehicle.getRoadID(veh_id)
            veh_type = traci.vehicle.getTypeID(veh_id)

            # Convert vehicles to Sybil if speed exceeds threshold
            if speed > self.SPEED_THRESHOLD and len(self.sybil_vehicles) < target_sybil_count:
                self.sybil_vehicles.add(veh_id)
                traci.vehicle.setColor(veh_id, self.SYBIL_COLOR)
                print(f"🚨 Vehicle {veh_id} converted to Sybil at {speed:.2f} m/s!")
            
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

    def save_data_to_excel(self):
        df = pd.DataFrame(self.data, columns=[
            "Timestamp", "Time (s)", "Vehicle ID", "Vehicle Type", "Speed (m/s)", "Acceleration (m/s²)",
            "Position X", "Position Y", "Edge ID", "Sybil", "Min Distance to Sybil", "Min Distance to Vehicle"
        ])
        df.to_excel(self.OUTPUT_FILE, index=False)
        print(f"Data saved to {self.OUTPUT_FILE}")
        self.plot_graphs(df)

    def plot_graphs(self, df):
        plt.figure(figsize=(10, 5))
        for veh_id in df["Vehicle ID"].unique():
            sub_df = df[df["Vehicle ID"] == veh_id]
            plt.plot(sub_df["Time (s)"], sub_df["Speed (m/s)"], label=veh_id)
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (m/s)")
        plt.title("Vehicle Speed Over Time")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    simulation = SybilAttackSimulation()
    simulation.run()
