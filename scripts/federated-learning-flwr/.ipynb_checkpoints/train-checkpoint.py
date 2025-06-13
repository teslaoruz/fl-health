import subprocess
import time


hospital_files = [
    "../../datasets/diabetes/processed_silos/hospital_1.csv",
    "../../datasets/diabetes/processed_silos/hospital_2.csv",
    "../../datasets/diabetes/processed_silos/hospital_3.csv",
    "../../datasets/diabetes/processed_silos/hospital_4.csv",
    "../../datasets/diabetes/processed_silos/hospital_5.csv"
]

clients = []
for file in hospital_files:
    print(f"Starting client for {file}...")
    clients.append(subprocess.Popen(["python", "client.py", file]))
    time.sleep(1)

for client in clients:
    client.wait()
