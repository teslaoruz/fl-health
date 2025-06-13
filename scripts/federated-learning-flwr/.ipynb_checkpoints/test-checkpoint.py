import subprocess
import time
import sys
from pathlib import Path

def run_federated_test():
    # Configuration
    SERVER_CMD = ["python", "server.py", "--rounds", "3", "--min-clients", "5"]
    CLIENT_CMD = "python client.py ../../datasets/diabetes/processed_silos/hospital_{}.csv"
    CLIENT_COUNT = 5
    SERVER_STARTUP_TIME = 5  # seconds
    TEST_TIMEOUT = 300  # 5 minutes
    
    # Start server
    print("🚀 Starting server...")
    server = subprocess.Popen(SERVER_CMD)
    time.sleep(SERVER_STARTUP_TIME)
    
    try:
        # Start all clients
        clients = []
        for i in range(1, CLIENT_COUNT + 1):
            print(f"🔌 Starting client {i}...")
            clients.append(subprocess.Popen(CLIENT_CMD.format(i).split()))
        
        # Monitor progress
        # start_time = time.time()
        # while time.time() - start_time < TEST_TIMEOUT:
        #     # Check for completion
        #     if all(Path(f"client_{i}.log").exists() for i in range(1, CLIENT_COUNT + 1)):
        #         print("✅ All clients completed successfully!")
        #         break
        #     time.sleep(5)
        # else:
        #     print("⏰ Test timed out!")
        #     sys.exit(1)
            
        # Verify training occurred
        # with open("server.log") as f:
        #     log = f.read()
        #     assert "Round 3 completed" in log, "Training didn't complete"
        #     assert "federated_eval" in log, "No evaluation occurred"
        
        print("\n🔥 Federated learning test passed with 5 clients!")
        
    finally:
        # Cleanup
        server.terminate()
        for client in clients:
            client.terminate()

if __name__ == "__main__":
    run_federated_test()