import socket
import json
import time
import csv
import os

def start_patient_stream(host="localhost", port=9999, csv_filename="data/diabetes_raw.csv"):
    # 1. Verify the data file exists before starting the server
    if not os.path.exists(csv_filename):
        print(f"CRITICAL ERROR: Data file '{csv_filename}' not found.")
        print("Please ensure the CSV is in the same directory as this script.")
        return

    # 2. Configure the TCP Socket Server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # SO_REUSEADDR prevents "Address already in use" errors if you restart the script quickly
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Live Clinical Patient Stream Engine operational on interface {host}:{port}...")
    print(f"Streaming real patient records from '{csv_filename}'...")

    while True:
        try:
            # Wait for the PySpark streaming application to connect
            conn, addr = server_socket.accept()
            print(f"Connected to PySpark Streaming Client: {addr}")
            
            # 3. Open the CSV and read row-by-row
            with open(csv_filename, mode='r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                
                for row in csv_reader:
                    try:
                        # 4. Map the CSV columns to our streaming JSON schema
                        # We cast numeric columns to integers to match our PySpark StructType schema
                        patient_data = {
                            "race": row.get("race", "Missing"),
                            "gender": row.get("gender", "Missing"),
                            "time_in_hospital": int(row.get("time_in_hospital", 1)),
                            "num_medications": int(row.get("num_medications", 1)),
                            "readmit_30_days": int(row.get("readmit_30_days", 0)),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # 5. Serialize to JSON and transmit over the socket
                        conn.send((json.dumps(patient_data) + "\n").encode("utf-8"))
                        print("Dispatched Real Patient Record:", patient_data)
                        
                        # Simulates real-time clinical traffic (1 record every 0.5 seconds)
                        time.sleep(0.5) 
                        
                    except (BrokenPipeError, ConnectionResetError):
                        print("Client disconnected. Halting stream and waiting for reconnection...")
                        break  # Breaks the CSV loop, goes back to waiting for a new connection
                    except ValueError as ve:
                        # Catches any rows where numeric conversion fails (e.g., missing data)
                        print(f"Skipping malformed row due to error: {ve}")
                        continue
                        
        except KeyboardInterrupt:
            print("\nShutting down stream engine manually...")
            break
        except Exception as e:
            print(f"Network stream failure: {e}")
            time.sleep(2) # Pause briefly before attempting to recover

if __name__ == "__main__":
    start_patient_stream()