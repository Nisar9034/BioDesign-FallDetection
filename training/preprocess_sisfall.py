#!/usr/bin/env python3
"""
SisFall Dataset Preprocessing Script
Converts raw SisFall text logs into calibrated, downsampled CSV files
suitable for TinyML training and Edge Impulse ingestion.
"""

import os
import glob
import pandas as pd
import numpy as np

# --- Configuration ---
DATASET_DIR = "./SisFall_dataset"
OUTPUT_DIR = "./processed_csv"
TARGET_HZ = 50                       # Target rate to match ESP32 MPU6050
ORIGINAL_HZ = 200                    # SisFall acquisition frequency
DOWNSAMPLE_RATIO = ORIGINAL_HZ // TARGET_HZ

# Sensor conversion constants from SisFall documentation:
# ADXL345: range +-16g, 13-bit resolution -> units in g
ACC_SCALE = 32.0 / 8192.0
# ITG3200: range +-2000 deg/s, 16-bit resolution -> units in deg/s
GYRO_SCALE = 4000.0 / 65536.0

def process_sisfall():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_pattern = os.path.join(DATASET_DIR, "**", "*.txt")
    files = glob.glob(file_pattern, recursive=True)

    if not files:
        print(f"No files found in {DATASET_DIR}. Please ensure the dataset is extracted here.")
        return

    print(f"Found {len(files)} files to convert.")
    successful = 0

    for filepath in files:
        filename = os.path.basename(filepath)
        
        # Determine classification label
        if filename.startswith('F'):
            label = "fall"
        elif filename.startswith('D'):
            label = "adl"
        else:
            continue

        try:
            with open(filepath, 'r') as f:
                raw_lines = [line.strip().rstrip(';,') for line in f if line.strip()]

            parsed_rows = [list(map(float, line.split(','))) for line in raw_lines]
            raw_matrix = np.array(parsed_rows)

            # Columns 0,1,2 = ADXL345 Accel; Columns 3,4,5 = ITG3200 Gyro
            acc_x = raw_matrix[:, 0] * ACC_SCALE
            acc_y = raw_matrix[:, 1] * ACC_SCALE
            acc_z = raw_matrix[:, 2] * ACC_SCALE
            gyr_x = raw_matrix[:, 3] * GYRO_SCALE
            gyr_y = raw_matrix[:, 4] * GYRO_SCALE
            gyr_z = raw_matrix[:, 5] * GYRO_SCALE

            df = pd.DataFrame({
                'accX': acc_x,
                'accY': acc_y,
                'accZ': acc_z,
                'gyrX': gyr_x,
                'gyrY': gyr_y,
                'gyrZ': gyr_z
            })

            # Downsample to match target ESP32 sampling rate
            df_downsampled = df.iloc[::DOWNSAMPLE_RATIO].reset_index(drop=True)

            # Add timestamp in milliseconds
            dt_ms = 1000.0 / TARGET_HZ
            df_downsampled.insert(0, 'timestamp', [int(i * dt_ms) for i in range(len(df_downsampled))])

            # Save processed CSV
            out_filename = f"{label}.{filename.replace('.txt', '.csv')}"
            df_downsampled.to_csv(os.path.join(OUTPUT_DIR, out_filename), index=False)
            successful += 1

        except Exception as err:
            print(f"Skipping {filename} due to error: {err}")

    print(f"Complete: {successful} files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_sisfall()
