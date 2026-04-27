# CF-GNN: Classical Fourier Graph Neural Network for Imbalance Failure

This repository contains a Python-based utility for preprocessing and visualizing class distributions in **5G Core Network Digital Twin** datasets. 

## Overview
In 5G Core (5GC) environments, network failure events (e.g., AMF overload, UDM timeouts) are often significantly imbalanced compared to normal operational data. This tool helps researchers visualize that imbalance before training Graph Neural Networks (GNNs).

## Features
* **GUI File Selection:** Uses Tkinter for easy CSV loading.
* **Pre-processing:** Automated feature scaling (MinMaxScaler) and label encoding.
* **Imbalance Visualization:** Stripplot-based density visualization to assess the sparsity of failure classes.

## Usage
1. Clone the repo: `git clone https://github.com/YourUsername/your-repo-name.git`
2. Install dependencies: `pip install -r requirements.txt.`
3. Run the script: `python main.py.`
4. Select your 5G telemetry CSV when prompted.
