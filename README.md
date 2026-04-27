# 5G Core Failure Distribution Analysis

This repository contains a Python-based utility for preprocessing and visualizing class distributions in **5G Core Network Digital Twin** datasets. 

## Overview
In 5G Core (5GC) environments, network failure events (e.g., AMF overload, UDM timeouts) are often significantly imbalanced compared to normal operational data. This tool helps researchers visualize that imbalance before training Graph Neural Networks (GNNs).

## Features
* **GUI File Selection:** Uses Tkinter for easy CSV loading.
* **Pre-processing:** Automated feature scaling (MinMaxScaler) and label encoding.
* **Imbalance Visualization:** Stripplot-based density visualization to assess the sparsity of failure classes.

## Usage
1. Clone the repo: `git clone https://github.com/YourUsername/your-repo-name.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the script: `python main.py`
4. Select your 5G telemetry CSV when prompted.

## Research Context
This tool is part of a broader framework for **Knowledge Graph Failure Localization** in autonomous 5G networks.

CF-GNN/
├── data/               # Sample telemetry CSVs (or download scripts)
├── models/
│   ├── __init__.py
│   └── cf_gnn.py       # The refined code above
├── notebooks/
│   └── analysis.ipynb  # Visualization of class distribution/GFT spectral plots
├── utils/
│   └── preprocessing.py
├── requirements.txt    # Essential for reproducibility
├── README.md           # The "face" of the project
