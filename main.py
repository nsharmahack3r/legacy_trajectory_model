"""
Elephant Movement Trajectory Forecasting
Models: CNN, RCNN (CNN + RNN), LSTM
Predicts next (longitude, latitude) given a sliding window of past steps.

Usage:
    python complete_code.py --data /path/to/data.csv

For the full 0.78M-row dataset, use --sample_frac 1.0 (default 1.0).
The script handles chunked loading if memory is a concern.
"""

from src.main import main

if __name__ == "__main__":
    main()
