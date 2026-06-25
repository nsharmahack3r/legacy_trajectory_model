from dataclasses import dataclass


FEATURE_COLS = [
    "longitude",
    "latitude",
    "speed_kmh",
    "step_dist_m",
    "turning_angle",
    "bearing",
    "dir_persistence",
    "acceleration",
    "hour",
    "season_code",
    "time_of_day_code",
    "NDVI",
    "EVI",
    "LST_celsius",
    "elevation_m",
    "slope_deg",
    "water_occ_1km",
    "NDWI",
]
TARGET_COLS = ["longitude", "latitude"]


@dataclass
class DataPaths:
    actual_data: str = r"F:\dev\organisations\penn_state_univ\data\actual_data.csv"
    sample_data: str = r"F:\dev\organisations\penn_state_univ\data\sample_data.csv"
