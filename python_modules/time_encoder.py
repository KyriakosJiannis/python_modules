import numpy as np
import pandas as pd


class TimeEncoder:
    def __init__(self):
        pass

    @staticmethod
    def convert_time_to_minutes(time_hhmm):
        """
        Converts time in HHMM format to minutes.
        """
        if pd.isna(time_hhmm):
            return np.nan
        hour = int(time_hhmm // 100)
        minute = int(time_hhmm % 100)
        return hour * 60 + minute

    @staticmethod
    def encode_cyclical_time(time_minutes):
        """
        Encodes time in minutes to cyclical sin and cos components.
        """
        time_sin = np.sin(2 * np.pi * time_minutes / 1440)
        time_cos = np.cos(2 * np.pi * time_minutes / 1440)
        return time_sin, time_cos

    @staticmethod
    def encode_cyclical_weekday(weekdays):
        """
        Encodes weekdays to cyclical sin and cos components.
        """
        normalized_weekday = weekdays - 1  # Normalize to 0 - 6
        weekday_sin = np.sin(2 * np.pi * normalized_weekday / 7)
        weekday_cos = np.cos(2 * np.pi * normalized_weekday / 7)
        return weekday_sin, weekday_cos

    @staticmethod
    def encode_cyclical_month(months):
        """
        Encodes months to cyclical sin and cos components.
        """
        normalized_month = months - 1  # Normalize to 0 - 11
        month_sin = np.sin(2 * np.pi * normalized_month / 12)
        month_cos = np.cos(2 * np.pi * normalized_month / 12)
        return month_sin, month_cos


if __name__ == "__main__":
    # Example usage of the TimeEncoder class
    encoder = TimeEncoder()
    minutes = encoder.convert_time_to_minutes(1305)  # Example time in HHMM format
    time_sin, time_cos = encoder.encode_cyclical_time(minutes)
    weekday_sin, weekday_cos = encoder.encode_cyclical_weekday(5)  # Example weekday (e.g., Friday)
    month_sin, month_cos = encoder.encode_cyclical_month(12)  # Example month (e.g., December)

    # Print the results for demonstration
    print("Time converted to minutes:", minutes)
    print("Cyclical time encoding (sin, cos):", time_sin, time_cos)
    print("Cyclical weekday encoding (sin, cos):", weekday_sin, weekday_cos)
    print("Cyclical month encoding (sin, cos):", month_sin, month_cos)
