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
    # Sample data
    pd.set_option('display.max_columns', None)
    data = {
        'Time_HHMM': [1230, 900, 1530, 2200],
        'Weekday': [1, 5, 7, 3],
        'Month': [1, 7, 12, 4]
    }

    df = pd.DataFrame(data)

    # Applying the TimeEncoder methods
    df['Time_in_minutes'] = df['Time_HHMM'].apply(TimeEncoder.convert_time_to_minutes)
    df['Time_sin'], df['Time_cos'] = zip(*df['Time_in_minutes'].apply(TimeEncoder.encode_cyclical_time))
    df['Weekday_sin'], df['Weekday_cos'] = zip(*df['Weekday'].apply(TimeEncoder.encode_cyclical_weekday))
    df['Month_sin'], df['Month_cos'] = zip(*df['Month'].apply(TimeEncoder.encode_cyclical_month))
    print(df)
