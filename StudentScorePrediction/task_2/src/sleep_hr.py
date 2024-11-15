import pandas as pd

from datetime import datetime


# Function to calculate sleep hours based on sleep_time and wake_time
def cal_sleep_hours(sleep_time, wake_time):
    # Convert sleep_time and wake_time into datetime objects
    sleep_time = datetime.strptime(sleep_time, "%H:%M")
    wake_time = datetime.strptime(wake_time, "%H:%M")

    # If wake_time is earlier than sleep_time, assume wake_time is on the next day
    if wake_time < sleep_time:
        wake_time += pd.Timedelta(days=1)

    # Calculate different in hours
    sleep_duration = (
        wake_time - sleep_time
    ).total_seconds() / 3600  # Convert seconds to hours

    return sleep_duration
