import datetime as dt
import csv
import os
import time
from fmiopendata.wfs import download_stored_query

bbox = "bbox=21.0,62.5,24.0,64.5"
station_name = "Pietarsaari Kallan"

end_time = dt.datetime.now(dt.timezone.utc)
start_time = end_time - dt.timedelta(days=2*365)
delta = dt.timedelta(days=7)
all_data = {}

current_start = start_time

while current_start < end_time:
    current_end = min(current_start + delta, end_time)
    start_str = current_start.isoformat(timespec="seconds").replace("+00:00", "Z")
    end_str = current_end.isoformat(timespec="seconds").replace("+00:00", "Z")
    print(f"Fetching: {start_str} to {end_str}")

    # Retry logic for connection errors
    max_retries = 3
    obs = None
    for attempt in range(max_retries):
        try:
            obs = download_stored_query(
                "fmi::observations::weather::multipointcoverage",
                args=[bbox, f"starttime={start_str}", f"endtime={end_str}"]
            )
            break  # Success, exit retry loop
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2, 4, 6 seconds
                print(f"  Error: {e}. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts. Skipping this interval.")
                obs = None

    # Skip if request failed or no data in this chunk
    if obs is None or not obs.data:
        print("No data found for this interval.")
        current_start = current_end
        continue

    # Collect only one post per day for the station: choose closest to 12:00 UTC, if possible
    per_day_data = {}
    for t in sorted(obs.data.keys()):
        if station_name not in obs.data[t]:
            continue
        t_dt = dt.datetime.fromisoformat(t.replace("Z", "+00:00"))
        day = t_dt.date()
        hour_diff = abs((t_dt.hour + t_dt.minute/60) - 12)  # Difference from 12:00
        if day not in per_day_data or hour_diff < per_day_data[day][1]:
            per_day_data[day] = (t, hour_diff, obs.data[t][station_name])
    # Add to all_data
    for day, (t, _, data) in per_day_data.items():
        all_data[t] = data

    current_start = current_end
    
    # Add delay between requests to avoid rate limiting
    time.sleep(1.0)

if not all_data:
    print("No weather data found för Pietarsaari Kallan in the given period.")
else:
    # Prepare to write to CSV
    output_dir = "data/weather"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "weather_jakobstad.csv")

    # Collect all unique parameter names
    param_names = set()
    for t in all_data:
        param_names.update(all_data[t].keys())
    param_names = sorted(param_names)

    # Write header: Time, param1, param2, ...
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        header = ["Time"] + param_names
        writer.writerow(header)

        # Write one row per day (already filtered for one post per day)
        for t in sorted(all_data.keys()):
            row = [t]
            for param in param_names:
                value = all_data[t].get(param, {}).get("value", "")
                row.append(value)
            writer.writerow(row)

    print(f"Weather data for {station_name} saved to {output_file}")