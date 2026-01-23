import datetime as dt
import csv
import os
from fmiopendata.wfs import download_stored_query

bbox = "bbox=21.0,62.5,24.0,64.5"
station_name = "Pietarsaari Kallan"

# Only fetch weather for the last month
end_time = dt.datetime.now(dt.timezone.utc)
start_time = end_time - dt.timedelta(days=31)
delta = dt.timedelta(days=7)
all_data = {}

current_start = start_time
while current_start < end_time:
    current_end = min(current_start + delta, end_time)
    start_str = current_start.isoformat(timespec="seconds").replace("+00:00", "Z")
    end_str = current_end.isoformat(timespec="seconds").replace("+00:00", "Z")
    print(f"Fetching: {start_str} to {end_str}")

    obs = download_stored_query(
        "fmi::observations::weather::multipointcoverage",
        args=[bbox, f"starttime={start_str}", f"endtime={end_str}"]
    )

    # Skip if no data in this chunk
    if not obs.data:
        print("No data found for this interval.")
        current_start = current_end
        continue

    # Collect data for station
    for t in sorted(obs.data.keys()):
        if station_name in obs.data[t]:
            all_data[t] = obs.data[t][station_name]
    current_start = current_end

if not all_data:
    print("No weather data found for Pietarsaari Kallan in the given period.")
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

        for t in sorted(all_data.keys()):
            row = [t]
            for param in param_names:
                value = all_data[t].get(param, {}).get("value", "")
                row.append(value)
            writer.writerow(row)

    print(f"Weather data for {station_name} saved to {output_file}")