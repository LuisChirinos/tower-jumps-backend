import os
import pandas as pd
import argparse
from datetime import datetime, timedelta
import json
import math
import numpy as np

# For clustering
from sklearn.cluster import DBSCAN

# To improve (for border distance checks), we might use:
# from shapely.geometry import Point
# import geopandas as gpd


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute the distance in kilometers between two latitude/longitude points
    using the Haversine formula.
    """
    R = 6371.0  # approximate radius of earth in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon/2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


def process_csv(
    csv_path: str,
    weight_voice: int = 2,
    weight_sms: int = 1,
    weight_data: int = 1,
    time_gap_minutes: int = 15,
    start_date: str = None,
    end_date: str = None,
    # New DBSCAN params
    eps_meters: float = 100.0,
    min_samples: int = 2,
    # New speed filter
    max_speed_km_h: float = 120.0,
    # Optional shapefile param (commented out here):
    # border_shapefile: str = None
):
    """
    Reads the CSV file, cleans/enriches data, recovers/interpolates unknown states,
    groups rows by time intervals, applies DBSCAN to find a principal cluster's centroid,
    calculates the predominant state, checks border distance (optional), and filters
    out impossible jumps by speed.

    Original Steps (Recovery & Interpolation):
      1) Discard rows with lat=0, lon=0.
      2) If 'Record Type' is empty, assign the minimum weight.
      3) If 'State' is "unknown" or empty but lat/lon is valid, look for another row
         with the same lat/lon that has a known State/County, and "recover" that info.
      4) If the time difference is < time_gap_minutes and both previous and next have
         the same State, "interpolate" that State.
      5) Add 'imputed' column with 'original', 'recovered', or 'interpolated'.
         - 'interpolated' rows get min weight.
         - 'recovered' rows keep original Record Type weight.
      6) Group by time intervals, compute the predominant State and confidence.

    New Steps (Geometric Approach, Border Distance, Impossible Jumps):
      7) Inside each time group, use DBSCAN to find the largest cluster of lat/lon points.
      8) Compute that cluster's centroid, define a predominant state from points inside.
      9) (Optional) If the centroid is near the border, lower confidence.
     10) Filter impossible jumps between consecutive intervals if speed > max_speed_km_h.

    :param csv_path: Path to the CSV file with columns:
                     'Local Date & Time', 'Latitude', 'Longitude', 'County', 'State', 'Record Type'
    :param weight_voice: Weight for 'Voice' Record Type
    :param weight_sms: Weight for 'SMS' Record Type
    :param weight_data: Weight for 'Data' Record Type
    :param time_gap_minutes: time gap in minutes used to segment data into intervals
    :param start_date: optional start date filter, format 'YYYY-MM-DD'
    :param end_date: optional end date filter, format 'YYYY-MM-DD'
    :param eps_meters: radius in meters for DBSCAN clustering
    :param min_samples: minimum samples to form a cluster in DBSCAN
    :param max_speed_km_h: if the jump between two consecutive intervals requires a speed
                           above this value, we lower confidence or handle the jump
    :return: list of dictionaries with interval info
    """

    # --- 1. Verify the CSV file exists ---
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # --- 2. Read CSV with error handling ---
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV '{csv_path}': {e}") from e

    # --- 3. Validate required columns ---
    required_cols = ['Local Date & Time', 'Latitude', 'Longitude', 'County', 'State', 'Record Type']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # --- 4. Convert 'Local Date & Time' to datetime ---
    df['Local Date & Time'] = pd.to_datetime(
        df['Local Date & Time'], format='%m/%d/%y %H:%M', errors='coerce'
    )

    # --- 5. Filter by date range if provided ---
    if start_date:
        start_date_dt = pd.to_datetime(start_date, format='%Y-%m-%d', errors='coerce')
        df = df[df['Local Date & Time'] >= start_date_dt]

    if end_date:
        end_date_dt = pd.to_datetime(end_date, format='%Y-%m-%d', errors='coerce')
        df = df[df['Local Date & Time'] <= end_date_dt]

    if df.empty:
        raise ValueError("No data after applying date filters.")

    # --- 6. Sort by time ---
    df = df.sort_values(by='Local Date & Time').reset_index(drop=True)

    # --- 7. Discard rows with lat=0, lon=0 ---
    df = df[~((df['Latitude'] == 0) & (df['Longitude'] == 0))].copy()
    df = df.reset_index(drop=True)

    # --- 8. Remove duplicates ignoring 'Item Number' (original step #78) ---
    dedup_subset = ['Local Date & Time', 'Latitude', 'Longitude', 'County', 'State', 'Record Type']
    df = df.drop_duplicates(subset=dedup_subset, keep='first').reset_index(drop=True)

    # --- 9. Initialize 'imputed' column as 'original' ---
    df['imputed'] = 'original'

    # --- 10. Compute the minimum weight among voice/sms/data ---
    weight_min = min(weight_voice, weight_sms, weight_data)

    # --- 11. Handle empty Record Type => assign minimum weight ---
    df['raw_record_type'] = df['Record Type'].astype(str)  # ensure string
    df.loc[df['raw_record_type'].str.strip() == '', 'raw_record_type'] = 'empty'

    # Helper function for assigning weight
    def assign_weight(row):
        """
        Assigns a weight based on record type and 'imputed' status.
        If 'imputed' == 'interpolated', we override with the minimum weight.
        """
        rtype = row['raw_record_type'].lower().strip()
        imputed_status = row['imputed']

        # Base weight by record type
        if rtype == 'empty':
            base_weight = weight_min
        elif rtype == 'voice':
            base_weight = weight_voice
        elif rtype == 'sms':
            base_weight = weight_sms
        elif rtype == 'data':
            base_weight = weight_data
        else:
            base_weight = 1  # default fallback

        # If row is 'interpolated', forcibly use minimum weight
        if imputed_status == 'interpolated':
            return weight_min
        else:
            return base_weight

    # --- 12. Build a dict of known (lat, lon) -> (County, State) for "recovery" ---
    known_locations = {}
    for i, r in df.iterrows():
        lat, lon = r['Latitude'], r['Longitude']
        county, state = str(r['County']).strip(), str(r['State']).strip()
        if (county.lower() != 'unknown' and county != ''
            and state.lower() != 'unknown' and state != ''):
            known_locations[(lat, lon)] = (county, state)

    # --- 13. Recover unknown states if there's a known location with the same lat/lon ---
    for i in range(len(df)):
        lat, lon = df.at[i, 'Latitude'], df.at[i, 'Longitude']
        state_val = str(df.at[i, 'State']).strip().lower()
        if state_val == 'unknown' or state_val == '':
            if (lat, lon) in known_locations:
                rec_county, rec_state = known_locations[(lat, lon)]
                df.at[i, 'County'] = rec_county
                df.at[i, 'State'] = rec_state
                if df.at[i, 'imputed'] == 'original':
                    df.at[i, 'imputed'] = 'recovered'

    # --- 14. Interpolate unknown states if prev/next have the same known state ---
    for i in range(len(df)):
        state_val = str(df.at[i, 'State']).strip().lower()
        if state_val == '' or state_val == 'unknown':
            prev_i = i - 1
            next_i = i + 1
            if prev_i >= 0 and next_i < len(df):
                prev_state = str(df.at[prev_i, 'State']).strip().lower()
                next_state = str(df.at[next_i, 'State']).strip().lower()

                if (prev_state != '' and prev_state != 'unknown'
                    and next_state != '' and next_state != 'unknown'
                    and prev_state == next_state):
                    
                    prev_time = df.at[prev_i, 'Local Date & Time']
                    curr_time = df.at[i, 'Local Date & Time']
                    next_time = df.at[next_i, 'Local Date & Time']

                    gap1 = abs((curr_time - prev_time).total_seconds() / 60.0)
                    gap2 = abs((next_time - curr_time).total_seconds() / 60.0)

                    if gap1 <= time_gap_minutes and gap2 <= time_gap_minutes:
                        df.at[i, 'State'] = df.at[prev_i, 'State']
                        df.at[i, 'County'] = df.at[prev_i, 'County']
                        df.at[i, 'imputed'] = 'interpolated'

    # --- 15. Assign final weight to each row ---
    df['weight'] = df.apply(assign_weight, axis=1)

    # We will group by time intervals, then cluster within each group.
    intervals = []
    current_group = []

    def most_common_state(rows):
        """
        Find the predominant (weighted) state among a list of rows.
        We sum the 'weight' for each state and pick the max.
        """
        from collections import Counter
        state_weights = Counter()
        for r in rows:
            st = (r['State'] or 'unknown').lower()
            w = r['weight']
            state_weights[st] += w
        return max(state_weights, key=state_weights.get)

    def close_current_group(group):
        """
        Closes a time group:
         1) Use DBSCAN to cluster lat/lon within the group.
         2) Find the largest (non-noise) cluster, compute centroid, pick predominant state.
         3) Calculate confidence = (size of cluster / total points in group) * 100, by default.
         4) Return a dict with [start_time, end_time, state, confidence, counts, centroid coords].
        """
        if not group:
            return None

        # Sort by time for clarity
        group = sorted(group, key=lambda x: x['Local Date & Time'])
        start_time = group[0]['Local Date & Time']
        end_time = group[-1]['Local Date & Time']

        # Extract lat/lon in array
        coords = np.array([[row['Latitude'], row['Longitude']] for row in group])

        # Convert eps_meters to approximate degrees for DBSCAN
        meters_per_degree = 111000.0  # approx at mid-latitudes
        eps_degs = eps_meters / meters_per_degree

        # Run DBSCAN
        clustering = DBSCAN(eps=eps_degs, min_samples=min_samples).fit(coords)
        labels = clustering.labels_

        # Count how many points in each label (ignoring noise = -1)
        from collections import Counter
        label_counts = Counter(labels)
        if -1 in label_counts:
            del label_counts[-1]

        # If everything is noise, fallback to simple average
        if not label_counts:
            centroid_lat = coords[:, 0].mean()
            centroid_lon = coords[:, 1].mean()
            final_state = most_common_state(group)
            cluster_confidence = 50.0  # arbitrary fallback
        else:
            # Find the label with the largest cluster
            main_label = max(label_counts, key=label_counts.get)
            cluster_points = coords[labels == main_label]
            centroid_lat = cluster_points[:, 0].mean()
            centroid_lon = cluster_points[:, 1].mean()
            cluster_size = len(cluster_points)
            total_size = len(coords)
            cluster_confidence = (cluster_size / total_size) * 100.0

            # Determine the predominant state among the cluster points
            # We'll gather the subset of 'group' belonging to 'main_label'
            cluster_subset = []
            for row in group:
                idx_in_group = group.index(row)
                # The same index in group => the same index in 'labels'
                if labels[idx_in_group] == main_label:
                    cluster_subset.append(row)
            final_state = most_common_state(cluster_subset)

        # To improve: check distance to border shapefile (commented out example)
        # if border_shapefile:
        #     dist_to_border = distance_from_border(centroid_lat, centroid_lon, border_shapefile)
        #     if dist_to_border < 100.0:  # e.g. 100 meters
        #         cluster_confidence *= 0.8

        # Count how many are recovered/interpolated
        count_rows = len(group)
        count_recovered = sum(1 for r in group if r['imputed'] == 'recovered')
        count_interpolated = sum(1 for r in group if r['imputed'] == 'interpolated')

        # Return the dictionary
        return {
            'start_time': start_time,
            'end_time': end_time,
            'state': final_state,
            'confidence': round(cluster_confidence, 2),
            'centroid_lat': round(centroid_lat, 6),
            'centroid_lon': round(centroid_lon, 6),
            'count_rows': count_rows,
            'count_recovered': count_recovered,
            'count_interpolated': count_interpolated
        }

    # (To Improve, define a function for border distance if you have shapefiles)
    """
    def distance_from_border(lat, lon, shapefile_path):
        # Example: uses shapely and geopandas to measure distance in degrees or meters
        gdf = gpd.read_file(shapefile_path)
        point = Point(lon, lat)  # shapely uses (x=lon, y=lat)
        min_dist = float('inf')
        for geom in gdf['geometry']:
            dist = point.distance(geom)
            if dist < min_dist:
                min_dist = dist
        return min_dist
    """

    # --- Group rows by time gap ---
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        if not current_group:
            current_group.append(row_dict)
        else:
            prev_time = current_group[-1]['Local Date & Time']
            curr_time = row_dict['Local Date & Time']
            diff_minutes = (curr_time - prev_time).total_seconds() / 60.0
            if diff_minutes <= time_gap_minutes:
                current_group.append(row_dict)
            else:
                # Close off the previous group
                interval_info = close_current_group(current_group)
                if interval_info:
                    intervals.append(interval_info)
                # Start a new group
                current_group = [row_dict]

    # Close the last group if it exists
    if current_group:
        interval_info = close_current_group(current_group)
        if interval_info:
            intervals.append(interval_info)

    # --- Filter impossible jumps by speed between consecutive intervals ---
    for i in range(len(intervals) - 1):
        lat1, lon1 = intervals[i]['centroid_lat'], intervals[i]['centroid_lon']
        lat2, lon2 = intervals[i+1]['centroid_lat'], intervals[i+1]['centroid_lon']
        t1 = intervals[i]['end_time']
        t2 = intervals[i+1]['start_time']

        # Calculate time difference in hours
        time_diff_hours = (t2 - t1).total_seconds() / 3600.0
        if time_diff_hours <= 0:
            continue

        # Calculate distance in km
        dist_km = haversine_distance(lat1, lon1, lat2, lon2)
        speed_km_h = dist_km / time_diff_hours

        # If speed is unrealistic, reduce confidence or override
        if speed_km_h > max_speed_km_h:
            # Example: reduce the next interval's confidence by 50%
            intervals[i+1]['confidence'] = round(intervals[i+1]['confidence'] * 0.5, 2)
            # Alternatively, you could set intervals[i+1]['state'] = intervals[i]['state']
            # or do other business-logic as needed.

    return intervals


def main():
    parser = argparse.ArgumentParser(description="Process CSV for location data (Geometric Approach + Recovery/Interpolation)")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("weight_voice", type=int, help="Weight for Voice")
    parser.add_argument("weight_sms", type=int, help="Weight for SMS")
    parser.add_argument("weight_data", type=int, help="Weight for Data")
    parser.add_argument("time_gap_minutes", type=int, help="Time gap (minutes) to group records")
    parser.add_argument("--start_date", type=str, help="Start date (YYYY-MM-DD)", default=None)
    parser.add_argument("--end_date", type=str, help="End date (YYYY-MM-DD)", default=None)
    parser.add_argument("--eps_meters", type=float, default=100.0, help="DBSCAN eps in meters")
    parser.add_argument("--min_samples", type=int, default=2, help="DBSCAN min_samples")
    parser.add_argument("--max_speed_km_h", type=float, default=120.0, help="Max speed (km/h) to filter jumps")
    # parser.add_argument("--border_shapefile", type=str, default=None, help="Path to a border shapefile (optional)")

    args = parser.parse_args()

    try:
        results = process_csv(
            csv_path=args.csv_file,
            weight_voice=args.weight_voice,
            weight_sms=args.weight_sms,
            weight_data=args.weight_data,
            time_gap_minutes=args.time_gap_minutes,
            start_date=args.start_date,
            end_date=args.end_date,
            eps_meters=args.eps_meters,
            min_samples=args.min_samples,
            max_speed_km_h=args.max_speed_km_h
            # border_shapefile=args.border_shapefile
        )

        # Convert to a JSON-friendly structure
        serializable_results = []
        for r in results:
            item = {}
            for k, v in r.items():
                if isinstance(v, pd.Timestamp):
                    item[k] = v.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    item[k] = v
            serializable_results.append(item)

        # Print the final JSON
        print(json.dumps({"status": "success", "data": serializable_results}, indent=4))

    except FileNotFoundError as fnfe:
        print(json.dumps({"status": "error", "message": f"File not found: {fnfe}"}))
        exit(1)
    except ValueError as ve:
        print(json.dumps({"status": "error", "message": f"Processing issue: {ve}"}))
        exit(1)
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Unexpected error: {e}"}))
        exit(1)


if __name__ == "__main__":
    main()
