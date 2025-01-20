import os
import pandas as pd
import argparse
from datetime import datetime, timedelta
import json
def process_csv(
    csv_path: str,
    weight_voice: int = 2,
    weight_sms: int = 1,
    weight_data: int = 1,
    time_gap_minutes: int = 15,
    start_date: str = None,
    end_date: str = None,
):
    """
    Reads the CSV file, cleans and enriches the data, groups records within a time gap (time_gap_minutes),
    determines the predominant state, and computes a confidence percentage.

    Steps:
      1) Discard rows with lat=0, lon=0.
      2) If 'Record Type' is empty, assign the minimum weight.
      3) If 'State' is "unknown" or empty but lat/lon is valid:
         - Look for another row with the same lat/lon that has a known State/County,
           and use it to "recover" the missing info.
      4) If time difference is less than time_gap_minutes and both previous and next row have the same State,
         interpolate that State.
      5) Add 'imputed' column with 'original', 'recovered', or 'interpolated'.
         - 'interpolated' rows get the minimum weight.
         - 'recovered' rows keep the original weight for Record Type.
      6) Group by time intervals, compute the predominant State and confidence.
    
    :param csv_path: Path to the CSV file with at least:
                     'Local Date & Time', 'Latitude', 'Longitude', 'County', 'State', 'Record Type'
    :param weight_voice: Weight for 'Voice' Record Type
    :param weight_sms: Weight for 'SMS' Record Type
    :param weight_data: Weight for 'Data' Record Type
    :param time_gap_minutes: Time gap in minutes to group records
    :return: List of dictionaries: [ {start_time, end_time, state, confidence, count_rows, count_recovered, count_interpolated}, ... ]
    """

    # 1. Check if file exists
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # 2. Read CSV with error handling
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV '{csv_path}': {e}") from e

    # 3. Validate required columns
    required_cols = ['Local Date & Time', 'Latitude', 'Longitude', 'County', 'State', 'Record Type']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # 4. Convert 'Local Date & Time' to datetime
    try:
        df['Local Date & Time'] = pd.to_datetime(df['Local Date & Time'], format='%m/%d/%y %H:%M', errors='coerce')
    except Exception as e:
        raise ValueError(f"Error converting 'Local Date & Time' to datetime: {e}") from e
    
    # 5. Filtrar por rango de fechas, si se proporcionan
    if start_date:
        start_date_dt = pd.to_datetime(start_date, format='%Y-%m-%d', errors='coerce')
        df = df[df['Local Date & Time'] >= start_date_dt]

    if end_date:
        end_date_dt = pd.to_datetime(end_date, format='%Y-%m-%d', errors='coerce')
        df = df[df['Local Date & Time'] <= end_date_dt]

    if df.empty:
        raise ValueError("No data after applying date filters.")
    # 6. Sort by date/time
    df = df.sort_values(by='Local Date & Time').reset_index(drop=True)

    # 7. Discard rows with lat=0, lon=0
    df = df[~((df['Latitude'] == 0) & (df['Longitude'] == 0))].copy()
    df = df.reset_index(drop=True)

    # 78. Remove duplicates ignoring 'Item Number'
    #    We'll keep the first occurrence.
    #    Subset includes all columns except 'Item Number' if it exists.
    #    If your actual CSV name differs (e.g. "ItemNumber"), adjust accordingly.
    dedup_subset = ['Local Date & Time', 'Latitude', 'Longitude', 'County', 'State', 'Record Type']
    df = df.drop_duplicates(subset=dedup_subset, keep='first').reset_index(drop=True)
    
    # 9. Add 'imputed' column. Start as 'original' for all rows
    df['imputed'] = 'original'

    # 10. Determine the minimum weight
    weight_min = min(weight_voice, weight_sms, weight_data)

    # 11. Handle empty Record Type => assign minimum weight
    #    We'll store an intermediate column 'raw_record_type' to keep the original
    df['raw_record_type'] = df['Record Type'].astype(str)  # ensure string
    df.loc[df['raw_record_type'].str.strip() == '', 'raw_record_type'] = 'empty'

    # 12. Helper function to assign weight based on record type + 'imputed'
    def assign_weight(row):
        """
        Assigns a weight based on Record Type and whether it's interpolated.
        """
        rtype = row['raw_record_type'].lower().strip()
        imputed_status = row['imputed']
        
        # If it's 'empty', use the min weight
        if rtype == 'empty':
            base_weight = weight_min
        else:
            # Normal logic for record type
            if rtype == 'voice':
                base_weight = weight_voice
            elif rtype == 'sms':
                base_weight = weight_sms
            elif rtype == 'data':
                base_weight = weight_data
            else:
                base_weight = 1  # default

        # If row is 'interpolated', we override with the minimum weight
        if imputed_status == 'interpolated':
            return weight_min
        else:
            return base_weight

    # 13. Build a dictionary of lat/lon -> (County, State) from rows with real info
    known_locations = {}
    for i, r in df.iterrows():
        lat, lon = r['Latitude'], r['Longitude']
        county, state = str(r['County']).strip(), str(r['State']).strip()
        if county.lower() != 'unknown' and county != '' and state.lower() != 'unknown' and state != '':
            known_locations[(lat, lon)] = (county, state)

    # 14. "Recover" State/County if we find the same lat/lon with known data
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

    # 15. Interpolate if previous and next have same State, within time gap
    for i in range(len(df)):
        state_val = str(df.at[i, 'State']).strip().lower()
        if state_val == '' or state_val == 'unknown':
            prev_i = i - 1
            next_i = i + 1
            if prev_i >= 0 and next_i < len(df):
                prev_state = str(df.at[prev_i, 'State']).strip().lower()
                next_state = str(df.at[next_i, 'State']).strip().lower()

                if (prev_state != '' and prev_state != 'unknown' and
                    next_state != '' and next_state != 'unknown' and
                    prev_state == next_state):
                    
                    prev_time = df.at[prev_i, 'Local Date & Time']
                    curr_time = df.at[i, 'Local Date & Time']
                    next_time = df.at[next_i, 'Local Date & Time']

                    gap1 = abs((curr_time - prev_time).total_seconds() / 60.0)
                    gap2 = abs((next_time - curr_time).total_seconds() / 60.0)

                    if gap1 <= time_gap_minutes and gap2 <= time_gap_minutes:
                        df.at[i, 'State'] = df.at[prev_i, 'State']
                        df.at[i, 'County'] = df.at[prev_i, 'County']
                        df.at[i, 'imputed'] = 'interpolated'

    # 16. Assign final weight
    df['weight'] = df.apply(assign_weight, axis=1)

    # 17. Group by time intervals
    intervals = []
    current_group = []

    def close_current_group(group):
        """
        Closes a group and returns a dict with:
        [start_time, end_time, state, confidence, count_rows, count_recovered, count_interpolated].
        """
        if not group:
            return None

        start_time = group[0]['Local Date & Time']
        end_time = group[-1]['Local Date & Time']

        # Sum weights by state
        state_weights = {}
        total_weight = 0.0

        # We'll also count how many were recovered or interpolated
        count_rows = len(group)
        count_recovered = 0
        count_interpolated = 0

        for row in group:
            st = str(row['State']).strip()
            w = row['weight']
            imp = row['imputed']
            if st == '' or st.lower() == 'unknown':
                st = 'Unknown'
            state_weights[st] = state_weights.get(st, 0.0) + w
            total_weight += w

            if imp == 'recovered':
                count_recovered += 1
            elif imp == 'interpolated':
                count_interpolated += 1

        if total_weight == 0:
            return None

        # Predominant state
        likely_state = max(state_weights, key=state_weights.get)
        max_weight = state_weights[likely_state]
        confidence = (max_weight / total_weight) * 100

        return {
            'start_time': start_time,
            'end_time': end_time,
            'state': likely_state,
            'confidence': round(confidence, 2),
            'count_rows': count_rows,
            'count_recovered': count_recovered,
            'count_interpolated': count_interpolated
        }

    # Iterate through rows and group by time gap
    for idx, row in df.iterrows():
        if not current_group:
            current_group.append(row)
        else:
            prev_time = current_group[-1]['Local Date & Time']
            curr_time = row['Local Date & Time']
            diff_minutes = (curr_time - prev_time).total_seconds() / 60.0

            if diff_minutes <= time_gap_minutes:
                current_group.append(row)
            else:
                # Close the previous group
                interval_info = close_current_group(current_group)
                if interval_info:
                    intervals.append(interval_info)
                # Start a new group
                current_group = [row]

    # Close last group if remaining
    if current_group:
        interval_info = close_current_group(current_group)
        if interval_info:
            intervals.append(interval_info)

    return intervals


def main():
    # Argument parser para manejar los parámetros desde la línea de comandos
    parser = argparse.ArgumentParser(description="Process CSV for location data")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("weight_voice", type=int, help="Weight for Voice")
    parser.add_argument("weight_sms", type=int, help="Weight for SMS")
    parser.add_argument("weight_data", type=int, help="Weight for Data")
    parser.add_argument("time_gap_minutes", type=int, help="Time gap in minutes")
    parser.add_argument("--start_date", type=str, help="Start date for filtering (YYYY-MM-DD)", default=None)
    parser.add_argument("--end_date", type=str, help="End date for filtering (YYYY-MM-DD)", default=None)
    args = parser.parse_args()

    try:
        # Procesamos el archivo CSV
        results = process_csv(
            args.csv_file,
            weight_voice=args.weight_voice,
            weight_sms=args.weight_sms,
            weight_data=args.weight_data,
            time_gap_minutes=args.time_gap_minutes,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        # Convert to JSON
        serializable_results = [
        {key: (str(value) if isinstance(value, pd.Timestamp) else value) for key, value in result.items()}
        for result in results
        ]
        # Convertimos los resultados a JSON y los imprimimos como una única salida
        print(json.dumps({"status": "success", "data": serializable_results}, indent=4))

    except FileNotFoundError as fnfe:
        # Capturamos errores de archivo no encontrado y los imprimimos como JSON
        print(json.dumps({"status": "error", "message": f"File not found: {fnfe}"}))
        exit(1)  # Salimos con código de error

    except ValueError as ve:
        # Capturamos errores de valor y los imprimimos como JSON
        print(json.dumps({"status": "error", "message": f"Processing issue: {ve}"}))
        exit(1)

    except Exception as e:
        # Capturamos errores inesperados y los imprimimos como JSON
        print(json.dumps({"status": "error", "message": f"Unexpected error: {e}"}))
        exit(1)


if __name__ == "__main__":
    main()