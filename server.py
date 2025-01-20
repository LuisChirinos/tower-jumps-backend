import os
from flask import Flask, request, jsonify
from process_locations import process_csv

app = Flask(__name__)

@app.route('/process-csv', methods=['POST'])
def process_csv_endpoint():
    data = request.json
    file = data['file']
    weights = data['weights']
    time_gap = data['timeGap']
    start_date = data.get('dateFilter', {}).get('startDate')
    end_date = data.get('dateFilter', {}).get('endDate')
    
    print("Received data in Flask:")
    print("  Weights:", weights)
    print("  Time gap:", time_gap)
    print("  Start date:", start_date)
    print("  End date:", end_date)
    print("  File content (length):", len(file) if file else "No file")
    
    # Create a temporary directory for storing the CSV
    temp_dir = os.path.join(os.getcwd(), "tmp")  # Create a 'tmp' folder in the current working directory
    os.makedirs(temp_dir, exist_ok=True)        # Ensure the directory exists

    # Define the path for the temporary CSV file
    temp_file_path = os.path.join(temp_dir, "temp.csv")

    # Save the incoming file content to the temporary file
    with open(temp_file_path, "w") as temp_file:
        temp_file.write(file)

    try:
        # Process the CSV with the specified parameters
        results = process_csv(
            temp_file_path,
            weight_voice=weights['voice'],
            weight_sms=weights['sms'],
            weight_data=weights['data'],
            time_gap_minutes=time_gap,
            start_date=start_date,
            end_date=end_date,
        )
        
        print("Results from process_csv:", results)
        return jsonify({"data": results}), 200
    except Exception as e:
        # Return an error response if something goes wrong
        print("Error in process_csv_endpoint:", str(e))

        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
