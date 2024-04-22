import sys

def extract_logs_to_csv(input_file_path, output_file_path):
    # Define start and end patterns as exact strings
    start_pattern = "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss"
    end_pattern = "Trade History:"
    
    # Variables to manage state and buffer
    capture = False
    buffer = []

    with open(input_file_path, 'r', encoding='utf-8') as file, \
         open(output_file_path, 'w', encoding='utf-8') as output_file:
        
        # Write headers to the CSV file
        output_file.write(start_pattern + '\n')
        
        for line in file:
            if line.strip() == start_pattern:
                capture = True  # Start capturing after the header line
                continue  # Skip the header line
            
            if line.strip() == end_pattern and capture:
                capture = False  # Stop capturing after the end pattern
            
            if capture:
                output_file.write(line)  # Write the line if we are in the capturing state

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_logs_to_csv.py input_file output_file")
        sys.exit(1)
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    extract_logs_to_csv(input_file_path, output_file_path)
