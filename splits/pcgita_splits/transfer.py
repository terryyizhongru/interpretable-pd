import pandas as pd

# Read the CSV file
input_file = '/home/yzhong/gits/SSL4PR/pcgita_splits//all.csv'
output_file = '/home/yzhong/gits/SSL4PR/pcgita_splits//all.txt'

# Load CSV
df = pd.read_csv(input_file)

# Write to pipe-delimited text file
df.to_csv(output_file, sep='|', index=False)

print(f"Converted CSV to pipe-delimited text file: {output_file}")