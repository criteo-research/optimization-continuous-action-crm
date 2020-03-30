import sys
import pandas as pd 
import numpy as np 
if not len(sys.argv) == 3:
    print("usage:", sys.argv[0], "[input_dataset.csv] [output_dataset.npy]")
    sys.exit(-1)
print('Reading csv file...') 
full_dataset_path = sys.argv[1] 
output_path = sys.argv[2]
df = pd.read_csv(full_dataset_path) 
print('Csv file loaded!') 
array = df.values.astype(np.float64) 
np.save(output_path, array) print('Data saved!')
