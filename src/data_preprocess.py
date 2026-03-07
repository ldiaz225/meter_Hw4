import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessing:
    def __init__(self):
        pass

    def load_data(self, path):
        # Load raw data
        data = pd.read_csv(path, sep="\t", header=None).dropna()
        print(data.head())

        # Convert to numpy
        data = data.to_numpy()

    
        # 1. TRAIN / VALIDATION / TEST SPLIT
    
        train_validation, test = train_test_split(
            data, test_size=0.2, random_state=12
        )

        train, validation = train_test_split(
            train_validation, test_size=0.2, random_state=99
        )

     
        # 2. CHECK CLASS DISTRIBUTION
        
        print("Train:", set(train[:, -1]))
        print("Validation:", set(validation[:, -1]))
        print("Test:", set(test[:, -1]))

       
        # 3. SEPARATE FEATURES AND LABELS
     
        X_train = train[:, :-1]
        y_train = train[:, -1]

        X_val = validation[:, :-1]
        y_val = validation[:, -1]

        X_test = test[:, :-1]
        y_test = test[:, -1]

     
        # 4. SCALE USING ONLY TRAINING DATA
     
        scaler = StandardScaler().fit(X_train)

        X_train_scaled = scaler.transform(X_train)
        X_val_scaled   = scaler.transform(X_val)
        X_test_scaled  = scaler.transform(X_test)

     
        # 5. RETURN EVERYTHING CLEANLY
      
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "X_train_scaled": X_train_scaled,
            "X_val_scaled": X_val_scaled,
            "X_test_scaled": X_test_scaled,
            "scaler": scaler,
            "raw_data": data
        }
    

'''
***How to Use: 

from data_preprocess import DataPreprocessing

pre = DataPreprocessing()
data_dict = pre.load_data("data/Meter_A.txt")

X_train = data_dict["X_train"]
y_train = data_dict["y_train"]

X_train_scaled = data_dict["X_train_scaled"]
'''