import json
import pickle
import numpy as np

import os
script_dir = os.path.dirname(__file__)
json_file_path = os.path.join(script_dir, 'artifacts/columns.json')

model_file_path = os.path.join(script_dir, 'artifacts/banglore_home_prices_model.pickle')

print(json_file_path)
__locations = None
__data_columns  = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath 
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)


def get_location_names():
    load_saved_atrifacts()
    return __locations

def load_saved_atrifacts():
    print('loading saved artifacts')
    global __data_columns
    global __locations
    global __model

    with open(json_file_path, 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open(model_file_path, 'rb') as f:
        __model = pickle.load(f)
    
    print('loaded saved artifacts')

if __name__ == '__main__':
    load_saved_atrifacts()
    print(get_estimated_price('dhaka', 1000, 2,2))