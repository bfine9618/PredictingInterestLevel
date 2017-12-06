from Data_Cleaning_PreProcessing import cleanPreprocessData
from Final_Model_Function import runModel
import pandas as pd
import numpy as np

train = pd.read_json('./raw_data/train_data.json')
test = pd.read_json('./raw_data/test_data.json')

cleanedDF, cleanedTyped = cleanPreprocessData(train, test)
print('Data Cleaning and Preprocessing complete.')
print('Running model now...')

preds, test = runModel()