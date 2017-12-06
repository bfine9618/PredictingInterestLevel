import pandas as pd
import numpy as np

import statsmodels.formula.api as smf

from sklearn.linear_model import LogisticRegressionCV as logCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def runModel():
    
    train = pd.read_json('./cleaned/train.json')
    test = pd.read_json('./cleaned/test.json')
    
    #Determine the columns with which to run an OLS, exclude the indicator column
    data = train.drop('interestVal', axis=1).select_dtypes(exclude=['object'])

    #join columns to build to equation
    equation = ('+').join(data.columns)

    #run the OLS to determine significant columns
    model = smf.ols('interestVal~'+equation, data=train).fit()

    #make a DF of significant features
    sig_features = pd.DataFrame(model.pvalues, index=data.columns, columns={'P_Value'})

    sigCols = sig_features[sig_features['P_Value']<.1].index.values
    print('The data has {} significant columns'.format(len(sigCols)))
    print('The significant columns are: ')
    print(sig_features[sig_features['P_Value']<.1])
    print()

    sigCols = np.append(sigCols, 'interest_level')

    #Create a simplified df with only the significant columns
    validLogTest = test[~pd.isnull(test['prob_interest_building'])]
    
    simpleTrain = train[sigCols]
    simpleTest = validLogTest[sigCols]

    X_train, X_test, y_train, y_test = train_test_split(simpleTrain.drop('interest_level',axis=1),
                                                    simpleTrain['interest_level'], test_size=0.33, 
                                                    random_state=42)
    
    print('Running Logistic Regression on best data...')
    
    logReg = logCV(cv=10)
    logReg.fit(X_train, y_train)
    print(classification_report(logReg.predict(X_test), y_test))

    logReg.fit(simpleTrain.drop('interest_level', axis=1), simpleTrain['interest_level'])
    logPreds = logReg.predict(simpleTest.drop('interest_level', axis=1))

    validLogTest['interest_level'] = logPreds
    
    print('Running SVM on lower quality data...')
    
    #Because some of the data is still unknown, we have to use an SVM to classify about 48% of the test data
    svmTest = test[pd.isnull(test['prob_interest_building'])]
    
    #______________SVM____________#
    
    return validLogTest['interest_level'], test
    

