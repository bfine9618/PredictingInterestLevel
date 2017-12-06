from html.parser import HTMLParser
import pandas as pd
import numpy as np
import os

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def descrClean(x):
    des = strip_tags(x)
    return des.lower()

# Function to Classify Unit Types
def unitType(x, types):
    homeType = {    
        }
    for lst in types:
        homeType[lst[0]] = False
    
    for lst in types:
        for w in lst:
            if w in x:
                return lst[0]
    return 'other'


def cleanPreprocessData(train, test):
    print('Cleaning...')

    test['test'] = True

    df = train.append(test)
    df.reset_index(drop=True, inplace=True)
    df['test'].fillna(False, inplace=True)

    #Clean the column names for regressions and ML Models
    df.columns = [c.replace(' ', '_') for c in df.columns]
    df.columns = [c.replace('-', '_') for c in df.columns]
    df.columns = [c.replace('/', '_') for c in df.columns]

    #Confirm types for columns with numbers
    df['bedrooms'] = df['bedrooms'].apply(float)
    df['bedrooms'].fillna(0, inplace=True)
    df['bathrooms'].fillna(0, inplace=True)

    #Drop meaningless columns in data
    df.drop(['index', 'level_0'], axis=1, inplace=True)

    #Map Interest levels to values for OLS Regression
    df['interestVal'] = df['interest_level'].map({'high': 1, 'medium': 0.5, 'low':0})

    #Clean the HTML from descriptions to allow for NLP
    df['description'] = df['description'].apply(descrClean)

    # Aggregate to create one laundry in building column that isn't case sensitive
    df['laundry_in_building'] = df.apply(lambda row: row['Laundry_in_Building'] or row['Laundry_In_Building'], axis=1)

    # Drop old laundry in building columns
    df = df.drop(['Laundry_in_Building', 'Laundry_In_Building'], axis=1)

    cleanedDf = df

    print('Cleaning Complete. Processing descriptions to determine type...')

    # To determine the type of rental unit, we conduct a basic NLP
    # Define basic unit types
    apt = ['apartment', 'apt']
    condo = ['condominium', 'condo']
    walkUp = ['walk_up', 'walk-up', 'walkup', 'walk up']
    studio = ['studio']
    ph = ['ph', 'penhouse']
    townhome = ['townhome', 'duplex', 'townhouse']
    loft = ['loft']

    types = [apt, condo, walkUp, studio, ph, townhome, loft]

    #Determine rental type
    df['type'] = df['description'].apply(lambda x : unitType(x, types)) 

    #Determine if a type has been found
    df['foundType'] = ~df['type'].str.contains('other') 

    #Create binary dummy columns for each type
    df = pd.concat([df, pd.get_dummies(df['type'])], axis=1) 

    #Combine and drop the two loft column
    df['loft'].fillna(False, inplace=True)
    df['loft'] = df[['loft', 'Loft']].apply(lambda row : row['loft'] or row['Loft'], axis=1)
    df.drop('Loft', axis=1, inplace=True)

    cleanedTyped = df

    print('Typing Complete. Generating Interaction Terms...')

    # Generate interaction terms to find differentiators
    # Luxury Score Term - higher the score means the more luxury items included
    df['lux_score'] = (df['Exclusive'] + df['Doorman'] + df['Outdoor_Space'] + 
                        df['New_Construction'] + df['Roof_Deck'] + df['Fitness_Center'] + 
                        df['Swimming_Pool'] + df['Elevator'] + df['Laundry_in_Unit'] + 
                        df['Hardwood_Floors']) / 10

    #Group data by buildings and agents to determine expected interest -----MAGIC FEATURE-----
    agentGroup = df.groupby(['manager_id']).mean()
    buildingGroup = df.groupby(['building_id', 'manager_id']).mean()

    buildingAvg = buildingGroup[['interestVal']]
    buildingAvg.columns = ['prob_interest_building']
    buildingAvg.reset_index(inplace=True)

    managerAvg = agentGroup[['interestVal']]
    managerAvg.columns = ['prob_interest_manager']
    managerAvg.reset_index(inplace=True)

    #In case we are unfamiliar with the rental, we must decide how to assign probability
    def pBuildManager(pbuild, pmanager):
        if ~np.isnan(pbuild) and ~np.isnan(pmanager):
            return (pbuild + pmanager)/2
        return pmanager or pbuild

    #Merge back to original DF
    df = df.merge(managerAvg, on='manager_id', how='left')
    df = df.merge(buildingAvg, on=['building_id', 'manager_id'], how='left')

    #Compute expected interest by building and manager
    df['prob_buildManager'] = df.apply(lambda row: pBuildManager(row['prob_interest_building'], 
                                                             row['prob_interest_manager']), axis=1)

    #Count rooms and determine price per room
    df['rooms'] = df['bedrooms']+df['bathrooms']
    df['price_per_room'] = df['price']/df['rooms']

    # Number of Luxury Features Term
    df['num_luxury'] = (df['Exclusive'] + df['Doorman'] + df['Outdoor_Space'] + df['New_Construction'] + df['Roof_Deck'] + df['Fitness_Center'] + df['Swimming_Pool'] + df['Elevator'] + df['Laundry_in_Unit'] + df['Hardwood_Floors'])

    # Number of Features per Listing
    df['num_features'] = df['features'].apply(len)

    # ADA compatible interaction term
    # 1 if both elevator and wheelchair access, 0 if one or neither are included
    df['ada'] = df['Elevator'] * df['Wheelchair_Access']

    # Create transformed term that creates a score for outdoor spaces
    # Higher the score, the more of these features are included
    df['outdoor_score'] = (df['Outdoor_Space'] + df['Balcony'] + df['Common_Outdoor_Space'] 
                           + df['Garden_Patio'] + df['Roof_Deck'] + df['Terrace']) / 6

    # Create interaction term for fitness oriented
    # 1 if both swimming pool and fitness center are included, 0 if one or neither included
    df['fitness_oriented'] = df['Fitness_Center'] * df['Swimming_Pool']

    # Create interaction term for doorman/exclusive
    # 1 if both are included, 0 if one or neither are included
    df['door_excl'] = df['Doorman'] * df['Exclusive']

    # Create interaction term for cats and dogs allowed
    # 1 if both are allowed, 0 if one or neither are allowed
    df['pets_allowed'] = df['Cats_Allowed'] * df['Dogs_Allowed']

    #Compute price per feature and price per luxury feature. 
    #If no features exist, the value is empty
    df['price_per_feature'] = df['price']/df['num_features']
    df['price_per_feature'].replace(np.inf, np.nan, inplace=True)

    df['price_per_num_lux'] = df['price']/df['num_luxury']
    df['price_per_num_lux'].replace(np.inf, np.nan, inplace=True)

    #Determine expected prices by type of unit
    g1 = df.groupby(['type']).mean()
    g1.reset_index(inplace=True)

    #Columns we wish to average
    avgs = g1[['type','lux_score', 'num_features', 
               'num_luxury','outdoor_score', 'price_per_num_lux', 
               'price_per_feature']]

    pd.options.mode.chained_assignment = None  # default='warn'

    #Rename columns and merge back to original DF
    avgs.columns = ['avg_'+x for x in avgs]
    avgs.rename(columns={'avg_type':'type'}, inplace=True)
    df = pd.merge(df, avgs, on='type')

    #If no price was found, set the price for the column as average to avoid skewing the data
    df['price_per_num_lux'].fillna(df['avg_price_per_num_lux'], inplace=True)
    df['outdoor_score'].fillna(df['avg_outdoor_score'], inplace=True)
    df['lux_score'].fillna(df['avg_lux_score'], inplace=True)
    df['price_per_feature'].fillna(df['avg_price_per_feature'], inplace=True)

    df['price_lux_ratio'] = df['price_per_num_lux']/df['avg_price_per_num_lux']
    df['outdoor_ratio'] = df['outdoor_score']/df['avg_outdoor_score']
    df['lux_ratio'] = df['lux_score']/df['avg_lux_score']
    df['price_feature_ratio'] = df['price_per_feature']/df['avg_price_per_feature']


    #Compute the number of photos included in the listing
    df['numPhotos'] = df['photos'].apply(len)

    #Listing id is an arbitrary int label assined to each listing. not useful for classification
    df.drop(['listing_id'], axis=1, inplace=True)

    #Output new training and testing datasets
    train = pd.DataFrame(df[df['test']==False].dropna())
    train.dropna(inplace=True)
    train.drop('test', inplace=True, axis=1)
    
    test = pd.DataFrame(df[df['test']])
    test.reset_index(drop=True, inplace=True)
    test.drop('test', inplace=True, axis=1)

    if not os.path.exists('./cleaned/'):
        os.makedirs('./cleaned/')

    train.to_json('./cleaned/train.json')
    test.to_json('./cleaned/test.json')

    return cleanedDf, cleanedTyped








