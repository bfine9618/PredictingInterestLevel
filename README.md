# ESE 305 FinalProject
### Braden Fineberg, Brooke Behrbaum, Brianna Gallo
https://github.com/bfine9618/ese305FinalProject

The initial dataset can be found in ./rawData. I have also used the script 'clean data' to standardize columns, rows and names to use for the rest of the analysis. In the final model, there is a function [cleanPreprocessData()]('https://github.com/bfine9618/ese305FinalProject/blob/master/FINAL%20SCRIPT/Complete%20Script.ipynb') that does this to all train and test data. Finally, using the NLP script, I was successful in determining the type of rental from 75% of the data. After running a basic OLS regression with typed data, we are able to predict with higher accuracy what the interest level in each home is. 

#### Cleaning Data
We started by creating the "Cleaning Data". It takes in the raw data, drops unnecessary columns, cleans the description column, and outputs a cleaned data file. This allows all of our models to work off of the same data and keep our results consistent.

In a final model, our cleaning function converted an HTML string into a simpel string and ran a basic NLP. In addition, our preprocessing involved generating a range of interaction terms and grouping the data in various ways to determine expectations for each unit. We believe that the interest level in a home can be turned into a simple equation: 
$$
Net_Interest_Level = Reality-Expectation
$$

The reality consits of the actual characteristics of the unit while expectation is for a unit's given asking price, what features do you expect. The reality of the unit also includes where it is located. Specifically, if a unit is located in a high interest building, the unit is likely high interest. The expectation created a clear separation in the data. 

![alt](https://github.com/bfine9618/ese305FinalProject/blob/master/interestLevel.png)


#### Final Model



#### Future Plans
In future models, there may be a way to predict "prob_buildManager" by running a KNN on the lat long data. If we can do this with high accuracy we can improve our overall model. 

____


## Initial Exploration

#### Basic OLS
Unfortunately, the basic OLS resulting in $R^2=.07$. This told us that we needed to in some way improve our data to draw any kind of meaningful conclusions. We had several theses such as rental type (loft vs townhome vs penthouse vs ect) had different features that would make it desirable. We also thought that in each vertical, there was a price/luxury ratio. Essentially, for every additional feature, the market would be willing to pay more. Finally, using the average price/luxury in each home, we are hoping to create a price ratio, comparing the list price to the expected price given our ratio. If this price ratio is above 1, then the house is listed too high; if the price ratio is less than 1, the rental is a good deal; if it is approx. 1, then the listing is at market value. This will hopefully continue to segment our data. 

####  NLP and Improved OLS
In the file 'NLP', we use a basic NLP parser looking for key words, we are able to improve the correlation values in an OLS to upwards of .2. This indicates that our these that rental type does highly impact the interest level given features. We hope to improve our classification, and use this technique to further split trees, neural nets, and regressions. Eventually, we will use a Bayesian classifier to convert these linear values back to 'high', 'medium', and 'low' interest levels. We believe that this technique, combined with trees will allow us to better classify the edge cases. 


#### Interaction Terms
The file Brooke_Interaction_Terms explores adding interaction terms to the data and running linear regression to improve the model fit.  Interaction terms were added to combine specific related predictors, such as allowing pets in general by synthesizing allowing cats and allowing dogs.  By creating interaction terms, we hope to find an interaction term which allows us get stratification across interest level.  One interaction term we added, outdoor_score, achieved some stratification.  This term calculates an average score for the number of outdoor features a listing has.  We discovered that only high interest listings have all of these features.


#### Tree Exploration
The file Tree explores various tree models on the original dataset, starting at a standard DecisionTreeClassifier(), and then performing hyper parameter optimization, using K-Fold to perform cross validation on the tree at different depths. The results were analyzed and from there we moved on to bagging. The bagged tree performed better than the Decision Tree by almost 2%. In an effort to find a tree model that, on all the predictors, unchanged, would perform a better classification, we then moved on to Random Forests. The hyper parameters were chosen again by K-Fold cross validation; however, some difficulties were reached on having sufficient computational power to run the cross validation. After using warm_start, the number of optimal trees was determined to be roughly 170, although the cross-validation results showed slightly better accuracies for larger numbers (>300). Moving on to the max number of features to be used, another K-Fold cross validation was run, however warm_start no longer benefitted us, so the best results we were able to obtain are from running each tree only once and testing its accuracy (without using K-Fold). The results are unsatisfactory, but moving forward we will be attempting tree-based classification with better training data that will ideally speed up the hyper-parameter selection process. After the above tree based classification methods were simulated, we briefly studied the results of the most optimal (at this point in time) RandomForestClassifier, including the predicted probabilities for each class for each sample. Moving forward, we are going to study how potentially changing the threshold for classification may change our accuracy. As it stands, the tree based models have a very high accuracy for "low" interest houses at the expense of medium and high ones.
