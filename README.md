# Can we predict how a power outage occurred, right when it happened?




## Introduction
In a majority of contexts when power outages occur, utility control centers and monitoring agencies have a need to understand the root cause of an outage as it occurs in real time. Knowing whether an unexpected loss of power was caused by a weather related incident, an electricity grid malfunction, or even a malicious attack can be critical for handling the situation efficiently and reducing resources spent on diagnostics. Many modern "smart grid" systems collect vast amounts of data on electricity use and allocation, making it possible to utilize the trends in such data and allowing us to predict how disruptions may occur.


This is a project that utilizes a [publicly available dataset](https://www.sciencedirect.com/science/article/pii/S2352340918307182) containing data on major outages witnessed by different states in the United States from January 2000 to July 2016. Using a cleaned version of this dataset initially containing 1,533 observations and 55 variables, this project builds a machine learning classification model which predicts the root cause of a power outage at the time it occurs. Of these 55 variables, some of the most important to this problem are:
1. **MONTH, OUTAGE.START.TIME** – When the outage occurred 
2. **STATE, NERC.REGION** – Where the outage occurred
3. **CUSTOMERS.AFFECTED** – Customers impacted 
4. **IND.PERCEN, RES.PERCEN** – Socioeconomic data, such as industry vs. residential share of electricity usage
5. **CAUSE.CATEGORY** – Outage root cause label


The following sections catalogue this project's development process, including steps taken to explore the data as well as iteratively construct and improve predictive models.




## Data Cleaning and Exploratory Data Analysis


### Data Cleaning 
During cleaning, empty or columns which were deemed irrelevant to the problem were removed. Irrelevant columns include:
**OBS** - Observation number, instead used as an index
**HURRICANE.NAMES** - If the outage was caused by a hurricane, provide its name. This was dropped due to the sparsity of data in this column
**DEMAND.LOSS.MW** - Although intending to record the loss of electricity (in MW/hr), according to the dataset's source "in many cases, total demand is reported" instead. Coupled with a large proportion of missing values, this column does not contain enough real data to aid in predicting.


Additionally, basic type casting for numerical features was performed. After performing cleaning (as well as imputations and transformations, as described later), the first few rows of the dataset now look like:




|   YEAR |   MONTH | STATE   | NERC.REGION   | CLIMATE.REGION     |   ANOMALY.LEVEL | CLIMATE.CATEGORY   | OUTAGE.RESTORATION.DATE   | OUTAGE.RESTORATION.TIME   | CAUSE.CATEGORY     | CAUSE.CATEGORY.DETAIL   |   OUTAGE.DURATION |   CUSTOMERS.AFFECTED |   RES.PRICE |   COM.PRICE |   IND.PRICE |   TOTAL.PRICE |   RES.SALES |   COM.SALES |   IND.SALES |   TOTAL.SALES |   RES.PERCEN |   COM.PERCEN |   IND.PERCEN |   RES.CUSTOMERS |   COM.CUSTOMERS |   IND.CUSTOMERS |   TOTAL.CUSTOMERS |   RES.CUST.PCT |   COM.CUST.PCT |   IND.CUST.PCT |   PC.REALGSP.STATE |   PC.REALGSP.USA |   PC.REALGSP.REL |   PC.REALGSP.CHANGE |   UTIL.REALGSP |   TOTAL.REALGSP |   UTIL.CONTRI |   PI.UTIL.OFUSA |   POPULATION |   POPPCT_URBAN |   POPPCT_UC |   POPDEN_URBAN |   POPDEN_UC |   POPDEN_RURAL |   AREAPCT_URBAN |   AREAPCT_UC |   PCT_LAND |   PCT_WATER_TOT |   PCT_WATER_INLAND |   CUST.AFF.MISSING |   TIME.DATE.DOW |   TIME.HOUR-SIN |   TIME.HOUR-COS |
|--------|---------|---------|---------------|--------------------|-----------------|--------------------|---------------------------|---------------------------|--------------------|-------------------------|-------------------|----------------------|-------------|-------------|-------------|---------------|-------------|-------------|-------------|---------------|--------------|--------------|--------------|-----------------|-----------------|-----------------|-------------------|----------------|----------------|----------------|--------------------|------------------|------------------|---------------------|----------------|-----------------|---------------|-----------------|--------------|----------------|-------------|----------------|-------------|----------------|-----------------|--------------|------------|-----------------|--------------------|--------------------|-----------------|-----------------|-----------------|
|   2011 |       7 | MN      | MRO           | East North Central |            -0.3 | normal             | 2011-07-03 00:00:00       | 20:00:00                  | severe weather     | nan                     |              3060 |                70000 |       11.6  |        9.18 |        6.81 |          9.28 |     2332915 |     2114774 |     2113291 |       6562520 |      35.5491 |      32.225  |      32.2024 |     2.30874e+06 |          276286 |           10673 |       2.5957e+06  |        88.9448 |        10.644  |       0.411181 |              51268 |            47586 |          1.07738 |                 1.6 |           4802 |          274182 |       1.75139 |             2.2 |  5.34812e+06 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |                  0 |               4 |      0.0741085  |        0.99725  |
|   2014 |       5 | MN      | MRO           | East North Central |            -0.1 | normal             | 2014-05-11 00:00:00       | 18:39:00                  | intentional attack | vandalism               |                 1 |                63000 |       12.12 |        9.71 |        6.49 |          9.28 |     1586986 |     1807756 |     1887927 |       5284231 |      30.0325 |      34.2104 |      35.7276 |     2.34586e+06 |          284978 |            9898 |       2.64074e+06 |        88.8335 |        10.7916 |       0.37482  |              53499 |            49091 |          1.08979 |                 1.9 |           5226 |          291955 |       1.79    |             2.2 |  5.45712e+06 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |                  1 |               6 |      0.0812137  |        0.996697 |
|   2010 |      10 | MN      | MRO           | East North Central |            -1.5 | cold               | 2010-10-28 00:00:00       | 22:00:00                  | severe weather     | heavy wind              |              3000 |                70000 |       10.87 |        8.19 |        6.07 |          8.15 |     1467293 |     1801683 |     1951295 |       5222116 |      28.0977 |      34.501  |      37.366  |     2.30029e+06 |          276463 |           10150 |       2.5869e+06  |        88.9206 |        10.687  |       0.392361 |              50447 |            47287 |          1.06683 |                 2.7 |           4571 |          267895 |       1.70627 |             2.1 |  5.3109e+06  |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |                  0 |               1 |      0.0871557  |        0.996195 |
|   2012 |       6 | MN      | MRO           | East North Central |            -0.1 | normal             | 2012-06-20 00:00:00       | 23:00:00                  | severe weather     | thunderstorm            |              2550 |                68200 |       11.79 |        9.25 |        6.71 |          9.19 |     1851519 |     1941174 |     1993026 |       5787064 |      31.9941 |      33.5433 |      34.4393 |     2.31734e+06 |          278466 |           11010 |       2.60681e+06 |        88.8954 |        10.6822 |       0.422355 |              51598 |            48156 |          1.07148 |                 0.6 |           5364 |          277627 |       1.93209 |             2.2 |  5.38044e+06 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |                  0 |               1 |      0.0196337  |        0.999807 |
|   2015 |       7 | MN      | MRO           | East North Central |             1.2 | warm               | 2015-07-19 00:00:00       | 07:00:00                  | severe weather     | nan                     |              1740 |               250000 |       13.07 |       10.16 |        7.74 |         10.43 |     2028875 |     2161612 |     1777937 |       5970339 |      33.9826 |      36.2059 |      29.7795 |     2.37467e+06 |          289044 |            9812 |       2.67353e+06 |        88.8216 |        10.8113 |       0.367005 |              54431 |            49844 |          1.09203 |                 1.7 |           4873 |          292023 |       1.6687  |             2.2 |  5.48959e+06 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |                  0 |               5 |      0.00872654 |        0.999962 |




### Exploratory Data Analysis - Univariate and Bivariate
To gain a better understanding of this dataset before formalizing a prediction task, a visual analysis of various distributions and their correlations was performed to identify any patterns. At first attempts were made to identify any correlations between critical variables, however in many cases no such pattern became apparent. One such instance occurs when attempting to plot OUTAGE.DURATION against CUSTOMERS.AFFECTED, which was initially done to search for ways to capture the "severity" of an outage:


<iframe
src="assets/duration-vs-customersaffected.html"
width="600"
height="450"
frameborder="0"
></iframe>


Although relationships like these do not reveal immediate patterns, they show that the patterns in this dataset are heavily nuanced by the wide range of types of power outages included. In other words, variables such as these are indeed dependent on CAUSE.CATEGORY, as the nature of a power outage naturally determines many of its characteristics. One such example of these patterns, which will later be used for its predictive ability, focuses on a pattern emerging from the fact that many missing values in the CUSTOMERS.AFFECTED column correlate with CAUSE.CATEGORY. The following plots showcase the distribution of CAUSE.CATEGORY, before and after filtering for these missing values:


<iframe
src="assets/causecategory-dist.html"
width="600"
height="450"
frameborder="0"
></iframe>
<iframe
src="assets/causecategory-filtered.html"
width="600"
height="450"
frameborder="0"
></iframe>


Notably, intentional attacks consist of nearly 50% of observations missing a value for CUSTOMERS.AFFECTED. While it is unknown how collecting the data used in this dataset may have led to these results, the following classifier models which leverage this information assume similar circumstances exist at the time an outage occurs. If so, this information can be used by including relevant features in a model.
Furthermore, the distribution of CAUSE.CATEGORY alone showcases the uneven distribution of outage causes in this dataset. This directly implies later models must account for this fact in order to make accurate predictions.


Investigating further, aggregating by CAUSE.CATEGORY yields some interesting statistics. For example, the variations across categories show how each type of outage affects customers as shown below.


| CAUSE.CATEGORY                |   outages |   total_customers_all_time |   avg_customers |
|-------------------------------|-----------|----------------------------|-----------------|
| severe weather                |       744 |                1.37634e+08 |        181336   |
| system operability disruption |       123 |                1.97349e+07 |        156626   |
| intentional attack            |       403 |                1.20169e+07 |         28748.6 |
| equipment failure             |        55 |                4.59074e+06 |         80539.2 |
| public appeal                 |        69 |                3.32178e+06 |         48141.7 |
| fuel supply emergency         |        38 |                2.37765e+06 |         47553   |
| islanding                     |        44 |           743319           |         16159.1 |




### Imputations
Imputations were performed on the following columns:
**CUSTOMERS.AFFECTED** - report the number of customers affected
**RES.PERCEN** - Percentage of residential electricity consumption compared to the total electricity consumption in the state (in %)
**IND.PERCEN** - Percentage of industrial electricity consumption compared to the total electricity consumption in the state (in %)


CUSTOMERS.AFFECTED was imputed using a conditional median strategy, grouping on NERC.REGION (a location metric) then imputing using the median of CUSTOMERS.AFFECTED. The median was chosen over the mean due to the skewed distribution of CUSTOMERS.AFFECTED, as a number of the most extreme outages in the united states caused by hurricanes or major grid failures affected exponentially more consumers. The effects are visualized below:


<iframe
src="assets/customersaffected-pre-impute.html"
width="600"
height="450"
frameborder="0"
></iframe>
<iframe
src="assets/customersaffected-post-impute.html"
width="600"
height="450"
frameborder="0"
></iframe>


RES.PERCEN and IND.PERCEN were both imputed using a strategy that grouped STATE for a given YEAR, then imputing using the mean. For remaining observations with a unique year/state combination, groupings are only performed by state. The effects are visualized below:


<iframe
src="assets/respercen-pre-impute.html"
width="600"
height="450"
frameborder="0"
></iframe>
<iframe
src="assets/respercen-post-impute.html"
width="600"
height="450"
frameborder="0"
></iframe>
<iframe
src="assets/indpercen-pre-impute.html"
width="600"
height="450"
frameborder="0"
></iframe>
<iframe
src="assets/indpercen-post-impute.html"
width="600"
height="450"
frameborder="0"
></iframe>




## Framing and Formalizing the Prediction Problem


After performing this deep dive on the dataset and identifying some of the key patterns lying within it, we are ready to formalize this data into a prediction problem fit for a machine learning classification algorithm. To be explicit, the question the model will answer is:


**Given the immediate context of a power outage as it occured, what caused it?**


This results in the need for a multiclass classification algorithm which treats CAUSE.CATEGORY as the response variable. The metric used to evaluate models will be overall accuracy, however confusion matrices are used to evaluate recall for specific categories.


Finally, it is worth noting that the only features which are unavailable at the time of prediction are OUTAGE.DURATION and OUTAGE.RESTORATION.DATE–these have been omitted from the project to prevent data leakage. There is some ambiguity of whether CUSTOMERS.AFFECTED should be included in this group based on how this data was actually collected, however as mentioned previously this project assumes a modern monitoring system capable of implementing such a classifier also has the technology to gauge the number of affected customers, or at least produce a valid estimate. Since models will also encode the missingness of such information as its own feature, any extraordinary cases are accounted for.






## Baseline Model


A standard **Multi-class Logistic Regression** model is used to create a baseline for this project. This model will utilize the following list of features, organized by which information dimension they belong to. Features that are engineered (and not in the original dataset) are included and specified in the list.


- **Time features** 
 - `MONTH`: Month (1=Jan ... 12=Dec) of outage (ordinal)
 - `TIME.DATE.DOW` (**Engineered**): Day of week (0=Mon ... 6=Sun) (ordinal)
 - `TIME.HOUR‑SIN`, `TIME.HOUR‑COS`  (**Engineered**): Sine/cosine transforms of outage start time (hour + minute), to capture cyclical daily patterns (quantitative)


- **Location features** 
 - `STATE`: U.S. postal code (nominal) 
 - `NERC.REGION`: North American Electric Reliability region (nominal) 


- **Climate feature** 
 - `ANOMALY.LEVEL`: The oceanic El Niño/La Niña (ONI) index referring to the cold and warm episodes by season (quantitative) 


- **Customer features** 
 - `CUSTOMERS.AFFECTED`: Number of customers impacted (quantitative) 
 - `CUST.AFF.MISSING` (**Engineered**): Indicator (0/1) for originally missing CUSTOMERS.AFFECTED counts (quantitative - binary) 
 - `POPPCT_URBAN`: Percent urban population in that state (quantitative) 


- **Economic features** 
 - `RES.PERCEN`: Residential electricity consumption percentage of total load (quantitative) 
 - `IND.PERCEN`: Industrial electricity consumption percentage of total load (quantitative) 
 - `UTIL.CONTRI`: Utility industry's contribution to the total GSP in the State (as %) (quantitative) 
 - `TOTAL.REALGSP`: Real GSP contributed by all industries in a given state (measured in 2009 U.S dollars) (quantitative)


Of these 14 total features, 10 are quantitative/numeric, 2 are ordinal, and 2 are nominal/categorical.


The following schema was followed to produce the model:
- **Pipeline**: 
 1. One‑hot encode nominal/ordinal features 
 2. Standardize all numeric features 
 3. Fit a `LogisticRegression` (with l1 regularization)  
- **Hyperparameter search**: 20 values of inverse‑regularization `C` (log‑spaced 1e‑4 to 1e2) 
- **Evaluation**: 80% train / 20% test stratified split 




### How did it do?
This model yields 76.72% training accuracy and 70.82% test accuracy. Overall, this a promising benchmark to set for future revisions. These results indicate the presence of a strong, real pattern in the data, however there are still several issues and opportunities for improvement to address in future models. While this first baseline is good in that it has promising accuracy, the goal now is to produce a model that is less overfit (as indicated by the difference in train/test scores) and which has better performance.






## Final Model


To improve upon the baseline model several key strategies are employed. These aim to address the following potential issues with the baseline model:
- The issue of an error occuring during training where rare categorical features are seen in validation sets but not test sets (more on this below)
- Feature selection: Perhaps 14 features was too many, the impact of each must be evaluated
- Feature engineering: Raw performance as well as ability to generalize can be improved with new or modified features
- Model selection: While a logistic regression was used initially, other models may show more promising results




### The issue with the 'public appeal' category...
Surveying the results of the baseline model's performance across each CAUSE.CATEGORY, a problem became apparent relating to the ability to predict 'public appeal.' The model suffers from low recall (~16%) and only moderate precision (~50%), showcasing an apparent struggle in predicting public appeal.
Upon further research into the dataset itself, the reason becomes apparent. According to this dataset (and the NERC), most public appeal events are instances where officials urge electricity consumers to lower their usage for some reason (like a storm). Such events are planned, and oftentimes do not even represent a loss of power. Beyond intuition, These results prove 'public appeal' events are not fit for this problem–the model struggles as it searches for patterns that aren't there.


As a result, the first major change is to remove 'public appeal' observations from the dataset (less than 5% of the overall data)


### Feature Selection
To examine which features are (un)important to the model, the absolute value of the mean of each coefficient across all classes is used to produce a ranking of feature importance. However, this process only resulted in one feature which can surely be removed: RES.PERCEN, which averaged 0.0 importance across all categories indicating it did not help predict any category. Following these results, RES.PERCEN will be dropped from the final model.


### Feature Engineering
When training the baseline model via cross-validation, a particular error occurred where rare categorical features were encountered in validation sets but not training sets. An exact example output is as follows:

`UserWarning: Found unknown categories in columns [...] during transform. These unknown categories will be encoded as all zeros`.

Such a warning indicates that in order to negate this information loss and produce better results, rare categories should be grouped into an "Other" category to prevent errors. This is confirmed by surveying the distribution of the STATE and NERC.REGION variables:


<iframe
src="assets/state-dist.html"
width="600"
height="450"
frameborder="0"
></iframe>
<iframe
src="assets/nercregion-dist.html"
width="600"
height="450"
frameborder="0"
></iframe>


To resolve these, a cutoff for state was set which binned states which represent less than .5% of all data into an "Other" category, and a cutoff after ECAR region was set for NERC.REGION data. Retraining the baseline model with these new binned features, 'STATE-T' and 'NERC.REGION-T', does not report any warnings. Additionally, binning these categories as new features may **reduce overfitting**–when surveying feature importance, the values of certain coefficients relating to rare states/regions imply the model may have "memorized" these categories and their CAUSE.CATEGORY in the training set.


Finally, the same logic is applied to CAUSE.CATEGORY, the response variable. CAUSE.CATEGORY-T maps 'equipment failure', 'fuel supply emergency', and 'islanding' all as 'physical failures', representing a broad hardware malfunction leading to an outage. This is done as these categories are similar in nature in reality, which is hinted at in the baseline model's tendency to misclassify each of these for each other.
However, binning the response to create less categories overall will almost certainly improve raw accuracy–this is more done as an exercise in curiosity. To examine whether the gains in accuracy surpass the loss of information, all candidate final models are evaluated on **both** CAUSE.CATEGORY and CAUSE.CATEGORY-T separately.




### Model Selection


To optimize performance, several other classification algorithms were tested as candidates for the final model. The other algorithms tested were Random Forest classification and a multi-layer perceptron (MLP), as both algorithms are powerful classifiers which have potential to yield promising results. A Decision Tree model was also included in this set, however this model was only included to visualize a decision tree in the case Random Forest proved best.


With 4 algorithms and 2 variations of a response variable, CAUSE.CATEGORY and CAUSE.CATEGORY-T, 8 total models were created and evaluated. For each model, all hyperparameters were tuned across a vast set of combinations utilizing sklearn GridSearchCV. The following graph outlines the results:


<iframe
src="assets/final-models-results.html"
width="600"
height="450"
frameborder="0"
></iframe>


The resulting final model chosen from this set is the multi-class **Logistic Regression**, which classifies on the non-engineered CAUSE.CATEGORY. This model yields a 79.20% training score, and a 78.08% test score. While the logistic regression using 'physical failures' did perform strictly better, overall a ~1.5% increase in test accuracy does not justify losing specificity in predictions. Within the context of this project, a power outage diagnostician is likely more interested in the specifics of a physical failure than marginal improvements in accuracy.


The final model has the following hyperparameters, and was trained by one hot encoding all categorical features and standardizing all numerical features:


| Parameter      | Value                   | Description                                        |
| -------------- | ----------------------- | -------------------------------------------------- |
| `C`            | `78.80462815669921`     | Inverse of regularization strength (L2)            |
| `penalty`      | `"l2"`                  | Regularization norm                                |
| `solver`       | `"liblinear"`           | Optimization algorithm (supports L1 & L2)          |
| `max_iter`     | `10000`                 | Maximum iterations for solver convergence          |


Features used:


- **Time features** 
 - `MONTH`: Month (1=Jan ... 12=Dec) of outage (ordinal)
 - `TIME.DATE.DOW` (**Engineered**): Day of week (0=Mon ... 6=Sun) (ordinal)
 - `TIME.HOUR‑SIN`, `TIME.HOUR‑COS`  (**Engineered**): Sine/cosine transforms of outage start time (hour + minute), to capture cyclical daily patterns (quantitative)


- **Location features** 
 - `STATE-T` (**Engineered**): U.S. postal code (nominal) 
 - `NERC.REGION-T` (**Engineered**): North American Electric Reliability region (nominal) 


- **Climate feature** 
 - `ANOMALY.LEVEL`: The oceanic El Niño/La Niña (ONI) index referring to the cold and warm episodes by season (quantitative) 


- **Customer features** 
 - `CUSTOMERS.AFFECTED`: Number of customers impacted (quantitative) 
 - `CUST.AFF.MISSING` (**Engineered**): Indicator (0/1) for originally missing CUSTOMERS.AFFECTED counts (quantitative - binary) 
 - `POPPCT_URBAN`: Percent urban population in that state (quantitative) 


- **Economic features** 
 - `IND.PERCEN`: Industrial electricity consumption percentage of total load (quantitative) 
 - `UTIL.CONTRI`: Utility industry's contribution to the total GSP in the State (as %) (quantitative) 
 - `TOTAL.REALGSP`: Real GSP contributed by all industries in a given state (measured in 2009 U.S dollars) (quantitative)


Overall, this model has improved significantly over the baseline. While the choice of algorithm remains a multi-class logistic regression, this new model has both strictly better performance while also being far less overfit than the baseline.






## Conclusion and Next Steps


In conclusion, the final model's test accuracy of 78.08% demonstrates that there is real predictive ability in determining the cause of power outages. To improve further, improvements can be made moving forward to further increase performance that were not made throughout this project. For instance, many incorrect predictions are caused by the uneven distribution of CAUSE.CATEGORY–the rare categories lose precision/recall performance due to the model's frequent biasing towards the more common groups. While creating a logistic regression that balances the weight of each class does not resolve this issue, more advanced methods can be utilized in the future.

