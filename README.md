# Can we predict how a power outage occured, right when it happened? 


## Introduction
In a majority of contexts when power outages occur, utility control centers and monioring agencies have a need to understand the root cause of an outage as it occurs in real time. Knowing whether an unexpected loss of power was caused by a weather related inceident, an electricity grid malfunction, or even a malicious attack can be critical for handling the situation efficiently and possibly reduce resources spent on diagnostics. Many modern "smart grid" systems collect vast amounts of data on electricity use and allocation. It is possible to utilize the trends in such data, allowing us to predict how disruptions may occur.

This is a project that utilizes a [publically available dataset](https://www.sciencedirect.com/science/article/pii/S2352340918307182) containing data on major outages witnessed by different states in the United States from January 2000 to July 2016. Using a cleaned version of this dataset initially containing 1,533 observations and 55 variables, this project builds a machine learning classification model which predicts the root cause of a power outage at the time it occurs. Of these 55 variables, some of the most important to this problem are: 
1. **MONTH, OUTAGE.START.TIME** – When the outage occurred  
2. **STATE, NERC.REGION** – Where the outage occured
3. **CUSTOMERS.AFFECTED** – Customers impacted  
4. **IND.PERCEN, RES.PERCEN** – Socioeconomic data, such as industry vs. residential share of electricity usage
5. **CAUSE.CATEGORY** – Outage root cause label

The following sections catalogue this project's development process, including steps taken to explore the data as well as iteravely construct and improve predictive models.


## Data Cleaning and Exploratory Data Analysis

### Data Cleaning  
During cleaning, empty or columns which were deemed irrelevant to the problem were removed. "Irrelevant columns" include: 
**OBS** - Observation number, instead used as an index
**HURRICANE.NAMES** - If the outage was a hurricane, provide it's name. This was dropped due to the sparsisty of data in this column
**DEMAND.LOSS.MW** - Although intendinng to record the loss of electicity (in MW/hr), according to the dataset's source "in many cases, total demand is reported" instead. Coupled with a large proprotion of missing values, this column does not contain enough real data to aid in predicting. 

Additionally, basic type casting for numerical features was performed. After performing cleaning (as well as imputations and transformations, as described later), the first few rows of the dataset now look like: 


|   YEAR |   MONTH | STATE   | NERC.REGION   | CLIMATE.REGION     |   ANOMALY.LEVEL | CLIMATE.CATEGORY   | OUTAGE.RESTORATION.DATE   | OUTAGE.RESTORATION.TIME   | CAUSE.CATEGORY     | CAUSE.CATEGORY.DETAIL   |   OUTAGE.DURATION |   CUSTOMERS.AFFECTED |   RES.PRICE |   COM.PRICE |   IND.PRICE |   TOTAL.PRICE |   RES.SALES |   COM.SALES |   IND.SALES |   TOTAL.SALES |   RES.PERCEN |   COM.PERCEN |   IND.PERCEN |   RES.CUSTOMERS |   COM.CUSTOMERS |   IND.CUSTOMERS |   TOTAL.CUSTOMERS |   RES.CUST.PCT |   COM.CUST.PCT |   IND.CUST.PCT |   PC.REALGSP.STATE |   PC.REALGSP.USA |   PC.REALGSP.REL |   PC.REALGSP.CHANGE |   UTIL.REALGSP |   TOTAL.REALGSP |   UTIL.CONTRI |   PI.UTIL.OFUSA |   POPULATION |   POPPCT_URBAN |   POPPCT_UC |   POPDEN_URBAN |   POPDEN_UC |   POPDEN_RURAL |   AREAPCT_URBAN |   AREAPCT_UC |   PCT_LAND |   PCT_WATER_TOT |   PCT_WATER_INLAND |   CUST.AFF.MISSING |   TIME.DATE.DOW |   TIME.HOUR-SIN |   TIME.HOUR-COS |
|--------|---------|---------|---------------|--------------------|-----------------|--------------------|---------------------------|---------------------------|--------------------|-------------------------|-------------------|----------------------|-------------|-------------|-------------|---------------|-------------|-------------|-------------|---------------|--------------|--------------|--------------|-----------------|-----------------|-----------------|-------------------|----------------|----------------|----------------|--------------------|------------------|------------------|---------------------|----------------|-----------------|---------------|-----------------|--------------|----------------|-------------|----------------|-------------|----------------|-----------------|--------------|------------|-----------------|--------------------|--------------------|-----------------|-----------------|-----------------|
|   2011 |       7 | MN      | MRO           | East North Central |            -0.3 | normal             | 2011-07-03 00:00:00       | 20:00:00                  | severe weather     | nan                     |              3060 |                70000 |       11.6  |        9.18 |        6.81 |          9.28 |     2332915 |     2114774 |     2113291 |       6562520 |      35.5491 |      32.225  |      32.2024 |     2.30874e+06 |          276286 |           10673 |       2.5957e+06  |        88.9448 |        10.644  |       0.411181 |              51268 |            47586 |          1.07738 |                 1.6 |           4802 |          274182 |       1.75139 |             2.2 |  5.34812e+06 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |                  0 |               4 |      0.0741085  |        0.99725  |
|   2014 |       5 | MN      | MRO           | East North Central |            -0.1 | normal             | 2014-05-11 00:00:00       | 18:39:00                  | intentional attack | vandalism               |                 1 |                63000 |       12.12 |        9.71 |        6.49 |          9.28 |     1586986 |     1807756 |     1887927 |       5284231 |      30.0325 |      34.2104 |      35.7276 |     2.34586e+06 |          284978 |            9898 |       2.64074e+06 |        88.8335 |        10.7916 |       0.37482  |              53499 |            49091 |          1.08979 |                 1.9 |           5226 |          291955 |       1.79    |             2.2 |  5.45712e+06 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |                  1 |               6 |      0.0812137  |        0.996697 |
|   2010 |      10 | MN      | MRO           | East North Central |            -1.5 | cold               | 2010-10-28 00:00:00       | 22:00:00                  | severe weather     | heavy wind              |              3000 |                70000 |       10.87 |        8.19 |        6.07 |          8.15 |     1467293 |     1801683 |     1951295 |       5222116 |      28.0977 |      34.501  |      37.366  |     2.30029e+06 |          276463 |           10150 |       2.5869e+06  |        88.9206 |        10.687  |       0.392361 |              50447 |            47287 |          1.06683 |                 2.7 |           4571 |          267895 |       1.70627 |             2.1 |  5.3109e+06  |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |                  0 |               1 |      0.0871557  |        0.996195 |
|   2012 |       6 | MN      | MRO           | East North Central |            -0.1 | normal             | 2012-06-20 00:00:00       | 23:00:00                  | severe weather     | thunderstorm            |              2550 |                68200 |       11.79 |        9.25 |        6.71 |          9.19 |     1851519 |     1941174 |     1993026 |       5787064 |      31.9941 |      33.5433 |      34.4393 |     2.31734e+06 |          278466 |           11010 |       2.60681e+06 |        88.8954 |        10.6822 |       0.422355 |              51598 |            48156 |          1.07148 |                 0.6 |           5364 |          277627 |       1.93209 |             2.2 |  5.38044e+06 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |                  0 |               1 |      0.0196337  |        0.999807 |
|   2015 |       7 | MN      | MRO           | East North Central |             1.2 | warm               | 2015-07-19 00:00:00       | 07:00:00                  | severe weather     | nan                     |              1740 |               250000 |       13.07 |       10.16 |        7.74 |         10.43 |     2028875 |     2161612 |     1777937 |       5970339 |      33.9826 |      36.2059 |      29.7795 |     2.37467e+06 |          289044 |            9812 |       2.67353e+06 |        88.8216 |        10.8113 |       0.367005 |              54431 |            49844 |          1.09203 |                 1.7 |           4873 |          292023 |       1.6687  |             2.2 |  5.48959e+06 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 |                  0 |               5 |      0.00872654 |        0.999962 |


### Exploratory Data Analysis - Univariate and Bivariate 
To gain a better understanding of this dataset before formalizing a prediction task, a visual analysis of various distrobutions and their correlations was performed to identify any patterns. At first attempts were made to identify any correlations between  cirtical variables, however in many cases no such pattern became appararent. One such instance occurs when attempting to plot OUTAGE.DURATION against CUSTOMERS.AFFECTED, which was initially done to search for ways to capture the "severity" of an outage: 

<iframe
 src="assets/duration-vs-customersaffected.html"
 width="600"
 height="450"
 frameborder="0"
 ></iframe>

Although relationships like these do not reveal immediate patterns, they show that the patterns in this dataset are heavily nuanced by the wide range of types of power outages included in this dataset. In other words, variables such as these are indeed dependent on CAUSE.CATEGORY, as the nature of a power outage naturally determines many of its characteristics. One such example of these patterns, which will later be used for it's predictive ability, focuses on a pattern emerging from the fact that many missing values in the CUSTOMERS.AFFECTED column correlate with CAUSE.CATEGORY. The following plots showcase the distribution of CAUSE.CATEGORY, before and after filtering for these missing values:

<iframe
 src="assets/causecategory-filtered.html"
 width="600"
 height="450"
 frameborder="0"
 ></iframe>
<iframe
 src="assets/causecategory-dist.html"
 width="600"
 height="450"
 frameborder="0"
 ></iframe>

Notably, intentional attacks consist of 50% of observations missing a value for CUSTOMERS.AFFECTED. While it is uknown how collecting the data used in this dataset may have led to these results, the following classifier models which leverage this information assume similar circumstances exist at the time an outage occurs. If so, this information can be used by including relevant features in a model.
Furthermore, the distribution of CAUSE.CATEGORY alone showcasses the uneven distrobution of outages causes in this dataset. This direclty implies later models must account for this fact in order to make accurate predictions.

Investigating further, aggregating by CAUSE.CATEGORY yields some interesting statistics. For example, the variations across cateogories show how each type of outage affects customers as shown below.

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

CUSTOMERS.AFFECTED was imputed using a conditional median stretegy, grouping on NERC.REGION (a location metric) then imputing using the median of CUSTOMERS.AFFECTED. The median was chosen over the mean due to the skewed distrobution of CUSTOMERS.AFFECTED, as a number of the most extreme outages in the united states caused by hurricanes or major grid failures affected exponentially more consumers. 

RES.PERCEN and IND.PERCEN were both imputed using a strategy that grouped STATE for a given YEAR, then imputing using the mean. For remaining observations with a unique year/state combination, groupings are only performed by state. 