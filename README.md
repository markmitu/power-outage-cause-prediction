# Can we predict how a power outage occured, right when it happens? 

## Introduction
In a majority of contexts when power outages occur, utility control centers and monioring agencies have a need to understand the root cause of an outage as it occurs in real time. Knowing whether an unexpected loss of power was caused by a weather related inceident, an electricity grid malfunction, or even a malicious attack can be critical for handling the situation efficiently and possibly reduce resources spent on diagnostics. Many modern "smart grid" systems collect vast amounts of data on electricity use and allocation. It is possible to utilize the trends in such data, allowing us to predict how disruptions may occur.

This is a project that utilizes a [publically available dataset](https://www.sciencedirect.com/science/article/pii/S2352340918307182) containing data on major outages witnessed by different states in the United States from January 2000 to July 2016. Using a cleaned version of this dataset iniitally containing 1,533 observations and 55 variables, this project builds a machine learning classification model which predicts the root cause of a power outage at the time it occurs. Of these 55 variables, some of the most important to this problem are: 
1. **MONTH, OUTAGE.START.TIME** – When the outage occurred  
2. **STATE, NERC.REGION** – Where the outage occured
3. **CUSTOMERS.AFFECTED** – Customers impacted  
4. **IND.PERCEN, RES.PERCEN** – Seocioeconomic data, such as Industry vs. residential share of electricity usage
5. **CAUSE.CATEGORY** – Outage root cause label