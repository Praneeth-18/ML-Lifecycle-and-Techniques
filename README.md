# ML-Lifecycle-and-Techniques

Business Objective: What properties should an investor buy such that the HOA (Home Owner Association) fees  + Mortgage Fees < Rent so the rental property investment is generating the maximum return on investment (Now with the rental. and in the future with the prediction of house prices). Suggest the least, most, and most desirable properties that will fulfill their investment objectives.

### Business Case and Value:

The core hypothesis is that the properties with a combination of low HOA (Home Owner Association) fees, manageable mortgage payments, and high rental income potential will offer the best investment returns.

The primary goal here is to identify properties that will lead to the maximum return on investment (ROI) by ensuring that the sum of Home Owner Association (HOA) fees and mortgage fees is less than the rental income. This strategy should cover not only current rental income but also consider the potential for future property price appreciation. 

Data Narrative: To guide an investor, retiree, or any individual interested in property investment, we'll analyze the dataset to identify properties that align with our investment criteria. 

#### This involves:

Identifying properties with HOA + mortgage fees < rental income.

Assessing factors that could influence future property values.

Categorizing properties into 'least', 'more', and 'most' desirable based on their investment potential.

#### Process:

To achieve our objective we will follow the following steps:

Perform EDA and Data Cleaning on the on the main dataset

Scrape or calculate the values of HOA fee and Morage fee

Perform Fractal clustering based on the business objective function to find golden cluster and use the insights from fractal clustering for assigning label for buying recommendation

Find latent variables and datasets that could support them

Scarpe or calculate what the value of a property could be in different time intervals

Use the above data for building a regression model that predicst the cost of a property in future

If the regression model has good performance, use it’s prediction as one of the features when building the classification model for classifying properties to buy.

#### Columns and their meanings:

rank: A ranking of properties, likely based on relevance or desirability.

property_id: Unique identifier for each property.

address, latitude, and longitude: Location details of the property.

price: The sale price of the property in USD.

bathrooms and bedrooms: Number of bathrooms and bedrooms.

area: The living area size is in square feet.

land_area: The land area size (only 49 entries have this data). (Area of the plot)

zestimate: Zillow’s estimated market value for the property.

rent_zestimate: Zillow’s estimated rent price.

days_on_zillow: Number of days the property has been listed on Zillow.

Other details such as listing_type, status_text, broker_name, and links to the property on Zillow.

Information about the data gotten from EDA

The shape of the data is  1755x23, with missing values present in different quantities throughout the data.

#### Insights:

The dataset only has listings from California US, with the majority of the listings being from the 4 following cities San Diego,  San Jose, Irvine, and Carlsbad

Houses can be of multiple types such as townhouse, family house etc. Our dataset has listing for just plots of land as as well however as the amount of data is very less we will discard them will only focus on different types of houses

There were way to man missing values , to do any firther mre analysis we would first have to clean the data. 

## Data Cleaning

List of steps we will be performing for preparing our data:

Drop the sold date as it is none for all

Address is made up of multiple parts eg: 3262 Loma Riviera Dr, San Diego, CA 92110. From it, we will extract all individual columns i.e. Street, City, state, and ZIP (done)

Check if all Prices are in USD -if they are we will drop the column, and if they are not we will convert everything to USD. Drop the currency column
Converting Land Area and area to Integer (as of now it is being represented as vali sqft or acre). We will convert the acre values to sqft.  

Dropping Columns that have only a single value

Removing Columns which we know we don't need for training or exploring the data

Handling Missing Values in:

Zestimate - wherever it is missing just replace it with the price

Area - find the average price for each house type and use it to fill the area. (doing so we found that  one of the house types is a plot of land, our objective is to look for a house so we delete all records where the house type is a lot)

Bathroom and bedroom - we find the average value based on group by house type and zip and use that to fill

Rent_ezestimate: https://www.nasdaq.com/articles/determining-how-much-you-should-charge-for-rent#:~:text=The%20amount%20of%20rent%20you,%242%2C000%20and%20%242%2C750%20each%20month. As per the article the rent is roughly 1.1% of the house’s value.

We will use this logic to fill in the missing values

Fill in the missing values for lat and longitude - fill with the average value for the zip (most likely we won't be using it for training)

Making sure all variables are in the correct data type format

Generate the HOA fee and mortgage_fee

We spent considerable time trying to scrape these values, however due to zillow’s bot protection we weren’t able to scrape the values for more than 50 rows.
Calculating HOA value: as per https://www.doorloop.com/blog/hoa-statistics, 37% of the population in California lives under HOA and the average HOA value is $387 per month. So we use this to assign random values of HOA to 37% of the out data (all the random values combined follow a normal distribution with mean around 387), the remaining 63% of the data was given a value of 0.

Calculating Mortgage fee values: As per https://www.moneygeek.com/mortgage/how-to-buy-a-house-in-california/, 20% of the house price should be downpayment. We take tenure of 30 years and a fixed rate of interest of 6.621% (the reason for using this is these are the values Zillow uses for its calculations). We use these values too calculate the monthly mortgage fee.


## Fractal Clustering

Objective → Our objective is to Maximize return and Minimize investment. 

where, 
		return = Rent - Mortgage - HOA  
		investment = Price

Ratio for evaluation is the investment_ratio = (return/investment). The more this ratio, the more desirable.

To Find the clusters that are most, more and least desirable. The Golden Cluster is the most desirable one. These are also the entries that are Labeled as the ‘Most Desirable’ category for Classification Task. The Clusters giving negative return will be tagged as the ‘Least Desirable’ Category (Because no one wants a negative return on investment). The clusters that are left would have positive return but not as big as the golden cluster and thus will be tagged as More Desirable (in comparison to least desirable) as there is some positive return.

### Iteration 1:

The Elbow method is used to find the optimal number of clusters. 

KMeans clustering is initially performed with 3 clusters.

An 'investment_ratio' is calculated, and the average value for each cluster is determined. Cluster 1 is found to have the highest average investment ratio of 0.0076 ~ 0.76 percentage monthly but only contains 4 records, suggesting these are exceptionally profitable properties.

On the contrary, Cluster 2 is found to have negative average returns with 39 entries.

Cluster 0 has the most data i.e. 1589 records with evaluation ratio 0.0005 ~ 0.05 percentage monthly returns.

The silhouette score is high (0.853), indicating that the clusters are clearly distinguished and are well apart from each other.

But as there are only 4 records we can’t go deeper into Cluster 1.

So, we’ll rather perform further clustering on Cluster 0 as it has the most scope to find some good cluster

## Iteration 2:

Further analyzing the largest cluster (cluster 0 from iteration 1), again using the Elbow method and KMeans to find subclusters.

Three subclusters are identified.

Here,  subcluster 2 shows a high average investment ratio of 0.0075 but contains only 38 records and are considered as most desirable. 

Subcluster 1 is showing negative average returns. 

Subcluster 0 again has the highest amount of data (1291 records) and about the same (0.0005 evaluation ratio) 

The silhouette score for these subclusters is lower (0.641) than in Iteration 1, indicating less distinction between them.

Based on the fact that we still have only 38 records we will rather dig deeper into subcluster 0 which has the most records and slight positive returns.

## Iteration 3:

Further analyzing the subcluster 0 from Iteration 2 and repeating the clustering process.

Three new subclusters are formed, with subcluster 2 having the highest investment ratio of
0.007 ~ 0.7 percentage monthly returns and a total 36 records.

Subcluster 0 shows negative returns and sub cluster 1 has 0.0007 ~ 0.07 percentage monthly return with 785 entries.

A lower silhouette score (0.522) suggests less clarity in the separation between these subclusters.

We went for another iteration and the best evaluation ratio we could find was 0.01% which is a drastic drop so we discarded that iteration.
We stop here on iteration 3.

#### Conclusion and Inferences:

The “Golden Cluster” comprises the most profitable investments with a return ratio of 0.7% or higher monthly return on investment which is phenomenal as it would mean ~ 8.5+ percentage yearly returns. Also, all the properties that fall into the golden cluster are considered to be in the most desirable category and these entries are ideal candidates for the classification task.

More desirable investments, while not as profitable as the golden cluster, offer low returns but are preferable in comparison to the least desirable ones.
The least desirable investments are properties that are prone to giving negative returns, identified in the lowest-performing clusters from each iteration.
Below is the divide and conquer flow and desirable categories highlighted:



Note: For classification task, labeled data is formed by:

Most Desirable: Combining Cluster 1 of overall data, subcluster2 of cluster_2_data, subcluster2 of cluster_3_data

Least Desirable: Combining Cluster 2 of overall data, subcluster1 of cluster_2_data, subcluster0 of cluster_3_data

More Desirable: subcluster1 of cluster_3_data

## Regression

Objective → To predict the price of property in the next 1, 2 and 5 years.

We do this by using the average house price change information based on county in a span of the last 1, 2 and 5 years.

Now to determine the multiplying factor for the prices after 1, 2 and 5 years, we are assuming that the trend from the past 1 year will be followed the next 1 year. Similarly the trend from the past 2 years will be followed in the upcoming 2 years and same goes for 5 year change.

Now as the trend is going to be identical the price change in terms of percentage will also be identical. To calculate the percentage change, we modified a dataset from zillow and found the change of average house pricing per county.

Example of calculation of SanDiego_1 (multiplying factor for 1 year for San Diego) would be:

SanDiego_1 = Average Price in San Diego on Feb 29, 2024 / Average Price on Feb 28, 2023 in San Diego

Define the multiplying factors for each county and apply the factors to the relevant columns.

Define functions and objects needed for Muller's Loop and Evaluation Plots.

The list of all regressors we have used is MLP Regressor, Linear Regression, Random Forest Regressor, KNN Regressor, XGBoost Regressor, Extra Trees Regressor, Stochastic Gradient Descent Regression.

##### Making A Regression Model to forecast price for 2025 i.e. Next Year.

The factors we applied suggest an appreciation in property values across the counties we have data for.

The residuals plot indicates a pattern where residuals increase as the predicted values increase, suggesting heteroscedasticity. The presence of heteroscedasticity means that the model's performance varies at different levels of the dependent variable, and it may not be predicted as accurately for higher-value properties.

Despite this pattern, the lowess line is relatively flat, indicating that the model is consistent across different predicted values but with increasing variance.

The scatter plot of actual vs. predicted values shows a strong linear relationship, suggesting the model can predict property values fairly well.

The points generally follow the diagonal line, which represents perfect predictions. Deviations from this line are errors in prediction.

There are some outliers, particularly at the higher end of property values, indicating that the model may struggle with very high-value properties.

##### Making A Regression Model to forecast price for 2026 (2 Years in the Future)

The residuals are fairly randomly distributed, which is a good sign that the model doesn't suffer from heteroscedasticity.

Most residuals are clustered near the zero line, indicating good performance for lower-priced properties.

However, there is a pattern where higher-priced properties have higher residuals, suggesting that the model is less accurate for higher-valued homes.

The predicted values are closely aligned with the actual values, especially for lower and medium-range properties, as shown by the proximity to the dashed line representing perfect predictions.

The model appears to underestimate the value of the highest-priced homes, as shown by the points deviating from the line in the upper right.

The model seems to perform well, with a trend line that matches the diagonal (indicating accurate predictions) for a significant part of the data.

##### Making A Regression Model to forecast price for 2029 (5 Years in the Future)

Similar to previous models, the residuals are mostly centered around the horizontal zero line, which indicates a good fit for a significant portion of the data.

There is an increased spread in residuals as the predicted values get larger, suggesting the model is less accurate for predicting higher-priced homes.

The data points are aligned along the diagonal line, showing a good match between the actual and predicted values, particularly for properties valued at the lower end of the price range.

#### Let’s Perform the Same Loop with Latent Variable - Crime Rates.

We do all the steps that we have done above here again with additional latent variables violent_crimes	and property_crimes.

##### Making A Regression Model to forecast price for 2025 with Latent Variables.

The residuals plot shows some pattern - as predicted values increase, the residuals tend to be more spread out, suggesting that the model may have more difficulty predicting higher value homes accurately.

The actual vs. predicted values plot indicates a strong linear relationship for lower-priced homes, but deviations are noticeable at higher price points, supporting the residuals plot observations.

The inclusion of crime rates is based on the hypothesis that crime rates have an impact on property values, which is a realistic consideration in real estate valuation.

##### Making A Regression Model to forecast price for 2026 with Latent Variables.

violent_crimes and property_crimes have negative coefficients, even though they are much smaller in magnitude compared to other features. This suggests a slight decrease in property value with an increase in crime rates.

The residuals plot displays a pattern where the residuals increase with the predicted values.

Most predicted values are closely aligned with the actual values for lower and mid-range properties, as shown by the points following closely to the dashed diagonal line.

Similar to the residuals plot, the model seems to underestimate the value of the highest-priced homes. This is shown by the actual values deviating above the dashed line in the upper region of the plot.

##### Making A Regression Model to forecast price for 2029 with Latent Variables.

The residuals plot displays a noticeable trend, which suggests that the model’s predictions are systematically off for certain values.

The actual vs. predicted plot shows a close alignment along the dashed line, which represents perfect predictions, especially for properties with lower zestimate values.

#### Feature Importance (Linear Regression Model):

The bar chart of feature importance from the Linear Regression Model shows that house type and location (latitude and longitude) are among the most influential factors affecting property value predictions. 

Interestingly, violent crimes and property crimes, which are the latent variables added to the dataset, show relatively low importance according to the linear regression model. 

This might imply that the impact of crime rates on property values is not as significant as other variables or is possibly being captured by other variables in the dataset.

#### Feature Importance (Extra Trees Model):

The Extra Trees model gives the most importance to the mortgage fee, followed by area square footage and rent estimate, which differs from the linear regression model's assessment. 

This variation highlights that different models can capture the importance of features in different ways.

#### Conclusion:
Adding latent variables to the model has changed the dynamics of prediction, although these new variables do not seem to be the strongest predictors compared to traditional real estate features.

## Classification

Objective → To classify which properties are worth investment.

We will use the data labeled by fractal clustering to perform the task at hand.
We are aware that there is a class imbalance , however as the imbalance for the best possible investment property the risk of misclassifying something else as the best property becomes too high, so to overcome we are first processing without imbalance and if during the experimentation we feel that the performance is too low we would explore Upsmalping techniques.

Model evaluation Criteria:

We will be using the f1-score and precision for class 2, as our primary metric for evaluation along with the confusion matrix. The reason for picking up these metrics is that we want to be sure that we pick a model which doesn't have a large false positive when it comes to class 2 prediction.
Miss-classifying class 0 and 1 is much more forgivable , when we decide on what to act in terms of buying the property.

### Iterations:

Step 1: Experimenting with Base Data

Using just the base data without any latent variables for building the model.
After training on the base data we found that the best model for our use case is a DecisionTree, which has an F1 score of 83.8% and precision of 86.67% however from the decision matrix we can see that hgh number of class 0 elements are being classified as class 2. This is the behavior we are trying to avoid.

Our 5 most promising feature as per SHAP values are ['rent_zestimate' 'price' 'area_sqft' 'zestimate' 'morgage_fee']

Step 2: Experimenting with First Latent Variable and Amalgamation

One of the factors which would affect a house purchasing decision would be the crime rate in the neighborhood of the house. To capture this we join our data with crime rates data found in a dataset resent on https://www.ppic.org/data-set/crime-rates-in-california/ i.e property crimes
We use the county and zip mapping to join the 2 data sets.
From the crimes dataset would be first looking at the total property per zip crimes feature this includes crimes such as break ins, burglary, vehicle theft and larceny theft.

After running the muller's loop on the first amalgamation where we include the number of property crimes in the area of the house, the best performing model was once again DecisionTree model with a precision of 100% and F1 score of 96.7. When you look at the confusion matrix you can see we greatly reduce the false positives for class 2. This is the behavior which we are looking for. However one small problem with this model is that I miss-classify a chunk of class 2 properties as class 0 . This is better than misclassifying class 0 as class 2. One of the possible reasons is that from the model's shaps values we can see that class 0 and class 2 have the same top 5 important features.

Note: During the mid-sem as work was going on in parallel, one person was working on cleaning the data and another person was working on finding the latent variables and joining it together. 
Link for latent variable augmentation : Link for latent variable augmentation 

Step 3: Experimenting with Second Latent Variable

We use the same logic we used for latent feature one, however this time instead of looking at property crimes we will be looking at property and violent crimes in an area. We used the violent crimes data variable on https://www.ppic.org/data-set/crime-rates-in-california/

After fitting the model on both the latent variables and amalgamations we, we find that the best performing model is a Decision tree with precision of 1 and f1 of 97, this is same as when we used the property crime variable, indicating that the violent crime in a property has a lesser impact on buying decision compared to property crimes.

Conclusion:

After trying 2 different latent variables with supporting data amalgamation, we found that the model that fulfills all our requirements, i.e. high f1 score and precision score with respect to class 2. With almost 0 false positives for class 2, was built using a DecisionTree classifier which used the base data and just the property crime as a latent variable. Including violent crimes in an area did not affect our best model in any way, leading us to a conclusion that violent crimes would play a lesser role then property crimes when it comes to making the purchasing decision.
The top 5 features from our classification are 'price' - current selling price 'rent_zestimate' - current rent estimate ,'area_sqft' 'zestimate' - estimated market value 'morgage_fee' - monthly mortgage if decided to buy.

Additional DataSets Used:
Zillow Home Value Index (for all houses): https://www.zillow.com/research/data/
 Crime Rate Dataset OR https://www.ppic.org/data-set/crime-rates-in-california/ 
Dataset for Zip Mapping: https://www.unitedstateszipcodes.org 
