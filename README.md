# Capstone_CC_Churn

"In today's subscription-based economy, high customer churn poses a significant threat to enterprises and service providers. This capstone project tackles this critical challenge by leveraging machine learning to predict customer churn risk within a contact center. By analyzing data encompassing customer interactions, agent performance, and demographic factors, the project aims to develop a predictive model capable of identifying customers at high risk of churn. This model will empower contact centers to proactively implement targeted retention strategies, ultimately boosting customer satisfaction and minimizing revenue attrition.

# 1. Business Understanding
### Churn 
Churn refers to a customer discontinuing their relationship with a business. In the context of a contact center, churn is defined as:
A customer canceling their subscription or contract.
A customer reducing their usage significantly before canceling.
A customer switching to a competitor after repeated negative experiences.
Long periods of inactivity or non-engagement with the contact center.

## Business Goal
The primary goal is to accurately predict customer churn risk using contact center interaction data. By identifying key factors leading to churn, the business can proactively implement retention strategies, improve customer satisfaction, and reduce revenue loss.
The Capstone will focus on answering:
What are the leading indicators of customer churn in contact center interactions?
How can businesses use predictive models to intervene before customers churn?
What are the most impactful variables contributing to churn (e.g., call resolution time, sentiment in interactions, frequency of complaints, agent performance, etc.)?

## Success Metrics
To evaluate the effectiveness of the churn prediction model, success will be measured using:
Prediction Accuracy – Improvement in churn prediction accuracy (e.g., ROC-AUC score, F1 score).
Reduction in Actual Churn Rate – Percentage decrease in customer churn after implementing predictive insights.
Retention Strategy Impact – Increase in customer retention rates after proactive interventions.
Operational Efficiency – Reduction in the number of calls per churned customer, indicating improved first-call resolution and service quality.

## Relevance of Churn
Churn has a significant impact on business profitability, making its prediction and prevention critical. High churn rates lead to:

## Revenue Loss – Losing existing customers is more expensive than acquiring new ones.
Increased Customer Acquisition Costs (CAC) – More resources are needed to attract new customers.
Brand Reputation Damage – Poor customer experiences spread through reviews and word-of-mouth.
Operational Strain – Unhappy customers may increase call volume before leaving, burdening the contact center.

By proactively identifying at-risk customers, businesses can implement targeted retention strategies, such as personalized offers, better service, and proactive outreach, to enhance customer satisfaction and loyalty.

## Project Constraints
Several challenges may impact the project's scope and execution:

Data Availability – Access to comprehensive and high-quality customer interaction data may be limited.
Time Constraints – Model development, testing, and validation must be completed within the project timeline.
Computational Resources – Running machine learning models on large datasets may require high processing power.
Interpretability – Ensuring the model is explainable for business stakeholders to act on insights effectively.

# 2. Data Understanding

Data source: "data/SiddiCC_Churn_data.csv" containing customer data
The dataset consists of 3,150 records and 25 features, covering various aspects of customer behavior, usage, and interactions with the contact center.

## Key Features:
Customer Interaction Metrics

Call Failure: Number of failed call attempts.
Complains: Whether the customer has made complaints (binary).
Seconds_of_Use: Total call duration in seconds.
Frequency_of_Use: Number of calls made.
Frequency_of_SMS: Number of SMS sent.
Distinct_Called_Numbers: Unique numbers called by the customer.
Transfer_Count: Number of times a call was transferred.
Callback_Count: Number of callbacks made to the customer.
Customer Subscription & Financial Metrics

Subscription_Length: Duration of the subscription (in months).
Charge Amount: Total charges billed to the customer.
Customer_Value: A numerical value representing the overall importance of the customer.
Tariff_1 & Tariff_2: Boolean indicators of different tariff plans.
Quality of Service & Sentiment Metrics

AHT (Average Handling Time): Time taken to resolve a call.
FCR (First Call Resolution): Whether the issue was resolved on the first call (binary).
Sentiment_Score: Sentiment of customer interactions (likely derived from text analysis).
SLA_Compliance: Compliance with Service Level Agreements (percentage).
Service_Gap: Difference between SLA target and actual performance.
Complexity_Score: Complexity of the customer’s queries.
Demographics & Customer Status

Age & Age_Group_Numeric: Customer’s age and numerical group classification.
Status: Customer’s status (potentially active, inactive, or on-hold).
Target Variable

Churn: Binary indicator of whether the customer has churned (1) or not (0).
Observations:
No missing values detected in the dataset.
The dataset contains a mix of numerical, categorical (binary), and continuous features.
Churn is the primary target variable for prediction.
Features like Sentiment_Score, FCR, Service_Gap, and Complexity_Score may provide strong insights into customer dissatisfaction.

Next Steps:
Check for class imbalance in the Churn column.
Analyze feature distributions (e.g., correlation between features and churn).
Feature engineering (e.g., converting categorical variables, creating new derived features).
Outlier detection to identify potential anomalies in the data.

# 3. Data Preparation
1. Class Imbalance in Churn
84.29% of customers did not churn (Churn = 0)
15.71% of customers churned (Churn = 1)
The dataset is imbalanced, meaning we may need resampling techniques (e.g., SMOTE or weighted models) to improve predictions.
2. Correlation with Churn
Positively correlated features (higher churn risk):

Complains (0.53) → Customers with complaints are more likely to churn.
Status (0.50) → Certain customer statuses may indicate higher churn probability.
Tariff_1 (0.11) → Some tariff plans may lead to higher churn.
Negatively correlated features (less likely to churn):

Customer_Value (-0.29) → High-value customers are less likely to leave.
Seconds_of_Use (-0.30) → More call usage is associated with retention.
Frequency_of_SMS (-0.22) → Higher SMS usage means lower churn.
Subscription_Length (-0.03) → Longer subscription durations correlate with lower churn.
3. Outliers Detected (Potential Cleaning Required)
Significant outliers found in:

Charge Amount (370 outliers)
Frequency of SMS (368 outliers)
Subscription Length (282 outliers)
Status (782 outliers)
Age (688 outliers)
Churn (495 outliers)
Some features like AHT, FCR, Sentiment_Score, and SLA_Compliance do not have outliers.

Next Steps
Handle class imbalance (oversampling, undersampling, or class weights).
Remove or cap outliers to prevent them from skewing the model.
Feature selection based on correlation analysis.
Scale numerical features (normalization or standardization).
Convert categorical features (e.g., Tariff_1, Status) into machine-learning-friendly formats.

# Exploratory Data Analysis (EDA) Summary
Churn Distribution

The dataset is imbalanced, with significantly more non-churned customers (Churn = 0) compared to churned customers (Churn = 1).
Correlation Heatmap

Complains and Customer_Value show strong correlation with Churn.
Features like Seconds_of_Use, Subscription_Length, and Charge Amount also exhibit meaningful relationships with churn.
Box Plot Analysis

Customers who complained (Complains = 1) have a higher chance of churn.
Higher Customer_Value is associated with lower churn.
Lower Seconds_of_Use and Subscription_Length indicate a higher likelihood of churn.
Charge Amount shows significant variation but no clear trend.
Subscription Length vs. Churn

Churned customers tend to have shorter subscription lengths, whereas customers with longer subscriptions are more likely to stay.

Key Takeaways for Churn Prediction
Customer complaints, low usage, and short subscription lengths are strong churn indicators.
Handling class imbalance is crucial for accurate modeling.
Feature engineering can enhance predictive power (e.g., engagement metrics, sentiment analysis).






