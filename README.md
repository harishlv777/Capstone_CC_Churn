# Capstone_ContactCenter_Churn for Software as a Subscription (SaaS) Business

"In today's subscription-based economy, high customer churn poses a significant threat to enterprises and service providers. This capstone project tackles this critical challenge by leveraging machine learning to predict customer churn risk within a contact center. By analyzing data encompassing customer interactions, agent performance, and demographic factors, the project aims to develop a predictive model capable of identifying customers at high risk of churn. This model will empower contact centers to proactively implement targeted retention strategies, ultimately boosting customer satisfaction and minimizing revenue attrition.

# 1. Business Understanding
### Churn 
Churn refers to a customer discontinuing their relationship with a business.
- A customer canceling their software/hardware subscription or contract.
- A customer reducing their usage significantly before canceling.
- Long periods of inactivity or non-engagement with the contact center.
- A customer switching to a competitor after repeated negative experiences.

## Business Goal
Primary goal is to accurately predict customer churn risk using contact center interaction data. By identifying key factors leading to churn, the business can proactively implement retention strategies, improve customer satisfaction, and reduce revenue loss.

Capstone_ContactCenter_Churn will focus on answering:
- What are the leading indicators of customer churn in contact center interactions?
- How can businesses use predictive models to intervene before customers churn?
- What are the most impactful variables contributing to churn (e.g., complaints, status, frequency of use, usage in seconds, Average Handle time, sentiment score etc.)?

## Success Metrics
To evaluate the effectiveness of the churn prediction model, success will be measured using:
- Prediction Accuracy – Improvement in churn prediction accuracy (e.g., ROC-AUC score, F1 score).
- Reduction in Actual Churn Rate – Percentage decrease in customer churn after implementing predictive insights.
- Retention Strategy Impact – Increase in customer retention rates after proactive interventions.
- Operational Efficiency – Reduction in the number of calls per churned customer, indicating improved first-call resolution and service quality.

## Relevance of Churn
Churn has a significant impact on business profitability, making its prediction and prevention critical. High churn rates lead to:

## Revenue Loss – Losing existing contact center subscription customers is more expensive than acquiring new ones.
- Increased Customer Acquisition Costs (CAC) – More resources are needed to attract new customers.
- Brand Reputation Damage – Poor customer experiences spread through reviews and word-of-mouth.
- Operational Strain – Unhappy customers may increase call volume before leaving, burdening the contact center.

By proactively identifying at-risk customers, businesses can implement targeted retention strategies, such as personalized offers, better service, and proactive outreach, to enhance customer satisfaction and loyalty.

## Project Constraints
Several challenges may impact the project's scope and execution:
- Data Availability – Access to comprehensive and high-quality customer interaction, agent performance and contact center data may be limited. Contact Center data consisting of 3K+ records was considered.
- Time Constraints – Model development, testing, and validation is critical given realtime nature of contact center and as-a-subscription business
- Computational Resources – Running machine learning models on large datasets may require high processing power.
- Interpretability – Ensuring the model is explainable for business stakeholders to act on insights effectively.

# 2. Data Understanding
Data source: "data/SiddiCC_Churn_data.csv" containing customer data
The dataset consists of 3,150 records and 25 features, covering various aspects of customer behavior, usage, and interactions with the contact center.

## Key Features:
### Customer Interaction Metrics
- Call Failure: Number of failed call attempts.
- Complains: Whether the customer has made complaints (binary).
- Seconds_of_Use: Total call duration in seconds.
- Frequency_of_Use: Number of calls made.
- Frequency_of_SMS: Number of SMS sent.
- Distinct_Called_Numbers: Unique numbers called by the customer.
- Transfer_Count: Number of times a call was transferred.
- Callback_Count: Number of callbacks made to the customer.

### Customer Subscription & Financial Metrics
- Subscription_Length: Duration of the subscription (in months).
- Charge Amount: Total charges billed to the customer.
- Customer_Value: A numerical value representing the overall importance of the customer.
- Tariff_1 & Tariff_2: Boolean indicators of different tariff plans.

### Quality of Service & Sentiment Metrics
- AHT (Average Handling Time): Time taken to resolve a call.
- FCR (First Call Resolution): Whether the issue was resolved on the first call (binary).
- Sentiment_Score: Sentiment of customer interactions (likely derived from text analysis).
- SLA_Compliance: Compliance with Service Level Agreements (percentage).
- Service_Gap: Difference between SLA target and actual performance.
- Complexity_Score: Complexity of the customer’s queries.

### Demographics & Customer Status
- Age & Age_Group_Numeric: Customer’s age and numerical group classification.
- Status: Customer’s status (potentially active, inactive, or on-hold).

### Target Variable
- Churn: Binary indicator of whether the customer has churned (1) or not (0).

## Observations:
- No missing values detected in the dataset.
- The dataset contains a mix of numerical, categorical (binary), and continuous features.
- Churn is the primary target variable for prediction.
- Features like Sentiment_Score, First Call Resolution (FCR), Service_Gap, and Complexity_Score may provide strong insights into customer dissatisfaction.

## Next Steps:
- Checked for class imbalance in the Churn column.
- Analyzed feature distributions (e.g., correlation between features and churn).
- Feature engineering (e.g., converting categorical variables, creating new derived features).
- Outlier detection to identify potential anomalies in the data.

# 3. Data Preparation

## 1. Class Imbalance in Churn
- 84.29% of customers did not churn (Churn = 0)
- 15.71% of customers churned (Churn = 1)
- The dataset is imbalanced, meaning we may need resampling techniques (e.g., SMOTE or weighted models) to improve predictions.

## 2. Correlation with Churn
### Positively correlated features (higher churn risk):
- Complains (0.53) → Customers with complaints are more likely to churn.
- Status (0.50) → Certain customer statuses may indicate higher churn probability.
- Tariff_1 (0.11) → Some tariff plans may lead to higher churn.

### Negatively correlated features (less likely to churn):
- Customer_Value (-0.29) → High-value customers are less likely to leave.
- Seconds_of_Use (-0.30) → More call usage is associated with retention.
- Frequency_of_SMS (-0.22) → Higher SMS usage means lower churn.
- Subscription_Length (-0.03) → Longer subscription durations correlate with lower churn.

## 3. Outliers Detected (Potential Cleaning Required)
- Significant outliers found in:
  - Charge Amount (370 outliers)
  - Frequency of SMS (368 outliers)
  - Subscription Length (282 outliers)
  - Status (782 outliers)
  - Age (688 outliers)
  - Churn (495 outliers)
  - Some features like AHT, FCR, Sentiment_Score, and SLA_Compliance do not have outliers.

## Next Steps
- Handled class imbalance (oversampling, undersampling, or class weights).
- Removed or cap outliers to prevent them from skewing the model.
- Feature selection was performed based on correlation analysis.
- Scaled numerical features (normalization or standardization).
- Converted categorical features (e.g., Tariff_1, Status) into ML friendly formats.

# Exploratory Data Analysis (EDA) Summary
Refer to the Capstone_CC_Churn_Plots_v1 https://github.com/harishlv777/Capstone_CC_Churn/blob/main/plots/Capstone_CC_Churn_plots_v1.pdf for detailed analysis.

## Churn Distribution
The dataset is imbalanced, with significantly more non-churned customers (Churn = 0) compared to churned customers (Churn = 1).
-  Customers with Status = 1 (basic software subscription) have a higher churn rate compared to customers with Status = 0.
-  It would be good to target the customers with basic subscription plans through campaigns, dedicated customer success managers, discounts, trial  to uplift them to premium subscription with value added features.
-  There more customers with “Basic Subscription (base plan)” who would cancel a subscription service than engaged customers with added features who actively use subscription programs
- 
## Correlation Heatmap
- Complains and Customer_Value show strong correlation with Churn.
- Features like Seconds_of_Use, Subscription_Length, and Charge Amount also exhibit meaningful relationships with churn.

## Box Plot Analysis
- Customers who complained have a higher chance of churn.
- Higher Customer_Value is associated with lower churn.
- Lower Seconds_of_Use and Subscription_Length indicate a higher likelihood of churn.
- Charge Amount shows significant variation but no clear trend.

## Subscription Length vs. Churn
- Churned customers tend to have shorter subscription lengths, whereas customers with longer subscriptions are more likely to stay.

## Key Takeaways for Churn Prediction
- Customer complaints, low usage, and short subscription lengths are strong churn indicators.
- Handling class imbalance is crucial for accurate modeling.
- Feature engineering can enhance predictive power (e.g., engagement metrics, sentiment analysis).
- There more customers with “Basic Subscription (base plan)” who would cancel a subscription service than engaged customers with added features who actively use a program

# Model Development
## 1. Data Preprocessing
- Encoded categorical variables (e.g., Status, Tariff Plan).
- Normalized numerical features using StandardScaler.
- Handled class imbalance using SMOTE to balance churn/non-churn cases.
- Split data into 80% training and 20% testing sets.

## 2. Model Training & Evaluation
Trained following models:
- Logistic Regression – Baseline model.
- Random Forest – Tree-based model for better feature importance.
- XGBoost – Optimized gradient boosting model.
- Support Vector Machine (SVM) - Supervised learning algorigthm for classification and regression tasks.
- K-nearest neighbor (KNN) - Supervised learning classifier, which uses proximity to make classifications or predictions

## Best Model: 
- Random Forest or Gradient Boosting typically outperformed other models (like Logistic Regression, SVM, or KNN) in terms of AUC-ROC (e.g., ~0.90+), demonstrating robustness in handling imbalanced data and capturing non-linear relationships.

- Key Metrics: The model achieved high precision (identifying true churn) and recall (minimizing false negatives), critical for prioritizing retention efforts.

## Model Performance Metrics
### Model Performance (Accuracy & AUC)
- Model	              Accuracy	AUC
- Logistic Regression	0.90	    0.92
- Random Forest	      0.95	    0.99
- Gradient Boosting	  0.95	    0.98
- SVM	                0.90	    0.93
- KNN	                0.90	    0.89

### Precision, Recall, and F1-Score (Class 0)
- Model	              Precision (0)	Recall (0)	F1-score (0)
- Logistic Regression	0.90	        0.99	      0.94
- Random Forest	      0.96	        0.98	      0.97
- Gradient Boosting	  0.96	        0.98	      0.97
- SVM	                0.90	        1.00	      0.94
- KNN	                0.91	        0.98	      0.94

### Precision, Recall, and F1-Score (Class 1)
- Model	              Precision (1)	Recall (1)	F1-score (1)
- Logistic Regression	0.85	        0.41	      0.56
- Random Forest	      0.87	        0.78	      0.82
- Gradient Boosting	  0.88	        0.77	      0.82
- SVM	                1.00	        0.37	      0.54
- KNN	                0.79	        0.48	      0.60

### Macro and Weighted Averages
- Model	              Macro Avg Precision	  Macro Avg Recall	  Macro Avg F1-Score	Weighted Avg Precision	Weighted Avg Recall	  Weighted Avg F1-Score
- Logistic Regression	0.88	                    0.70	                0.75	                0.89	                0.90	                0.88
- Random Forest	      0.91	                    0.88	                0.89	                0.94	                0.95	                0.94
- Gradient Boosting	  0.92	                    0.87	                0.90	                0.95	                0.95	                0.95
- SVM	                0.95	                    0.69	                0.74	                0.91	                0.90	                0.88
- KNN	                0.85	                    0.73	                0.77	                0.89	                0.90	                0.89

### Confusion Matrix
- Model	                      Confusion Matrix (0,0)	Confusion Matrix (0,1)	Confusion Matrix (1,0)	Confusion Matrix (1,1)
- Logistic Regression	              524	                         7	                  58	                      41
- Random Forest	                    519	                        12	                  22	                      77
- Gradient Boosting	                521	                        10	                  23	                      76
- SVM	                              531	                         0                  	62	                      37
- KNN	                              518	                        13	                  51	                      48

### Key Observations:

Top-performing model was chosen Best model based on ROC-AUC & F1-Score.

- **Best Model for Accuracy**: Random Forest and Gradient Boosting, both with an accuracy of 0.95.
- **Best Model for AUC**: Random Forest (0.99) performs the best for AUC, closely followed by Logistic Regression (0.92).
- **Precision** (Class 0): Random Forest and Gradient Boosting showed the highest precision for class 0, at 0.96 and 0.96, respectively.
- **Recall (Class 1)**: Logistic Regression has the lowest recall for class 1 (0.41), while Random Forest, Gradient Boosting, and SVM are better at capturing class 1 instances, with recall values of 0.78, 0.77, and 0.37, respectively.
- **F1-score (Class 0)**: Both Random Forest and Gradient Boosting show high F1-scores for class 0 (0.97), indicating a good balance between precision and recall.
- **Confusion Matrix**: The confusion matrices indicate that models like Random Forest and Gradient Boosting have fewer false positives (class 0 predicted as 1), with SVM and KNN showing more false negatives for class 1.
- **KNN's recall scores misses majority of actual churners, it is considered riskier for business.**
  
- #### Overall, Random Forest stands out as the best-performing model, closely followed by Gradient Boosting, especially in terms of overall accuracy, AUC, and balanced performance across both classes.

## Hyperparameter Tuning
Used GridSearchCV to fine-tune Random Forest parameters:
- n_estimators: 50, 100, 200
- max_depth: None, 10, 20
- min_samples_split: 2, 5, 10
- Found best-performing hyperparameters and re-trained the model.

## Key Insights
- Top churn predictors: Analyzing feature importance showed that variables like Call Duration, Complaint History, and Tariff Plan significantly impacted churn.
- SMOTE balancing will improve recall, reducing false negatives (customers likely to churn).
- Next steps: Deploy model in a contact center Customer Relationship Management (CRM) systems to predict churn and trigger proactive retention strategies.

# Conclusion
Top Features Impacting Churn
- **Status**: Customers with Status = 1 (basic software subscription) have a higher churn rate compared to customers with Status = 0. It would be good to target the customers with basic subscription plans through campaigns, dedicated customer success managers, discounts, trial  to uplift them to premium subscription with value added features. There more customers with “Basic Subscription (base plan)” who would cancel a subscription service than engaged customers with added features who actively use subscription programs
- **Call Failure**: Higher call failures strongly correlated with churn (technical issues drive dissatisfaction).
- **Complains**: Complaints were a direct indicator of dissatisfaction.
- **Usage Patterns**: Low "Seconds of Use" or "Frequency of Use" signaled disengagement.
- **Customer Value**: Lower-value customers were more likely to churn.
- **Subscription Length**: Newer customers showed higher churn risk.

# 3. Business Impact
- The model enables proactive retention strategies (e.g., targeted discounts for high-risk customers).
- Reducing churn by even 5% could save significant revenue for Contact Center Software as a Subscription companies (especially with thousands to millions of customers, and associated annual spend).

# Next Steps & Recommendations
**1. Model Improvements**
Address Class Imbalance: Use SMOTE or ADASYN to handle class imbalance and improve minority class prediction.
Feature Engineering: Create interaction terms (e.g., "Call Failure per Usage") or temporal features.
Advanced Models: Experiment with XGBoost, LightGBM, or neural networks for better performance.

**2. Deployment Strategies (for Customer Retention, Reactive >> Proactive apprach)**
- Real-Time Integration: Embed the model into Customer Relationship Management (CRM) systems systems to flag high-risk customers during service calls.
- Reactive to Preemptive/Proactive approach: Leverage the prediction insights to drive Proactive/Preemptive "Next best" customer interactions to avoid churn rather than a reactive approach.
- Automated Alerts: Trigger automated retention offers and/or assign Customer Success Managers, Customer Success Specialists to key accounts/regions which have predicted churners. Leverage Email/Chat/Outbound call campaigns for proactive value delivery and customer intimacy, retention.

**3. Ethical Considerations**
Bias Mitigation: Audit the model for fairness across demographics (e.g., age groups or regions).
Transparency: Use SHAP/LIME to explain predictions to customers and build trust.

**4. A/B Testing**
Test retention strategies on a subset of high-risk customers and measure churn reduction compared to a controlled group.

**5. Continuous Monitoring**
Retrain the model quarterly with fresh data to adapt to changing customer behavior.
Retrain the model for seasonal data as well and for specific industries (for eg., for Healthcare customer buying SaaS subcriptions, do model training during Open Enrollment phase of the year, for retail customer during Thanksgiving, Christmas time et al)
Track feature importance shifts over time (e.g., new pain points like "low adoption", "not signing up for new features", "customer stuck in a specific lifecycle" and not progressing to take best benefits of the subscription).

**Final Recommendation:** Prioritize campaigns to take care of "Basic subscription (base plan)" customers, have a dedicated customer success manager, proactive outbound campaign, discounts to uplift the customer from basic subscription to enhanced/premium subscriptions. Solve for technical issues (call failures) and improving customer service (reducing complaints) while deploying the model to target at-risk customers. This holistic approach will maximize retention and profitability.

# Additional considerations may also include 
- Hyperparameter tuning - Further experimentation with hyperparameter tuning for all  models, particularly the Logistic Regression, to see if you can squeeze out better performance
- Interaction Terms: Based on EDA (especially the correlations and bar charts), create interaction terms between features that seem to have a combined effect on the target variable.
- Polynomial Features: Consider adding polynomial features to capture non-linear relationships.
- Feature Selection/Dimensionality Reduction:
  - Regularization: For Logistic Regression, experiment with L1 (Lasso) or L2 (Ridge) regularization to reduce overfitting and potentially improve generalization.
  - PCA/Feature Importance: Use PCA or feature importance from a tree-based model (e.g., Random Forest) to select the most relevant features and reduce dimensionality.
- Recommend incorporating external data sources (e.g., consumer behvior, churn due to price vs competition offers) to enrich contact center feature set

# Additional Findings with XGBoost
In addition to the above, performed Extreme Gradient Boosting classifier based analysis. Refer to the https://github.com/harishlv777/Capstone_CC_Churn/blob/main/plots/Capstone_CC_Churn_plots_v1.pdf to review XGBoost classifier performance before and after hyperparameter tuning. 

## Key Takeaways
•	The XGBoost model performs well initially but shows bias toward the majority class.
•	Hyperparameter tuning significantly improves the model's performance, particularly for the minority class, as seen in better precision, recall, and fewer misclassifications.
•	The tuned model is more balanced and effective at distinguishing between the two classes while maintaining high overall accuracy and AUC.

## SHAP Interpretation
I tried to perform SHAP based interpretation to gain deeper insights into the decision-making process of  XGBoost model, helping validate its reliability and interpretability. However, I wasn’t able to execute SHAP due to shortage of compute resources on my laptop. I will look at leveraging SHAP for my future analysis

# Plots
https://github.com/harishlv777/Capstone_CC_Churn/blob/main/plots/Capstone_CC_Churn_plots_v1.pdf

# Files
- SiddiCC_Churn.ipynb - Jupyter notebook
- data/SiddiCC_Churn_data.csv - Contact Center dataset
- plots/Capstone_CC_Churn_plots.pdf - plots supporting the analysis
- readings - CRISP-DM-BANK.pdf CRISP-DM methodology document
- readme.md - current file

# Requirements
- Python 3.x, pandas, numby, matplotlib, seaborn, scikit-learn Note
- plot_helpers is required to render_plot
- Run pip list | grep plot_helpers to check if plot_helpers exists. If missing, either install it or replace render_plot with Matplotlib/Seaborn functions

# How to execute
- Clone the repository
- Build the environment with required packages, followed by Jypyter notebook execution.
