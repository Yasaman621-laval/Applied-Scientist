# Applied-Scientist

Accessible Summary 
•	I improved the company’s original forecasting approach by adding recent trends and patterns into the models, not just repeating seasonal cycles.
•	For both redemptions and sales, I included the previous day’s count, the 7-day moving average, and the day of the week as extra features, which help the models understand recent changes and weekly habits.
•	I used modern machine learning models (Random Forests and Quantile Random Forests) to predict future ticket sales and redemptions, instead of just using the average seasonal pattern.
•	My new models give not only more accurate daily forecasts, but also a confidence range for each prediction, so planners can see both the likely numbers and the level of uncertainty.
•	Careful data cleaning and thorough validation were done to make sure the results are reliable.
•	In summary, my approach gives much better accuracy and practical value than the original model, making it easier for the business to plan for demand and resources.
________________________________________
Technical Summary 
At the start of the project, the provided base model only used simple yearly seasonality to forecast daily redemptions or sales, limiting its ability to react to real-world changes. My approach focused on enhancing predictive power and practical usability for both redemptions and sales.
Data Preparation and Feature Engineering:
I ensured all timestamps were ordered and in the correct format. I engineered new features to reflect recent demand: the ticket count from the previous day (lag), the 7-day rolling mean, and the day of the week. These features help models capture both short-term shifts and weekly seasonality. I also filtered out any rows with missing or zero values in the target and key features, which could otherwise distort the accuracy metrics.
Model Building and Validation:
Using time-based cross-validation, I compared several models:
•	Base Model: Relied only on repeating seasonal patterns.
•	Linear Regression: Incorporated the new features, improving the model’s ability to follow recent and weekly changes.
•	Random Forest: A machine learning model able to detect more complex relationships, which consistently lowered the forecasting error.
•	Quantile Random Forest (QRF): Extended the random forest by adding bootstrapped intervals, providing a prediction range to express uncertainty, not just a single number.
I added comprehensive print statements to the code, allowing me to monitor data splits, feature values, and sample predictions at each step. This was especially useful when diagnosing why some periods (such as fold 0) were more challenging, confirming that the issue was not with data quality but with the unpredictability of those periods.
Results and Comparison:
Across all models and folds, the random forest and QRF models gave the lowest Mean Absolute Percentage Error (MAPE), particularly in folds 1, 2, and 3. The QRF model was especially valuable, as it provided both accurate point predictions and useful prediction intervals, which are important for risk-aware planning. In contrast, the base model had the highest errors, and linear regression was less reliable.
Conclusion:
By adding recent and weekly trends, using more advanced models, and providing prediction intervals, my approach offers much better accuracy and greater practical value than the original base model. This improved forecasting enables better day-to-day planning and resource allocation for both redemptions and sales.



