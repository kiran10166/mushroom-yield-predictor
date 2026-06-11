\# Linear Regression Diagnostics



\## Residual Definition



Residual = Actual Yield - Predicted Yield



A positive residual indicates the model underpredicted yield.



A negative residual indicates the model overpredicted yield.



\---



\## Diagnostic Findings



\### Residuals vs Predicted Yield



The residuals are centered around zero, indicating that the model does not exhibit severe systematic bias.



Any visible funnel shape would indicate heteroscedasticity.



\### Residuals vs Humidity



Residuals are randomly distributed across humidity values with no strong pattern.



A visible curve would indicate a nonlinear relationship not captured by Linear Regression.



\---



\## Recommendation



The baseline Linear Regression model provides a useful starting point.



If residual plots show curvature or increasing variance:



\- Add engineered features.

\- Try polynomial features.

\- Evaluate nonlinear models such as Random Forest Regression.



Otherwise, continue with the linear baseline as a benchmark.

