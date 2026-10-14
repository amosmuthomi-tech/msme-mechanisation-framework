# MSME Mechanisation Adoption Framework
### What drives small construction firms to adopt machinery? · NCA Kenya · 2025

Survey-based study of 250 construction MSMEs across Kenya, analysed with factor analysis, binary logistic regression (adoption yes/no) and OLS (mechanisation intensity). Reported on the real (unpublished) survey data: access to affordable credit was the strongest predictor of adoption (OR ≈ 3.2), technology-awareness training raised adoption probability by about 40%, and urban counties adopted about 3× more than rural ones.

Survey data belongs to the National Construction Authority and is not published. `analysis/demo_logit.py` reproduces the **method** on synthetic data with effects planted in the generator. Its odds ratios are not the study's findings.

## Run the demo
```bash
pip install -r requirements.txt
python analysis/demo_logit.py
```
Prints varimax factor loadings, odds ratios with 95% CIs, and an OLS intensity model with robust errors.

## Method
1. **Factor analysis** to reduce correlated survey items to finance-access and awareness factors
2. **Binary logit** on adoption, reporting odds ratios and confidence intervals
3. **OLS** on an intensity score with robust standard errors

## Limitations
- Cross-sectional survey: associations, not causal effects
- Self-reported adoption and access-to-credit measures
- The demo cannot validate the real results

`Python` `statsmodels` `scikit-learn` `SPSS` `R` `Power BI`
