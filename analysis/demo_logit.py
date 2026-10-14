"""MSME mechanisation adoption - demo on SYNTHETIC survey data (n=250).

Effects are PLANTED in the generator so the demo can show the method recovers them.
These are not the study's findings. Steps: factor analysis -> binary logit (odds ratios
with 95% CI) -> OLS on an intensity score.
"""
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.decomposition import FactorAnalysis


def make_survey(n=250, seed=11):
    r = np.random.default_rng(seed)
    urban = r.integers(0, 2, n)
    big = (r.integers(1, 40, n) > 10).astype(int)
    credit = r.binomial(1, 0.4 + 0.1 * urban, n)
    trained = r.binomial(1, 0.45, n)
    # correlated Likert-style items to feed factor analysis
    fin = r.normal(0, 1, n) + credit; aw = r.normal(0, 1, n) + trained
    items = pd.DataFrame({"loan_ease": fin + r.normal(0, .5, n), "interest_affordable": fin + r.normal(0, .5, n),
                          "knows_tech": aw + r.normal(0, .5, n), "seen_demo": aw + r.normal(0, .5, n)})
    lp = -1.8 + 1.1 * credit + 0.9 * trained + 1.0 * urban + 0.8 * big
    adopt = r.binomial(1, 1 / (1 + np.exp(-lp)))
    intensity = 20 + 12 * adopt + 4 * credit + 3 * trained + r.normal(0, 6, n)
    return pd.DataFrame({"adopted": adopt, "credit": credit, "trained": trained, "urban": urban, "firm_gt10": big,
                         "intensity": intensity}).join(items)


def main():
    d = make_survey()
    fa = FactorAnalysis(2, rotation="varimax", random_state=0).fit(d[["loan_ease", "interest_affordable", "knows_tech", "seen_demo"]])
    print("factor loadings (rows = items):\n", pd.DataFrame(fa.components_.T, index=["loan_ease", "interest_affordable", "knows_tech", "seen_demo"],
                                                            columns=["F1", "F2"]).round(2), "\n")
    X = sm.add_constant(d[["credit", "trained", "urban", "firm_gt10"]])
    logit = sm.Logit(d["adopted"], X).fit(disp=0)
    ci = np.exp(logit.conf_int())
    print("odds ratios (95% CI):")
    print(pd.DataFrame({"OR": np.exp(logit.params), "lo": ci[0], "hi": ci[1], "p": logit.pvalues}).round(2).to_string())
    ols = sm.OLS(d["intensity"], sm.add_constant(d[["adopted", "credit", "trained", "urban", "firm_gt10"]])).fit(cov_type="HC3")
    print("\nOLS intensity (robust SE):\n", ols.summary().tables[1])


if __name__ == "__main__":
    main()
