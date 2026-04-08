# ICRAS Field Guide

Explains every input field, what it means in the context of the Home Credit dataset,
realistic value ranges, and how much it influences the model's prediction.

## Dataset Context

The model was trained on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
dataset. Home Credit is a consumer finance company operating in emerging markets,
primarily issuing **long-tenor secured installment loans** for large purchases
(vehicles, home appliances, property). This means:

- Loan amounts are typically **3–5x annual income** — normal given 10–20 year repayment periods
- `AMT_GOODS_PRICE` is the purchase price of the financed asset (collateral)
- The `annuity_income_ratio` acts as an implicit affordability/tenor proxy
- Loan tenor is not an explicit feature — it is indirectly captured through annuity figures

---

## Traditional Features

### Annual Income (`amt_income_total`)

Total declared annual income of the applicant. Not sent to the model directly —
used only to derive the ratio fields below.

- Typical range: $30,000 – $200,000
- Example: `160000`

### Loan Amount (`AMT_CREDIT`)

Total loan principal requested.

- Typical approved range: $100,000 – $700,000
- Expected to be a large multiple of income given long tenors
- Example: `526500`

### Loan Annuity (`AMT_ANNUITY`)

Annual repayment amount. One of the top 7 most important features.
**`annuity_income_ratio` is the single most important affordability signal in the model.**
The model treats ~0.15 (15% of income) as a hard cliff — below it approvals are common,
above it the default probability jumps sharply to ~0.47+ regardless of other factors.
This is the model's proxy for tenor: a large loan with a low annuity ratio implies a
long repayment period and manageable monthly burden.

- Derived ratio: `annuity_income_ratio = AMT_ANNUITY / income`
- Safe range: **below 0.13** — probability stays low
- Danger zone: **0.15 and above** — probability spikes dramatically
- `credit_income_ratio` on its own has minimal impact; it's the annuity ratio that matters
- Example: `24750` (15% of $160k income — borderline; `4000` on $200k income = 0.02 is very safe)

### Goods Price (`AMT_GOODS_PRICE`)

Purchase price of the asset being financed. Functions as collateral value.
Should be close to but slightly below the loan amount.

- `goods_price_credit_ratio = goods_price / credit` — healthy range: 0.85–0.95
- A ratio near 1.0 means the loan is almost fully asset-backed (lower risk)
- A low ratio means a large unsecured component (higher risk)
- Example: `463500` (ratio = 0.88 on a $526,500 loan)

### Credit History Age (`bureau_days_credit_mean`)

Average age of the applicant's credit accounts in days. Enter as a **positive number** —
the backend automatically negates it to match the model's training convention.

- Approved median: ~1,097 days (~3 years)
- Longer history is mildly positive but has moderate model importance
- Example: `1097`

### Credit Diversity (`bureau_credit_types`)

Number of distinct credit product types held (e.g. mortgage, car loan, credit card).

- Approved median: 2
- Moderate model importance
- Example: `2`

### Active Accounts (`bureau_active_count`)

Number of currently open credit accounts. One of the top 5 most important features.
Fewer open accounts strongly correlates with lower default risk.

- Approved median: **1**
- Having 3+ active accounts significantly increases predicted default probability
- Example: `1`

### Closed Accounts (`bureau_closed_count`)

Number of previously closed (fully repaid) credit accounts.

- Approved median: 2
- More closed accounts suggests a track record of successfully repaid loans
- Mild positive signal
- Example: `2`

### CC Limit (`cc_limit_mean`)

Average credit card limit across all cards. Many approved applicants have no
credit card (enter `0`). Low model importance.

- Typical range if applicable: $5,000 – $15,000
- Example: `0` or `8000`

### CC Balance (`cc_balance`)

Current outstanding credit card balance. Used to derive:
`cc_utilization = balance / limit`. Zero utilization is ideal but even 20–32%
has minimal impact on this model.

- If CC limit is 0, enter `0`
- Example: `0` or `2560` (32% of $8,000 limit)

---

## Alternative Features

> **Financial Inclusion Context**
> In the Home Credit dataset these features are derived from internal loan records.
> However the broader purpose of this project is to demonstrate that alternative data
> can substitute for formal credit history for underserved populations. In a real-world
> deployment these fields could be populated from informal sources:
>
> - Microfinance / savings group repayments (SACCOs, ROSCAs)
> - Utility and mobile data bill payment history
> - Mobile money transaction records (M-Pesa, etc.)
> - Rent payment history
> - Merchant or supplier credit repayment records
>
> The model does not distinguish the source — it treats `inst_payment_ratio = 1.0`
> as evidence of payment discipline regardless of whether that came from a bank loan
> or a microfinance group. This is the bridge between informal and formal financial
> systems that financial inclusion initiatives aim to build.

### Prev. Approval Rate (`prev_approved_ratio`)

Proportion of the applicant's previous loan applications that were approved.
One of the top 6 most important features. Enter as a decimal (0.0–1.0).

- Approved median: 0.82
- A high rate signals the applicant has consistently met lender criteria
- Example: `0.82`

### Installment Payment Ratio (`inst_payment_ratio`)

Proportion of installment payments made on time across all previous loans.
One of the top 4 most important features. Enter as a decimal (0.0–1.0).

- Approved median: **1.0** (perfect payment history)
- Anything below 0.95 starts to meaningfully increase default probability
- Example: `1.0`

### Avg DPD (`inst_dpd_mean`)

Average days past due on installment payments. Enter as a **positive number** —
the backend automatically negates it to match the model's training convention.

- Approved median: ~10 days (enter `10`)
- `0` means the applicant has never been late
- Lower is better; one of the top 9 most important features
- Example: `0` or `10`

### Geographic Stability (`geo_stability`)

Score reflecting how long the applicant has lived at their current address/region.
Very low discriminating power in this model — median is 0 for both approved and rejected.

- Range: 0–3
- Example: `0`

### Owns Real Estate (`FLAG_OWN_REALTY`)

Whether the applicant owns property. Minimal model impact.

- Set based on actual applicant information

### Mobile Phone (`FLAG_MOBIL`)

Whether the applicant provided a mobile number. Near-zero model impact.

- Set based on actual applicant information

### Work Phone (`FLAG_EMP_PHONE`)

Whether the applicant has a work/employer phone. Marginally increases predicted
default probability when set to Yes — likely a dataset artifact rather than
a true causal signal. Low importance overall.

- Set based on actual applicant information

### Work Phone Registered (`FLAG_WORK_PHONE`)

Whether the work phone is registered. Near-zero model impact.

- Set based on actual applicant information

---

## Feature Importance Ranking

From the model's training (Gradient Boosting feature importance scores):

| Rank | Feature                  | Type        | Importance |
| ---- | ------------------------ | ----------- | ---------- |
| 1    | bureau_days_credit_mean  | Traditional | 0.1619     |
| 2    | cc_utilization           | Traditional | 0.1124     |
| 3    | goods_price_credit_ratio | Traditional | 0.0982     |
| 4    | inst_payment_ratio       | Alternative | 0.0954     |
| 5    | bureau_active_count      | Traditional | 0.0862     |
| 6    | prev_approved_ratio      | Alternative | 0.0759     |
| 7    | AMT_ANNUITY              | Traditional | 0.0709     |
| 8    | annuity_income_ratio     | Traditional | 0.0677     |
| 9    | inst_dpd_mean            | Alternative | 0.0512     |
| 10   | AMT_GOODS_PRICE          | Traditional | 0.0501     |

The flag fields (MOBIL, EMP_PHONE, WORK_PHONE, OWN_REALTY) and geo_stability
collectively account for less than 5% of model importance and can be treated
as supplementary context rather than decision drivers.

---

## Sample Approved Profiles

Based on the median approved applicant in the test set (probability: ~3.75%):

| Field                 | Value   | Notes                         |
| --------------------- | ------- | ----------------------------- |
| Annual Income         | 160,000 | Implies credit/income of 3.29 |
| Loan Amount           | 526,500 |                               |
| Annuity               | 24,750  | 15% of income                 |
| Goods Price           | 463,500 | Ratio 0.88                    |
| Credit History (Days) | 1,097   | ~3 years                      |
| Credit Diversity      | 2       |                               |
| Active Accounts       | 1       | Keep low                      |
| Closed Accounts       | 2       |                               |
| CC Limit              | 0–8,000 | Low importance                |
| CC Balance            | 0       | 0% utilization ideal          |
| Prev Approval Rate    | 0.82    |                               |
| Installment Pay Ratio | 1.0     | Perfect payment history       |
| Avg DPD               | 0       | Never late                    |
| Geographic Stability  | 0       | Low importance                |
| All flags             | No      | Low importance                |

| Field                 | Value   | Notes                         |
| --------------------- | ------- | ----------------------------- |
| Annual Income         | 200,000 | Implies credit/income of 0.10 |
| Loan Amount           | 20,000  |                               |
| Annuity               | 4,000   | Only 2% of income             |
| Goods Price           | 18,000  | Ratio 0.90                    |
| Credit History (Days) | 720     | ~2 years                      |
| Credit Diversity      | 2       |                               |
| Active Accounts       | 1       | Keep low                      |
| Closed Accounts       | 1       |                               |
| CC Limit              | 2,000   |                               |
| CC Balance            | 300     | 15% utilization               |
| Prev Approval Rate    | 0.90    |                               |
| Installment Pay Ratio | 1.0     | Perfect payment history       |
| Avg DPD               | 0       | Never late                    |
| Geographic Stability  | 3       |                               |
| All flags             | Yes     | Low importance                |
