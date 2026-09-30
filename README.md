# csurv — A Mixture Cure Survival Model for Customer Retention

A restaurant owner watches customers come and go, week after week.  Many visit only once, but some return again and again.  After a regular doesn't return for several weeks, she wonders if she should worry - what happened on their last visit?  From the service floor, it's hard to see patterns in guest loyalty - coincidences look like patterns and real trends are often invisible.  

One approach to answering these questions comes from the world of clinical trials.  Used across indications, but especially in oncology, **survival analysis** is a family of approaches to answering questions about *time-to-event* data, especially *censored data*, where the outcome of some observations is unknown.  

In this repo, we explore both a classical survival analysis approach and, in response to its flaws in the restaurant context, explore an alternative approach.  

## Ideas from Oncology
Imagine you are running a small trial - 10 patients being treated for a rare disease receive a drug designed to prevent a common complication.  Two patients have the complication soon after being treated - say, after two days.  Two more patients have it eventually, but the treatment works for longer, say after two weeks.   Of the remaining six two left the study early for unknown reasons, and four don't have the complication for the duration of the study.  How could you describe for how long we can expect patients taking this drug to avoid the complication? If we only average the times to occurrence for the patients who had the complication, we'd be wildly underestimating the drug's effectiveness.  But for six of the subjects, we don't know for how long the treatment will work.  We say that the time-to-event data for these subjects is *censored*.

Our restaurant owner wants to know how long it takes for guests to return, so she knows when to follow-up with a regular who hasn't come back in a while. Applying the standard oncology approach to restaurant data, we obtain a *Kaplan-Meier* estimate of the survival function (the complement of the CDF of the time-to-return random variable).  At first glance, this curve seems to suggest that 12% of customers never return.  

![KM curve](notebooks/km_gap_time.png)

However, the curve is potentially misleading.  The standard approach used in oncology isn't built for an event that can occur repeatedly and under reasonable assumptions, this asymptote shows the reciprocal of the average number of visits per customer.    

If customers flipped a (biased) coin after each visit to decide whether or not to return, the proportion of customers who fail to return after one visit would be equal to the reciprocal of the expected value of the number of visits per customer - but that's not reflective of the reality of a small business.  First-time visitors could be easily turned off by a mediocre experience, while regulars may be forgiving of a bad experience.

We want an approach to the problem that explicitly models both the dynamics of repeat return customers and this underlying probability that a customer may never return.

## The Data

To explore our alternate model, we generate synthetic data by simulating a real restaurant.  Each customer is generated individually, assigned traits drawn from appropriate distributions (mostly beta distributions), and then the impact of those traits on likelihood of return are also drawn from beta distributions.  

At each visit, customers' likelihood to churn remembers (partially) their likelihood - intuitively, their attitude towards the restaurant - at the previous visit and adjusts based on their experience at the current visit (modeled using more beta distributions and random selections of dishes ordered and other visit characteristics)

The generated data look plausible on first glance (we'll get back to this):

![EDA](data/eda_verification.png)

## The Model
The model we implement is [discussed further in the mathematical writeup](docs/cure_model.md).  In brief summary, we use a parametric mixture-cure model.  The advantage of a parametric model is that it gives us parameters we can train - and using a Weibull distribution gives us a flexible hazard ratio - we can account for a wide variety of return dynamics.

The parameters are optimized using an EM algorithm with L-BFGS-B on the M step.

The model learns linear maps from normalized feature vectors (describing the characteristics of a particular customer and visit) the outputs of which, after one non-linear step, become the parameters in that individual's instance of the mixture-cure model.  We then use these parameter to simulate the time to churn for over the full population.

One note: a common survival analysis tool, lifelines, does provide a univariate mixture cure fitter.  However it does not support covariates in either the incidence or latency component — this project implements the full regression version.  

## Results

In the original simulation, we saved the "true" event/non-event distinction (not visible in the synthetic data fed to the model) for model validation - the true churn rate on each customer's final visit exactly matches the predicted churn rate.

| $\pi_{\text{churn}}$ | $\widehat{\pi}_{\text{churn}}$ |
|------|-----------|
|$12.2\%$ | $12.2\%$ |


The shape and scale parameters specified in the synthetic data were also closely predicted:

| $k$ | $\widehat{k}$ |
|------|-----------|
|$1.5$ | $1.36$ |

| $\lambda$ | $\widehat{\lambda}$ |
|------|-----------|
|$14$ | $12.2$ |

The true value of $\pi$ also isn't too far off from the KM curve above.  So why go to all this trouble? Because the KM curve doesn't tell the whole story - churn is largely dependent on loyalty in these data, and the model learned a coefficient for `gap_num` (β = −6.25, r = −0.35), the variable which encoded visit number, which provided accurate predictions of per-visit churn. 

| Visit | $\pi_{\text{churn}}$ | $\widehat{\pi}_{\text{churn}}$ |
|------:|---------------------:|-------------------------------:|
| 1     | $44.8\%$             | $44.1\%$                       |
| 2     | $40.0\%$             | $33.1\%$                       |
| 3     | $22.1\%$             | $23.8\%$                       |
| 4     | $10.7\%$             | $16.3\%$                       |
| 5     | $7.3\%$              | $10.9\%$                       |
| 6     | $4.0\%$              | $7.2\%$                        |
| 7     | $3.5\%$              | $4.6\%$                        |
| 8     | $1.6\%$              | $3.0\%$                        |
| 9     | $1.6\%$              | $1.9\%$                        |
| 10    | $1.2\%$              | $1.2\%$                        |


Of 17 covariates with non-negligible impact on churn rate, the model predicted influence on churn correctly (with correct sign) for $15$ of them.

Estimated parameters were also used to simulate customer outcomes for the censored records.  The estimated parameters predict that approximately 20% of customers become regulars.

![Survival](notebooks/survival_curve.png)

To quickly explore the impact of reducing the churn factor on customer lifetime value, you can run our streamlit app in docker:

![streamlit](notebooks/streamlit.png)

## Usage

### Quick start
```bash
docker compose up
# API: localhost:8000/docs
# Dashboard: localhost:8501
```
### CLI
```python
uv run csurv demo              # generate + train + predict
uv run csurv predict --db ...  # score customers
```
### API endpoints
| Endpoint | Method | Description |
|---------|------|----------|
| /health | GET | Status check |
| /predict | POST | Customer predictions |
| /uplift | POST | CLV uplift analysis |
| /survival | POST | Survival curve |

## Testing
```python
uv run pytest
```
## License
MIT