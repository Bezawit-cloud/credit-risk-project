## Credit Scoring Business Understanding

### Basel II and Interpretability  
The Basel II framework emphasizes accurate measurement of credit risk and the requirement that banks hold sufficient capital against unexpected losses. In practice, this means credit models used for lending decisions must be **interpretable, well-documented, and auditable**, so that model behavior, assumptions, and limitations can be clearly explained to internal governance and external regulators. As we design a scoring product for Bati Bank, model transparency and traceability (data lineage, feature definitions, and validation) are therefore foundational requirements alongside predictive performance.

### Why a Proxy Label Is Necessary and the Risks  
The provided eCommerce dataset does not contain an explicit **default** field (i.e., an observed loan default). To train supervised models, we therefore must create a **proxy outcome** using observable behavior—for example, customers with chargebacks/refunds within *X* days, or customers who fall into the worst *X*% of RFM metrics.  
Using a proxy enables model building, but introduces business risk because the proxy may not perfectly represent true loan default. This can cause misclassification (false acceptances or false rejections). For this reason, we must (1) clearly document how the proxy is created, (2) test different proxy thresholds, and (3) treat model outputs as one input to credit decisions—not as the single source of truth.

### Tradeoffs: Simple vs. Complex Models  
Simple models (e.g., Logistic Regression with Weight-of-Evidence encoding) offer **high interpretability**, easy regulatory explanations, and simpler validation pipelines. They often perform competitively when features are well engineered and stable.  
Complex models (e.g., Gradient Boosting Machines) can achieve **higher predictive accuracy** and capture nonlinear patterns, but they are harder to explain and require stronger governance—such as feature importance validation, SHAP explainers, ongoing monitoring, and stricter overfitting checks.  
In a regulated financial environment, a balanced approach is to evaluate both: use an interpretable model as the primary decisioning tool, and use a high-performance model as a benchmark or secondary analysis, while documenting all reasoning for model choice.

### Planned Proxy for This Project  
For this challenge, we will define a proxy default label using **RFM-based customer behavior** and short-term negative events. Specifically:  
- Customers who fall into the **bottom decile (worst 10%)** of a combined RFM score, **or**  
- Customers who experience a **chargeback/refund within 90 days**  
will be labeled *high risk (bad)*.  
We will also test other thresholds (5%, 10%, 20%) and report sensitivity results to justify the final proxy selection.
