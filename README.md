## Credit Scoring Business Understanding

### Basel II and Interpretability  
The Basel II framework emphasizes accurate measurement of credit risk and the requirement that banks hold sufficient capital against unexpected losses. In practice, this means credit models used for lending decisions should be **interpretable, well-documented, and auditable**, so that model behavior, assumptions, and limitations can be clearly explained to internal governance and external regulators. As we design a scoring product for Bati Bank, model transparency and traceability (data lineage, feature definitions, and validation) are therefore foundational requirements alongside predictive performance.

### Why a Proxy Label Is Necessary and the Attendant Risks  
The provided eCommerce dataset does not contain an explicit **default** field (i.e., an observed loan default). To train supervised models, we must therefore create a **proxy outcome** derived from observable behavior—for example, customers with chargebacks/refunds within *X* days, or customers in the worst *X*% of RFM metrics.  
Using a proxy enables model building, but introduces business risk: the proxy may be an imperfect or biased signal of true default, leading to **misclassification** (false acceptances or false rejections). Therefore, we must (1) document how the proxy is created, (2) assess sensitivity to different proxy thresholds, and (3) treat model outputs as one input to credit decisions rather than the sole decision-making tool.

### Tradeoffs: Simple Interpretable Model vs. Complex High-Performance Model  
Simple models (e.g., Logistic Regression with Weight-of-Evidence encoding) offer **strong interpretability**, easy regulatory explanations, and simpler validation pipelines. They often perform competitively when features are well engineered and stable.  
Complex models (e.g., Gradient Boosting Machines) can achieve **higher predictive accuracy** and capture nonlinear interactions, but are harder to explain and require stronger governance—such as robust feature importance analysis, SHAP explainers, extensive monitoring, and stricter validation to prevent overfitting to dataset artifacts.  
In a regulated environment, a pragmatic approach is to evaluate both: use an interpretable model as the primary solution for decisioning, and a high-performance model as a benchmark or second opinion—while documenting why and when a less interpretable model may be used.

### Planned Proxy for This Project (Initial Proposal)  
For this challenge, we will define a proxy default label using **RFM-style customer behavior** and short-term negative events. Specifically:  
- Customers who fall into the **bottom decile (worst 10%)** on a combined RFM score, **or**  
- Customers who experience a **chargeback/refund within 90 days**  
will be labeled as **high risk (bad)**.  
We will also test sensitivity at several thresholds (5%, 10%, 20%) and report results to justify the final choice.
