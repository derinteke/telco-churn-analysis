# Customer Retention Playbook

## 1. Using the churn score
The churn model produces a probability between 0 and 1, and a risk level:

- **High risk** (probability at or above the model's decision threshold): contact the
  customer proactively within 48 hours. A retention offer is recommended.
- **Medium risk**: add the customer to the next nurture campaign (email/SMS) and
  review at the next contact.
- **Low risk**: no action. Upsell opportunities may be explored.

The score is accompanied by the main *drivers*, the features pushing the score up or
down. Always tailor the conversation to the drivers rather than offering a generic
discount.

## 2. Mapping drivers to actions

| Main driver | Interpretation | Recommended action |
|-------------|---------------|--------------------|
| Contract = Month-to-month | No commitment, easy to leave | RET-LOCK24, or RET-LOCK12 as a fallback |
| Low tenure (tenure_group 0-1yr) | Customer has not yet built loyalty | RET-NEWBIE |
| InternetService = Fiber optic combined with high MonthlyCharges | Price sensitivity on a premium plan | RET-FIBERVALUE, or RET-LOCK24 |
| OnlineSecurity = No / TechSupport = No | Weak ecosystem ties; problems go unresolved | RET-SECURE |
| PaymentMethod = Electronic check | Payment friction; strongly associated with churn | RET-AUTOPAY |
| Few services (num_services low) | Weak product engagement | RET-SECURE, plus a bundle conversation |
| PaperlessBilling = Yes, with billing complaints | Bill shock | Walk the customer through the bill, then RET-AUTOPAY |

## 3. Conversation guidelines
1. Open with the customer's experience, not the offer: "How has your service been lately?"
2. Confirm the pain point before proposing a solution.
3. Present at most two options. Too many choices reduce acceptance.
4. Never promise anything outside the active campaign list.
5. Log the outcome (accepted, declined, callback) and the campaign code in the CRM.

## 4. Escalation
- The customer explicitly requests cancellation and declines two offers: transfer to
  the Retention Specialist queue.
- A regulatory complaint is mentioned: transfer to the Compliance team and do not make
  offers.
- An offer requires more than a 35% total discount: Retention Manager approval is required.

## 5. Measuring success
Retention campaigns are evaluated with a hold-out control group (10% of high-risk
customers receive no proactive contact). The primary KPI is the 90-day churn rate
difference between the contacted and control groups. The secondary KPI is net revenue
retained after discount cost.
