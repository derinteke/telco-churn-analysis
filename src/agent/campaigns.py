"""Deterministic campaign eligibility and ranking.

Small local LLMs are unreliable at checking numeric eligibility rules
("monthly charge above $80", "tenure of at least 3 months") and at following
stacking rules. So the rules from ``knowledge_base/retention_campaigns.md``
are encoded here and the LLM only has to explain the result.

Ranking uses the model's own explanation: each campaign lists the churn
drivers it addresses, and its score is the sum of the positive SHAP impacts
of those drivers for this customer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

Profile = dict


@dataclass(frozen=True)
class Campaign:
    code: str
    title: str
    eligible: Callable[[Profile], bool]
    eligibility_text: str
    addresses: tuple[str, ...]          # driver features this campaign acts on
    conditions: tuple[str, ...] = ()    # must be confirmed by the agent with the customer
    conditions_tr: tuple[str, ...] = ()
    excludes: tuple[str, ...] = field(default_factory=tuple)
    offer: str = ""                     # what the customer gets, in one line
    eligibility_tr: str = ""            # Turkish rule text for code-rendered verdicts
    offer_tr: str = ""                  # same, in Turkish (rendered by code, never by the LLM)


def _has_internet(p: Profile) -> bool:
    return p.get("InternetService") not in (None, "No")


CAMPAIGNS: tuple[Campaign, ...] = (
    Campaign(
        "RET-LOCK24", "Upgrade to two-year contract",
        lambda p: p["Contract"] in ("Month-to-month", "One year") and p["tenure"] >= 3,
        "Month-to-month or one-year contract, tenure of at least 3 months",
        ("Contract", "MonthlyCharges", "monthly_to_total_ratio"),
        excludes=("RET-LOCK12",),
        offer="25% off the monthly fee for 24 months + free Tech Support",
        offer_tr="2 yıllık sözleşmeye geçişte 24 ay boyunca aylık ücrette %25 indirim + ücretsiz Tech Support",
        eligibility_tr="Aylık veya 1 yıllık sözleşme ve en az 3 aylık müşteri süresi",
    ),
    Campaign(
        "RET-LOCK12", "Upgrade to one-year contract",
        lambda p: p["Contract"] == "Month-to-month",
        "Month-to-month contract",
        ("Contract", "monthly_to_total_ratio"),
        excludes=("RET-LOCK24",),
        offer="15% off the monthly fee for 12 months",
        offer_tr="1 yıllık sözleşmeye geçişte 12 ay boyunca aylık ücrette %15 indirim",
        eligibility_tr="Aylık sözleşme",
    ),
    Campaign(
        "RET-SECURE", "Free security & support bundle",
        lambda p: _has_internet(p) and (p["OnlineSecurity"] == "No" or p["TechSupport"] == "No"),
        "Internet customer without Online Security or Tech Support",
        ("OnlineSecurity", "TechSupport", "num_services", "InternetService"),
        offer="Online Security + Tech Support free for 6 months, then 50% off for 6 months",
        offer_tr="Online Security + Tech Support 6 ay ücretsiz, sonraki 6 ay %50 indirimli",
        eligibility_tr="Online Security veya Tech Support ek hizmeti olmayan internet müşterisi",
    ),
    Campaign(
        "RET-AUTOPAY", "Automatic payment incentive",
        lambda p: p["PaymentMethod"] in ("Electronic check", "Mailed check"),
        "Pays by electronic check or mailed check",
        ("PaymentMethod", "PaperlessBilling"),
        offer="$5/month credit for 12 months after switching to automatic payment",
        offer_tr="Otomatik ödemeye geçişte 12 ay boyunca aylık $5 kredi",
        eligibility_tr="Elektronik çek veya posta çeki ile ödeme",
    ),
    Campaign(
        "RET-NEWBIE", "First-year care package",
        lambda p: p["tenure"] <= 12,
        "Tenure of 12 months or less",
        ("tenure", "tenure_group", "monthly_to_total_ratio", "TotalCharges", "charge_per_tenure"),
        offer="Welcome call, 1 month free, free speed upgrade for 3 months",
        offer_tr="Hoş geldin araması, 1 ay ücretsiz, 3 ay ücretsiz hız yükseltmesi",
        eligibility_tr="Müşteri süresi 12 ay veya daha az",
    ),
    Campaign(
        "RET-FIBERVALUE", "Fiber price match",
        lambda p: p["InternetService"] == "Fiber optic" and p["MonthlyCharges"] > 80,
        "Fiber optic customer with monthly charge above $80 who received a competitor offer",
        ("InternetService", "MonthlyCharges"),
        conditions=("Customer must have received a competitor offer",
                    "Requires a one-year contract",
                    "Team Lead approval if combined with another campaign"),
        conditions_tr=("Müşteri rakip bir teklif almış olmalı",
                       "1 yıllık sözleşme gerekir",
                       "Başka bir kampanyayla birlikte verilirse Takım Lideri onayı gerekir"),
        offer="$15/month off for 12 months (requires one-year contract)",
        offer_tr="12 ay boyunca aylık $15 indirim (1 yıllık sözleşme gerekir)",
        eligibility_tr="Aylık ücreti $80 üzerinde olan ve rakip teklif almış fiber müşterisi",
    ),
)
BY_CODE = {c.code: c for c in CAMPAIGNS}
MAX_CAMPAIGNS = 2  # stacking rule


def evaluate(profile: Profile, reasons: list[dict], max_campaigns: int = MAX_CAMPAIGNS) -> dict:
    """Return eligible/ineligible campaigns and a stacking-compliant recommendation.

    ``reasons`` are the predictor's top drivers: [{"feature", "value", "impact"}].
    """
    impact = {r["feature"]: r["impact"] for r in reasons}

    eligible, ineligible = [], []
    for c in CAMPAIGNS:
        entry = {"code": c.code, "title": c.title, "eligibility": c.eligibility_text}
        if not c.eligible(profile):
            ineligible.append(entry)
            continue
        matched = [f for f in c.addresses if impact.get(f, 0) > 0]
        entry.update(
            score=round(sum(impact[f] for f in matched), 3),
            addresses_drivers=matched,
            conditions_to_confirm=list(c.conditions),
        )
        eligible.append(entry)

    eligible.sort(key=lambda e: e["score"], reverse=True)

    recommended: list[dict] = []
    for e in eligible:
        if len(recommended) == max_campaigns or e["score"] <= 0:
            break
        if any(r["code"] in BY_CODE[e["code"]].excludes for r in recommended):
            continue
        recommended.append(e)

    return {
        "recommended": recommended,
        "other_eligible": [e for e in eligible if e not in recommended],
        "not_eligible": ineligible,
        "stacking_rules": "At most 2 campaigns; RET-LOCK24 and RET-LOCK12 cannot be combined; "
                          "total discount above 35% needs Retention Manager approval.",
    }


def check(profile: Profile, code: str) -> dict:
    """Is one specific campaign allowed for this customer? Explains why not."""
    code = code.strip().upper()
    c = BY_CODE.get(code)
    if c is None:
        return {"error": f"Unknown campaign '{code}'. Valid codes: {', '.join(BY_CODE)}."}
    ok = bool(c.eligible(profile))
    facts = {k: profile.get(k) for k in
             ("Contract", "tenure", "InternetService", "MonthlyCharges", "PaymentMethod",
              "OnlineSecurity", "TechSupport")}
    return {
        "campaign": code,
        "eligible": ok,
        "verdict": (f"YES, {code} can be offered." if ok else f"NO, {code} cannot be offered to this customer."),
        "rule": c.eligibility_text,
        "rule_tr": c.eligibility_tr,
        "customer_facts": facts,
        "offer": c.offer,
        "conditions_to_confirm": list(c.conditions),
    }


# Business-friendly names for model features (used in agent-facing tool output)
FEATURE_LABELS = {
    "Contract": "contract type", "tenure": "tenure (months)", "tenure_group": "tenure band",
    "InternetService": "internet service", "OnlineSecurity": "Online Security add-on",
    "TechSupport": "Tech Support add-on", "PaymentMethod": "payment method",
    "MonthlyCharges": "monthly charge", "TotalCharges": "total billed so far",
    "monthly_to_total_ratio": "new customer (few bills so far)", "num_services": "number of add-on services",
    "charge_per_tenure": "spend per month of tenure", "PaperlessBilling": "paperless billing",
    "OnlineBackup": "Online Backup add-on", "DeviceProtection": "Device Protection add-on",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming Movies",
    "MultipleLines": "multiple lines", "PhoneService": "phone service", "has_internet": "has internet",
    "SeniorCitizen": "senior citizen", "Partner": "has partner", "Dependents": "has dependents",
    "gender": "gender",
}


def describe_driver(reason: dict) -> str:
    label = FEATURE_LABELS.get(reason["feature"], reason["feature"])
    if reason["feature"] == "monthly_to_total_ratio":
        return label
    return f"{label}: {reason['value']}"


# Turkish labels for rendering campaign reasons (code-rendered text, never LLM-written)
FEATURE_LABELS_TR = {
    "Contract": "sözleşme tipi", "tenure": "müşteri süresi (ay)", "tenure_group": "müşteri süresi",
    "InternetService": "internet hizmeti", "OnlineSecurity": "Online Security ek hizmeti",
    "TechSupport": "Tech Support ek hizmeti", "PaymentMethod": "ödeme yöntemi",
    "MonthlyCharges": "aylık ücret", "monthly_to_total_ratio": "yeni müşteri (az sayıda fatura)",
    "num_services": "ek hizmet sayısı", "PaperlessBilling": "kağıtsız fatura",
}
VALUE_TR = {"Month-to-month": "aylık", "One year": "1 yıllık", "Two year": "2 yıllık", "No": "yok",
            "Yes": "var", "Electronic check": "elektronik çek", "Mailed check": "posta çeki"}


def describe_driver_tr(feature: str, value: str) -> str:
    label = FEATURE_LABELS_TR.get(feature, FEATURE_LABELS.get(feature, feature))
    if feature == "monthly_to_total_ratio":
        return label
    return f"{label}: {VALUE_TR.get(value, value)}"
