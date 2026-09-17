"""Tools the LLM can call. Each tool = JSON schema + a plain Python function.

The registry takes its dependencies (predictor, customer store, vector store)
as arguments, which keeps the tools unit-testable without a trained model or
an embedding model on disk.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from .. import config


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, predictor, store, vector_store):
        self.predictor, self.store, self.vector_store = predictor, store, vector_store
        self.tools = {t.name: t for t in self._build()}

    # --- tool implementations -----------------------------------------------
    def _not_found(self, customer_id: str) -> dict:
        return {"error": f"Customer '{customer_id}' not found. Customer IDs look like '9237-HQITU'. "
                         "For general policy or campaign-rule questions without a customer, "
                         "use search_knowledge_base instead."}

    def get_customer_profile(self, customer_id: str) -> dict:
        profile = self.store.profile(customer_id.strip())
        if profile is None:
            return self._not_found(customer_id)
        profile.pop("Churn", None)  # hide the label so the agent relies on the model
        return profile

    def predict_churn(self, customer_id: str) -> dict:
        from .campaigns import describe_driver

        cid = customer_id.strip()
        row = self.store.get(cid)
        if row is None:
            return self._not_found(customer_id)
        pred = self.predictor.predict_frame(row, customer_id=cid).to_dict()
        # Only pre-formatted values: small models mangle raw floats ("0.8383" -> "%8383")
        # and print SHAP log-odds as if they were percentages.
        return {
            "customer_id": cid,
            "risk_level": pred["risk_level"],
            "churn_probability_pct": f"{pred['churn_probability']:.0%}",
            "drivers_increasing_risk": [describe_driver(r) for r in pred["top_reasons"] if r["impact"] > 0],
            "drivers_decreasing_risk": [describe_driver(r) for r in pred["top_reasons"] if r["impact"] < 0
                                       and r["feature"] != "monthly_to_total_ratio"],  # "new customer" only reads as a risk factor
        }

    def recommend_campaigns(self, customer_id: str) -> dict:
        from .campaigns import describe_driver, evaluate

        cid = customer_id.strip()
        row, profile = self.store.get(cid), self.store.profile(cid)
        if row is None:
            return self._not_found(customer_id)
        pred = self.predictor.predict_frame(row, customer_id=cid, top_k=10).to_dict()
        result = evaluate(profile, pred["top_reasons"])
        by_feature = {r["feature"]: r for r in pred["top_reasons"]}

        from .campaigns import BY_CODE
        recommended = []
        for rec in result["recommended"]:
            recommended.append({
                "code": rec["code"],
                "title": rec["title"],
                "customer_gets": BY_CODE[rec["code"]].offer,
                "why_it_fits": [describe_driver(by_feature[f]) for f in rec["addresses_drivers"]],
                "why_features": [{"feature": f, "value": by_feature[f]["value"]} for f in rec["addresses_drivers"]],
                "conditions_to_confirm": rec["conditions_to_confirm"],
                "source": "retention_campaigns.md",
            })
        return {
            "customer_id": cid,
            "risk_level": pred["risk_level"],
            "churn_probability_pct": f"{pred['churn_probability']:.0%}",
            "drivers_increasing_risk": [describe_driver(r) for r in pred["top_reasons"][:5] if r["impact"] > 0],
            "driver_features": [{"feature": r["feature"], "value": r["value"]}
                                for r in pred["top_reasons"][:5] if r["impact"] > 0],
            "drivers_decreasing_risk": [describe_driver(r) for r in pred["top_reasons"][:5] if r["impact"] < 0
                                       and r["feature"] != "monthly_to_total_ratio"],  # "new customer" only reads as a risk factor
            "recommended_campaigns": recommended,
            "instruction": (
                "Write 2-4 short bullets explaining WHY this customer is at risk, using "
                "drivers_increasing_risk in plain language. Do NOT list, name or describe campaigns: "
                "the risk line and the exact campaign list are added automatically below your text."
                if recommended else
                f"No campaign is recommended: the customer is {pred['risk_level']} risk and no eligible "
                "campaign addresses their risk drivers. Say this clearly. Do NOT suggest any campaign, "
                "discount or free service."
            ),
        }

    def check_campaign_eligibility(self, customer_id: str, campaign_code: str) -> dict:
        from .campaigns import check

        profile = self.store.profile(customer_id.strip())
        if profile is None:
            return self._not_found(customer_id)
        return check(profile, campaign_code)

    def search_knowledge_base(self, query: str, top_k: int = config.RAG_TOP_K) -> dict:
        hits = self.vector_store.search(query, k=int(top_k))
        # Adaptive relevance cut: drop passages far below the best one. The 3B model mixes
        # facts across passages; see RAG_RELEVANCE_MARGIN (recall kept, ~1.85 passages
        # instead of 3, measured on eval_retrieval queries).
        if hits:
            best = max(h["score"] for h in hits)
            hits = [h for h in hits if h["score"] >= best - config.RAG_RELEVANCE_MARGIN] or hits[:1]
        procedure = bool(hits) and _is_procedure(hits[0]["text"])
        return {
            "results": [
                {"rank": i + 1, "source": h["source"], "section": h["section"], "text": h["text"]}
                for i, h in enumerate(hits)
            ],
            "answer_format": "numbered_steps" if procedure else "short",
            "instruction": (
                ("The rank-1 passage is a procedure: answer as a NUMBERED LIST of its steps, in order, "
                 "one step per line. " if procedure else
                 "Answer in 1-2 sentences. ")
                + "Keep names of teams, queues, documents, products and campaign codes exactly as "
                "written in English (e.g. 'Retention Specialist queue', 'Compliance team', 'Fiber 1000'). "
                "Answer ONLY the question that was asked, using the most relevant passage (usually "
                "rank 1). Ignore passages about other topics. Do not add related information that was "
                "not asked. Copy numbers and currencies ($) exactly; never convert them. If no passage "
                "answers the question, say so."
            ),
        }

    def find_high_risk_customers(self, limit: int = 10, contract: str | None = None) -> dict:
        from ..features import engineer

        df = self.store.df.reset_index()
        if contract:
            normalized = _normalize_contract(contract)
            if normalized is None:
                return {"error": f"Unknown contract '{contract}'. Use 'Month-to-month', 'One year' or 'Two year'."}
            df = df[df["Contract"] == normalized]
        if df.empty:
            return {"customers": []}
        X = engineer(df).drop(columns=[config.TARGET, config.ID_COL])
        df = df.assign(churn_probability=self.predictor.pipeline.predict_proba(X)[:, 1])
        top = df.nlargest(int(limit), "churn_probability")
        return {"customers": [
            {
                "customer_id": r[config.ID_COL],
                "churn_probability": round(float(r["churn_probability"]), 4),
                "contract": r["Contract"], "tenure": int(r["tenure"]),
                "monthly_charges": float(r["MonthlyCharges"]),
            }
            for _, r in top.iterrows()
        ]}

    # --- schemas --------------------------------------------------------------
    def _build(self) -> list[Tool]:
        cid = {"customer_id": {"type": "string", "description": "Customer ID, e.g. '7590-VHVEG'"}}
        return [
            Tool(
                "get_customer_profile",
                "Get a customer's account details: contract, tenure, services, charges, payment method.",
                {"type": "object", "properties": cid, "required": ["customer_id"]},
                self.get_customer_profile,
            ),
            Tool(
                "predict_churn",
                "Score a customer's churn probability with the ML model. Returns the risk level "
                "and the top drivers (features pushing risk up or down).",
                {"type": "object", "properties": cid, "required": ["customer_id"]},
                self.predict_churn,
            ),
            Tool(
                "recommend_campaigns",
                "Recommend retention campaigns for ONE customer (needs a customer ID like '9237-HQITU'). "
                "Eligibility and stacking rules are already applied. Use it when asked what to offer.",
                {"type": "object", "properties": cid, "required": ["customer_id"]},
                self.recommend_campaigns,
            ),
            Tool(
                "check_campaign_eligibility",
                "Check whether ONE specific campaign code (e.g. RET-FIBERVALUE) can be offered to a "
                "specific customer. Returns YES/NO and the rule.",
                {
                    "type": "object",
                    "properties": {**cid, "campaign_code": {"type": "string", "description": "e.g. 'RET-LOCK24'"}},
                    "required": ["customer_id", "campaign_code"],
                },
                self.check_campaign_eligibility,
            ),
            Tool(
                "search_knowledge_base",
                "Search internal documents: tariffs, retention campaigns and eligibility, the "
                "retention playbook, billing FAQ, technical troubleshooting. Use this before "
                "recommending any offer or policy.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query in English"},
                        "top_k": {"type": "integer", "description": "Number of passages (default 4)"},
                    },
                    "required": ["query"],
                },
                self.search_knowledge_base,
            ),
            Tool(
                "find_high_risk_customers",
                "List the customers with the highest predicted churn probability.",
                {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "How many customers (default 10)"},
                        "contract": {
                            "type": "string",
                            "description": "Optional filter: 'Month-to-month', 'One year' or 'Two year'",
                        },
                    },
                },
                self.find_high_risk_customers,
            ),
        ]

    # --- dispatch -------------------------------------------------------------
    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, arguments: dict | str | None) -> str:
        """Run a tool and return a JSON string. Errors are returned, never raised,
        so the LLM can read the error and recover."""
        tool = self.tools.get(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool '{name}'."})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                return json.dumps({"error": "Arguments were not valid JSON."})
        try:
            result = tool.fn(**(arguments or {}))
        except TypeError as e:
            result = {"error": f"Bad arguments for {name}: {e}"}
        except Exception as e:  # noqa: BLE001 - surface to the model
            result = {"error": f"{type(e).__name__}: {e}"}
        return json.dumps(result, ensure_ascii=False, default=str)


def _normalize_contract(value: str) -> str | None:
    """Accept loose/Turkish spellings: 'monthly', 'aylık', '1 year', 'iki yıllık'..."""
    v = value.strip().lower()
    if any(w in v for w in ("month", "aylık", "aylik")):
        return "Month-to-month"
    if any(w in v for w in ("two", "2", "iki")):
        return "Two year"
    if any(w in v for w in ("one", "1", "bir", "yıllık", "yillik", "year")):
        return "One year"
    return None


def _is_procedure(text: str) -> bool:
    """A passage with two or more numbered or bulleted lines describes steps to follow."""
    return len(re.findall(r"^\s*(\d+[.)]|[-*])\s", text, re.M)) >= 2
