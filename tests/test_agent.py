import json

import pandas as pd

from src.agent.agent import RetentionAgent
from src.agent.tools import ToolRegistry
from src.serving.predictor import CustomerStore, Prediction, Reason


class StubPredictor:
    threshold = 0.59

    def predict_frame(self, X, customer_id="adhoc", top_k=5):
        return Prediction(customer_id, 0.81, "high", self.threshold,
                          [Reason("Contract", "Month-to-month", 0.9)])


class ScriptedLLM:
    """Replays a fixed list of assistant messages and records what it was sent."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.replies.pop(0)


def _store():
    return CustomerStore(pd.DataFrame([{
        "customerID": "0001-ABCDE", "Contract": "Month-to-month", "tenure": 2, "Churn": 1,
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "TechSupport": "No",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 70.0,
    }]))


def _call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def test_agent_uses_tools_then_answers(kb_store):
    llm = ScriptedLLM([
        _call("search_knowledge_base", query="month-to-month retention offer"),
        {"role": "assistant", "content": "Month-to-month contract drives the risk; RET-LOCK12 fits."},
    ])
    agent = RetentionAgent(llm, ToolRegistry(StubPredictor(), _store(), kb_store))

    res = agent.run("Why is 0001-ABCDE at risk?")

    # Router ran recommend_campaigns before the first LLM call; the model then searched
    assert [t.name for t in res.trace] == ["recommend_campaigns", "search_knowledge_base"]
    assert res.trace[0].result["risk_level"] == "high"
    first_msgs = llm.calls[0]["messages"]
    assert any(m["role"] == "tool" and "81%" in m["content"] for m in first_msgs)
    assert res.answer.startswith("**Risk:** high (81%")          # added by finalize
    assert "**Recommended campaigns**" in res.answer              # rendered from rules


def test_tool_errors_are_returned_not_raised(kb_store):
    reg = ToolRegistry(StubPredictor(), _store(), kb_store)
    assert "not found" in json.loads(reg.call("predict_churn", {"customer_id": "nope"}))["error"]
    assert "Unknown tool" in json.loads(reg.call("drop_tables", {}))["error"]
    assert "Bad arguments" in json.loads(reg.call("predict_churn", {"wrong": 1}))["error"]


def test_profile_hides_label(kb_store):
    reg = ToolRegistry(StubPredictor(), _store(), kb_store)
    assert "Churn" not in json.loads(reg.call("get_customer_profile", '{"customer_id": "0001-ABCDE"}'))


def test_guardrail_sends_model_back_when_router_tool_was_skipped(kb_store):
    """If the routed tool is missing from the trace (e.g. history-only runs), nudge once."""
    from src.agent.agent import RetentionAgent as RA, ToolTrace

    nudge = RA._grounding_nudge("0001-ABCDE için kampanya?", "Offer Code 12345", [], already_nudged=False)
    assert nudge and "recommend_campaigns" in nudge
    done = [ToolTrace("recommend_campaigns", {}, {"error": "not found"}, 1)]
    assert RA._grounding_nudge("0001-ABCDE için kampanya?", "x", done, already_nudged=False) is None


def test_empty_reply_is_retried_once(kb_store):
    llm = ScriptedLLM([{"role": "assistant", "content": ""}, {"role": "assistant", "content": "Late fee is $7."}])
    agent = RetentionAgent(llm, ToolRegistry(StubPredictor(), _store(), kb_store))
    res = agent.run("Fatura geç ödenirse ne olur?")
    assert "Late fee is $7." in res.answer and len(llm.calls) == 2


def test_guardrail_ignores_questions_without_offers(kb_store):
    llm = ScriptedLLM([{"role": "assistant", "content": "Hello!"}])
    agent = RetentionAgent(llm, ToolRegistry(StubPredictor(), _store(), kb_store))
    assert agent.run("hi").answer == "Hello!" and len(llm.calls) == 1


def test_step_budget_forces_final_answer(kb_store):
    loop = [_call("get_customer_profile", customer_id="0001-ABCDE")] * 2
    llm = ScriptedLLM(loop + [{"role": "assistant", "content": "Final."}])
    agent = RetentionAgent(llm, ToolRegistry(StubPredictor(), _store(), kb_store), max_steps=2)

    res = agent.run("loop forever")

    assert res.answer == "Final."
    assert llm.calls[-1]["tools"] is None


def test_router_expected_tool():
    from src.agent.agent import expected_tool

    assert expected_tool("7590-VHVEG müşterisine RET-FIBERVALUE verebilir miyim?")[0] == "check_campaign_eligibility"
    assert expected_tool("1293-BSEUN numaralı müşteriye kampanya teklif etmeli miyim?")[0] == "recommend_campaigns"
    assert expected_tool("6023-YEBUP numaralı müşteriyi kaybetmemek için ne yapabilirim?")[0] == "recommend_campaigns"
    assert expected_tool("Fatura geç ödenirse ne olur?")[0] == "search_knowledge_base"
    assert expected_tool("hi")[0] is None
    assert expected_tool("RET-LOCK24 ve RET-LOCK12 birlikte verilebilir mi?")[0] == "search_knowledge_base"
    assert expected_tool("Aylık sözleşmeli en riskli 3 müşteriyi listele.") == (
        "find_high_risk_customers", {"limit": 3, "contract": "Month-to-month"})
    assert expected_tool("Show me the 5 riskiest customers on a two-year contract.") == (
        "find_high_risk_customers", {"limit": 5, "contract": "Two year"})


def test_contract_normalization():
    from src.agent.tools import _normalize_contract as n

    assert n("Aylık") == "Month-to-month" and n("monthly") == "Month-to-month"
    assert n("iki yıllık") == "Two year" and n("One year") == "One year"
    assert n("weekly") is None


def test_router_runs_expected_tool_before_llm(kb_store):
    llm = ScriptedLLM([{"role": "assistant", "content": "Policy says no."}])
    agent = RetentionAgent(llm, ToolRegistry(StubPredictor(), _store(), kb_store))

    res = agent.run("RET-LOCK24 ve RET-LOCK12 birlikte verilebilir mi?")

    assert [t.name for t in res.trace] == ["search_knowledge_base"]
    assert len(llm.calls) == 1                     # no wasted LLM round-trips
    assert "Kaynak:" in res.answer                 # sources appended by code


def test_finalize_adds_missing_risk_line():
    from src.agent.agent import ToolTrace, finalize

    trace = [ToolTrace("predict_churn", {}, {"risk_level": "high", "churn_probability_pct": "84%"}, 1)]
    assert finalize("müşteri neden riskli?", "Sözleşme aylık.", trace).startswith("**Risk:** yüksek (%84")
    assert finalize("why?", "It is 84% likely.", trace) == "It is 84% likely."


def test_finalize_percentage_match_ignores_digits_in_ids():
    from src.agent.agent import ToolTrace, finalize

    trace = [ToolTrace("recommend_campaigns", {}, {"risk_level": "low", "churn_probability_pct": "2%"}, 1)]
    out = finalize("1293-BSEUN müşterisine kampanya?", "1293-BSEUN için kampanya yok.", trace)
    assert out.startswith("**Risk:** düşük (%2 ")


def test_finalize_renders_campaign_details_from_rules():
    from src.agent.agent import ToolTrace, finalize

    result = {"risk_level": "high", "churn_probability_pct": "84%",
              "recommended_campaigns": [{"code": "RET-SECURE", "conditions_to_confirm": []}]}
    trace = [ToolTrace("recommend_campaigns", {}, result, 1)]
    out = finalize("müşteriye ne önermeliyim?", "RET-SECURE uygun, %84 risk.", trace)
    assert "**Önerilen kampanyalar**" in out
    assert "6 ay ücretsiz, sonraki 6 ay %50 indirimli" in out


def test_unknown_customer_is_answered_by_code_without_llm(kb_store):
    llm = ScriptedLLM([])  # any LLM call would raise IndexError
    agent = RetentionAgent(llm, ToolRegistry(StubPredictor(), _store(), kb_store))
    res = agent.run("9999-ZZZZZ numaralı müşteriye hangi kampanyayı önerirsin?")
    assert "bulunamadı" in res.answer and "RET-" not in res.answer and not llm.calls


def test_finalize_strips_llm_campaign_lines_and_fixes_terms():
    from src.agent.agent import ToolTrace, finalize

    result = {"risk_level": "high", "churn_probability_pct": "84%",
              "recommended_campaigns": [{"code": "RET-LOCK12", "conditions_to_confirm": [],
                                         "why_features": [{"feature": "Contract", "value": "Month-to-month"}]}]}
    llm_text = ("Müşteri %84 ayaklanma olasılığına sahip.\n- Sözleşme aylık.\n\n"
                "Önerilen kampanyalar:\n1. RET-LOCK12 (RET-FIBERVALUE): yanlış detay")
    out = finalize("müşteriye ne önermeliyim?", llm_text, [ToolTrace("recommend_campaigns", {}, result, 1)])
    assert "ayaklanma" not in out and "ayrılma olasılığı" in out
    assert "RET-FIBERVALUE" not in out and "yanlış detay" not in out
    assert "- **RET-LOCK12**: 1 yıllık sözleşmeye geçişte" in out and "Neden: sözleşme tipi: aylık" in out
    assert out.startswith("**Risk:** yüksek (%84") and "%84 ayrılma olasılığına sahip" not in out


def test_finalize_renders_eligibility_verdict():
    from src.agent.agent import ToolTrace, finalize

    no = {"campaign": "RET-AUTOPAY", "eligible": False, "rule": "Pays by electronic check or mailed check"}
    out = finalize("1215-FIGMP için RET-AUTOPAY uygun mu?", "Belki uygun olabilir.",
                   [ToolTrace("check_campaign_eligibility", {}, no, 1)])
    assert out.startswith("**Karar: HAYIR.** RET-AUTOPAY bu müşteriye verilemez.")


def test_turkish_campaign_card_has_no_english_conditions():
    from src.agent.agent import ToolTrace, finalize

    result = {"risk_level": "high", "churn_probability_pct": "68%",
              "recommended_campaigns": [{"code": "RET-FIBERVALUE",
                                         "conditions_to_confirm": ["Customer must have received a competitor offer"]}]}
    out = finalize("6362-QHAFM hakkında bilgi verir misin?", "", [ToolTrace("recommend_campaigns", {}, result, 1)])
    assert "Müşteri rakip bir teklif almış olmalı" in out and "competitor" not in out


def test_eligibility_question_is_answered_without_llm(kb_store):
    llm = ScriptedLLM([])  # any LLM call would raise
    agent = RetentionAgent(llm, ToolRegistry(StubPredictor(), _store(), kb_store))
    res = agent.run("0001-ABCDE müşterisine RET-NEWBIE verebilir miyim?")
    assert res.answer.startswith("**Karar: EVET.** RET-NEWBIE") and not llm.calls
    assert "1 ay ücretsiz" in res.answer


def test_finalize_removes_invented_links():
    from src.agent.agent import ToolTrace, finalize

    kb = {"results": [{"rank": 1, "source": "billing_faq.md", "section": "Late", "text": "A $7 late fee."}]}
    out = finalize("Geç ödeme?", "Ücret $7. [billing_faq.md](https://retail-docs.novatel.com/billing_faq)",
                   [ToolTrace("search_knowledge_base", {}, kb, 1)])
    assert "http" not in out and "[billing_faq.md]" in out


def test_script_drift_is_retried_then_stripped(kb_store):
    drifted = {"role": "assistant", "content": "Fiber 1000 aylık $85.\nFiber 1000套餐的月费是85美元。"}
    clean = {"role": "assistant", "content": "Fiber 1000 paketinin aylık ücreti $85."}
    llm = ScriptedLLM([drifted, clean])
    agent = RetentionAgent(llm, ToolRegistry(StubPredictor(), _store(), kb_store))
    res = agent.run("Fiber 1000 paketinin aylık ücreti ne kadar?")
    assert "套餐" not in res.answer and "$85" in res.answer and len(llm.calls) == 2
    # the drifted answer must not be left in the context of the retry
    assert all("套餐" not in m.get("content", "") for m in llm.calls[1]["messages"])

    from src.agent.agent import finalize
    assert "月费" not in finalize("ücret?", "Ücret $85.\n月费是85美元", [])


def test_kb_prose_answer_is_trimmed_but_step_lists_are_kept():
    from src.agent.agent import ToolTrace, finalize

    kb = {"results": [{"rank": 1, "source": "tariffs.md", "section": "Internet plans", "text": "Fiber 1000 $85"}]}
    trace = [ToolTrace("search_knowledge_base", {}, kb, 1)]
    prose = "Fiber 1000 paketinin aylık ücreti $85'dir. Bu paket 1 Gbps hızdadır. Milyon yıllık sözleşme gerekir."
    out = finalize("Fiber 1000 ücreti ne kadar?", prose, trace)
    assert "$85" in out and "Milyon" not in out
    steps = "1. ONT'yi yeniden başlatın.\n2. Hat testi yapın.\n3. Teknisyen çağırın."
    assert "3. Teknisyen" in finalize("İnternet kopuyor, ne yapmalıyım?", steps, trace)


def test_procedure_passages_are_not_trimmed():
    from src.agent.agent import ToolTrace, finalize
    from src.agent.tools import _is_procedure

    passage = "1. Check outage status.\n2. Restart the ONT.\n3. Run a line test."
    assert _is_procedure(passage) and not _is_procedure("A $7 late fee is added after 10 days.")
    kb = {"results": [{"rank": 1, "source": "support.md", "section": "Fiber", "text": passage}],
          "answer_format": "numbered_steps"}
    prose = "Önce kesinti durumunu kontrol edin. Sonra ONT'yi yeniden başlatın. Son olarak hat testi yapın."
    out = finalize("İnternet kopuyor, ne yapmalıyım?", prose, [ToolTrace("search_knowledge_base", {}, kb, 1)])
    assert "hat testi" in out


def test_customer_card_drops_llm_offer_durations():
    from src.agent.agent import ToolTrace, finalize

    result = {"risk_level": "high", "churn_probability_pct": "84%",
              "driver_features": [{"feature": "Contract", "value": "Month-to-month"}],
              "recommended_campaigns": [{"code": "RET-SECURE", "conditions_to_confirm": []}]}
    llm = "- Aylık sözleşme kullanıyor.\n2. Güvenlik paketi: 6 aylık Online Güvenlik ücretsiz hediye edilebilir."
    out = finalize("9237-HQITU ne önermeliyim?", llm, [ToolTrace("recommend_campaigns", {}, result, 1)])
    assert "hediye" not in out and "6 ay ücretsiz, sonraki 6 ay %50 indirimli" in out
