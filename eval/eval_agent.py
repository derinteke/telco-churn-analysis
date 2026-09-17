"""Evaluate the retention agent against a fixed set of cases with automatic checks.

Expected answers are computed from the deterministic parts of the system
(predictor + campaign rules), so the eval measures only what the LLM adds:
tool use, faithfulness and hallucination.

Usage:
    python -m eval.eval_agent --model qwen2.5:3b --tag iter4 [--repeats 2] [--cases c1,c2]
Results are written to eval/results/<tag>__<model>.json and a summary is printed.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src import config
from src.agent.agent import RetentionAgent
from src.agent.campaigns import CAMPAIGNS, evaluate
from src.agent.llm import OllamaChat
from src.agent.tools import ToolRegistry
from src.rag.vector_store import SentenceTransformerEmbedder, VectorStore
from src.serving.predictor import get_predictor, get_store

RESULTS_DIR = Path(__file__).parent / "results"
ALL_CODES = {c.code for c in CAMPAIGNS}
CODE_RE = re.compile(r"RET-[A-Z0-9]+")
# English words need word boundaries; Turkish negation lives in verb suffixes
# (-amaz/-emez, -ılamaz), which \b handles poorly next to non-ASCII letters.
NEGATION_RE = re.compile(
    r"\b(no|not|cannot|can't|isn't|ineligible)\b"
    r"|hayır|değil|olmaz|mümkün değil|yasak|[a-zçğıöşü]+(amaz|emez|ılamaz|ilemez|ulamaz|ülemez)", re.I)


# Mistranslations of "churn" / domain terms seen in earlier iterations
BAD_TURKISH = re.compile(r"tahliye|çürük|churning|ay-ağrı|ayaklı|ayaklanma|elektronik kontrol|mekanik kontrol|saatlik sözleşme", re.I)
TURKISH_Q = re.compile(r"[çğıöşüÇĞİÖŞÜ]|\b(mi|ve)\b")


@dataclass
class Case:
    id: str
    question: str
    checks: dict[str, Callable[[str, list], bool]] = field(default_factory=dict)


def codes_in(text: str) -> set[str]:
    return set(CODE_RE.findall(text))


# Key facts every description of a campaign must contain (TR or EN phrasing)
OFFER_FACTS = {
    "RET-LOCK24": [r"25\s?%|%\s?25"],
    "RET-LOCK12": [r"15\s?%|%\s?15"],
    "RET-SECURE": [r"\b6\b", r"ücretsiz|free|bedava", r"50\s?%|%\s?50"],
    "RET-AUTOPAY": [r"\$\s?5\b|\b5\s?\$|5 dolar"],
    "RET-NEWBIE": [r"hız|speed", r"1 ay|bir ay|one month|1 month"],
    "RET-FIBERVALUE": [r"\$\s?15\b|\b15\s?\$|15 dolar"],
}


def offer_details_faithful(answer: str, codes: list[str]) -> bool:
    """For each recommended campaign, at least one passage about it (from a mention of
    its code up to the next campaign code) must state all of its key facts."""
    spans = [(m.group(), m.start()) for m in CODE_RE.finditer(answer)]
    for code in codes:
        if code not in OFFER_FACTS:  # invented campaign code
            return False
        sections = [answer[start:(spans[i + 1][1] if i + 1 < len(spans) else len(answer))]
                    for i, (c, start) in enumerate(spans) if c == code]
        if not any(all(re.search(p, s, re.I) for p in OFFER_FACTS[code]) for s in sections):
            return False
    return True


def build_cases(registry: ToolRegistry) -> list[Case]:
    store, predictor = registry.store, registry.predictor

    def expected(cid: str):
        pred = predictor.predict_frame(store.get(cid), cid, top_k=10).to_dict()
        rec = [r["code"] for r in evaluate(store.profile(cid), pred["top_reasons"])["recommended"]]
        return pred, rec

    def called(name):
        return lambda ans, trace: any(t.name == name for t in trace)

    def no_invented_codes(ans, _):
        return codes_in(ans) <= ALL_CODES

    cases = []

    for cid, lang_q in [
        ("9237-HQITU", "{cid} numaralı müşteri neden riskli ve ona hangi kampanyayı önermeliyim?"),
        ("7590-VHVEG", "Why is customer {cid} at risk and which campaign should I offer?"),
    ]:
        pred, rec = expected(cid)
        pct = f"{pred['churn_probability']:.0%}".rstrip("%")
        cases.append(Case(
            f"offer_{cid}", lang_q.format(cid=cid),
            {
                "called_recommend_campaigns": called("recommend_campaigns"),
                "mentions_all_recommended": lambda a, t, rec=rec: set(rec) <= codes_in(a),
                "no_non_recommended_codes": lambda a, t, rec=rec: codes_in(a) <= set(rec),
                "no_invented_codes": no_invented_codes,
                "offer_details_faithful": lambda a, t, rec=rec: offer_details_faithful(a, rec),
                "states_probability": lambda a, t, pct=pct: re.search(rf"(%\s?{pct}|{pct}\s?%)", a) is not None,
            },
        ))

    # Low-risk customer: should not push offers that the rules don't recommend
    low = "1293-BSEUN"
    _, low_rec = expected(low)
    cases.append(Case(
        f"lowrisk_{low}", f"{low} numaralı müşteriye bir kampanya teklif etmeli miyim?",
        {
            "no_invented_codes": no_invented_codes,
            "only_recommended_codes": lambda a, t, rec=low_rec: codes_in(a) <= set(rec),
            "says_low_risk": lambda a, t: re.search(r"düşük|low", a, re.I) is not None,
        },
    ))

    # Ineligible campaign requested explicitly: must say no
    cases.append(Case(
        "ineligible_fibervalue", "7590-VHVEG numaralı müşteriye RET-FIBERVALUE verebilir miyim?",
        {
            "grounded": lambda a, t: any(x.name in ("recommend_campaigns", "search_knowledge_base", "check_campaign_eligibility") for x in t),
            "says_not_eligible": lambda a, t: NEGATION_RE.search(a) is not None,
            "no_invented_codes": no_invented_codes,
        },
    ))

    # Policy question answered from documents
    cases.append(Case(
        "policy_stacking", "RET-LOCK24 ve RET-LOCK12 aynı müşteriye birlikte verilebilir mi?",
        {
            "called_search": called("search_knowledge_base"),
            "says_no": lambda a, t: NEGATION_RE.search(a) is not None,
            "cites_source": lambda a, t: "retention_campaigns" in a,
        },
    ))

    top3 = [c["customer_id"] for c in
            registry.find_high_risk_customers(limit=3, contract="Month-to-month")["customers"]]
    cases.append(Case(
        "top_risk_list", "Aylık sözleşmeli en riskli 3 müşteriyi listele.",
        {
            "called_find": called("find_high_risk_customers"),
            "lists_correct_ids": lambda a, t, ids=top3: all(i in a for i in ids),
        },
    ))

    return _add_language_checks(cases)


def _add_language_checks(cases: list[Case]) -> list[Case]:
    """Global quality checks added to every case (introduced in iteration 13 after
    regex-only checks let reasoning dumps and fabricated numbers pass)."""
    for case in cases:
        turkish = bool(TURKISH_Q.search(case.question))
        if turkish:
            case.checks["no_bad_turkish_terms"] = lambda a, t: BAD_TURKISH.search(a) is None
        case.checks["no_reasoning_leak"] = lambda a, t: REASONING_LEAK.search(a) is None
        case.checks["language_matches"] = lambda a, t, tr=turkish: language_matches(a, tr)
        case.checks["numbers_grounded"] = numbers_grounded
    return cases


REASONING_LEAK = re.compile(r"<think>|^\s*(okay|alright|hmm)[,.]|\blet me (try|think|see|check)\b|\bI need to\b"
                            r"|\bthe user (asked|wants)\b", re.I | re.M)
_EN_STOP = {"the", "and", "is", "are", "for", "with", "this", "that", "customer", "which", "will", "have"}
_TR_STOP = {"ve", "bir", "bu", "için", "ile", "müşteri", "olarak", "daha", "gibi", "değil", "olan", "var"}


NON_LATIN = re.compile(r"[Ѐ-ӿ؀-ۿ぀-ヿ㐀-鿿가-힯]")


def language_matches(answer: str, turkish: bool) -> bool:
    """Prose words (outside tables/code/IDs) should be mostly in the question's language.
    Any Cyrillic/Arabic/CJK/Hangul text fails (Qwen occasionally drifts into Chinese)."""
    if NON_LATIN.search(answer):
        return False
    answer = " ".join(ln for ln in answer.splitlines() if not ln.lstrip().startswith(">"))  # quoted sources
    prose = re.sub(r"\|.*\||`[^`]*`|\[[^\]]*\]|RET-[A-Z0-9]+|\d{4}-[A-Z]{5}|\(Kural: [^)]*\)", " ", answer)
    words = re.findall(r"[a-zçğıöşü]+", prose.lower())
    en, tr = sum(w in _EN_STOP for w in words), sum(w in _TR_STOP for w in words)
    if en + tr < 3:
        return True
    return (tr >= en * 2) if turkish else (en >= tr * 2)


_NUMBER_CLAIM = re.compile(r"(?:\$\s?(\d+(?:[.,]\d+)?))|(?:(\d+(?:[.,]\d+)?)\s?%)|(?:%\s?(\d+(?:[.,]\d+)?))"
                           r"|(?:(\d+(?:[.,]\d+)?)\s?(?:TL|USD|dolar|gün|days?|ay|months?)\b)", re.I)


def numbers_grounded(answer: str, trace) -> bool:
    """Every money/percent/duration number in the answer must appear in some tool output
    (or in the rendered campaign texts). Catches '$80 discount' and '7 TL'."""
    from src.agent.campaigns import CAMPAIGNS

    results = [getattr(t, "result", t) for t in trace]
    source = json.dumps(results, ensure_ascii=False)
    # Code renders probabilities as rounded percentages (0.9551 -> 96%) and money as 100.80
    floats = [float(x) for x in re.findall(r"\d+\.\d+", source)]
    source += " " + " ".join(f"{round(f * 100)} {f:.2f} {f:g}" for f in floats)
    source += " ".join(c.offer + " " + c.offer_tr for c in CAMPAIGNS)
    if re.search(r"\d\s?TL\b", answer):  # the knowledge base has no TL prices at all
        return False
    for m in _NUMBER_CLAIM.finditer(answer):
        num = next(g for g in m.groups() if g).replace(",", ".")
        variants = {num, num.rstrip("0").rstrip(".") if "." in num else num}
        if not any(re.search(rf"(?<![\d.]){re.escape(v)}(?![\d])", source) for v in variants):
            return False
    return True


def build_heldout_cases(registry: ToolRegistry) -> list[Case]:
    """Cases NOT used while tuning iterations 1-8: new customers, new phrasings
    (some deliberately avoid the router's keywords), and new question types."""
    store, predictor = registry.store, registry.predictor

    def expected(cid: str):
        pred = predictor.predict_frame(store.get(cid), cid, top_k=10).to_dict()
        rec = [r["code"] for r in evaluate(store.profile(cid), pred["top_reasons"])["recommended"]]
        return pred, rec, f"{pred['churn_probability']:.0%}".rstrip("%")

    def has_pct(pct):
        return lambda a, t: re.search(rf"(%\s?{pct}(?!\d)|(?<!\d){pct}\s?%)", a) is not None

    def called(name):
        return lambda a, t: any(x.name == name for x in t)

    cases = []

    # H1: retention question with no offer keyword -> router can't help, model must choose
    cid = "6023-YEBUP"; _, rec, pct = expected(cid)
    cases.append(Case("h_retain_nokeyword", f"{cid} numaralı müşteriyi kaybetmemek için ne yapabilirim?", {
        "only_recommended_codes": lambda a, t, rec=rec: codes_in(a) <= set(rec),
        "mentions_a_recommended_code": lambda a, t, rec=rec: bool(codes_in(a) & set(rec)),
        "offer_details_faithful": lambda a, t: offer_details_faithful(a, sorted(codes_in(a))),
        "states_probability": has_pct(pct),
    }))

    # H2: campaign with conditions that must be surfaced
    cid = "3932-CMDTD"; _, rec, pct = expected(cid)
    cases.append(Case("h_offer_conditions_en", f"What should I offer customer {cid}?", {
        "mentions_all_recommended": lambda a, t, rec=rec: set(rec) <= codes_in(a),
        "no_non_recommended_codes": lambda a, t, rec=rec: codes_in(a) <= set(rec),
        "mentions_competitor_condition": lambda a, t: re.search(r"competitor|rakip", a, re.I) is not None,
        "states_probability": has_pct(pct),
    }))

    # H3: eligibility NO for a different campaign/customer, English-free Turkish phrasing
    cases.append(Case("h_ineligible_autopay", "1215-FIGMP için RET-AUTOPAY uygun mu?", {
        "called_check": called("check_campaign_eligibility"),
        "says_not_eligible": lambda a, t: NEGATION_RE.search(a) is not None,
    }))

    # H4: why-only question, medium risk, must not push campaigns
    cid = "6572-ADKRS"; pred, _, pct = expected(cid)
    cases.append(Case("h_why_medium_en", f"Why is {cid} flagged as risky?", {
        "states_probability": has_pct(pct),
        "says_medium": lambda a, t: re.search(r"medium|moderate", a, re.I) is not None,
        "no_invented_codes": lambda a, t: codes_in(a) <= ALL_CODES,
    }))

    # H5: billing FAQ (no customer, no campaign words)
    cases.append(Case("h_billing_late_fee", "Fatura geç ödenirse ne olur?", {
        "called_search": called("search_knowledge_base"),
        "states_fee": lambda a, t: re.search(r"\$\s?7\b|\b7\s?\$|7 dolar", a) is not None,
        "states_days": lambda a, t: re.search(r"\b10\b", a) is not None,
    }))

    # H6: troubleshooting
    cases.append(Case("h_fiber_troubleshoot", "Müşterinin fiber interneti sürekli kopuyor, ne yapmalıyım?", {
        "called_search": called("search_knowledge_base"),
        "gives_steps": lambda a, t: re.search(r"ONT|router|modem|yeniden başlat|restart|line test|hat testi|teknisyen|technician", a, re.I) is not None,
    }))

    # H7: unknown customer -> must not invent a profile or campaigns
    cases.append(Case("h_unknown_customer", "9999-ZZZZZ numaralı müşteriye hangi kampanyayı önerirsin?", {
        "no_codes": lambda a, t: not codes_in(a),
        "says_not_found": lambda a, t: re.search(r"bulunamadı|bulunmuyor|bulunamamış|kayıtlı değil|not found|yok", a, re.I) is not None,
    }))

    # H8: policy number, English
    cases.append(Case("h_discount_cap_en", "What is the maximum total discount allowed without manager approval?", {
        "called_search": called("search_knowledge_base"),
        "states_35": lambda a, t: re.search(r"35\s?%|%\s?35", a) is not None,
    }))
    return _add_language_checks(cases)


def build_test_cases(registry: ToolRegistry) -> list[Case]:
    """FINAL test set: written after iteration 9's design and never used for tuning.
    Run it once per candidate; do not change the system based on individual failures."""
    store, predictor = registry.store, registry.predictor

    def expected(cid: str):
        pred = predictor.predict_frame(store.get(cid), cid, top_k=10).to_dict()
        rec = [r["code"] for r in evaluate(store.profile(cid), pred["top_reasons"])["recommended"]]
        return rec, f"{pred['churn_probability']:.0%}".rstrip("%")

    def has_pct(pct):
        return lambda a, t: re.search(rf"(%\s?{pct}(?!\d)|(?<!\d){pct}\s?%)", a) is not None

    def has(pattern):
        return lambda a, t: re.search(pattern, a, re.I) is not None

    cases = []
    cid = "6861-XWTWQ"; rec, pct = expected(cid)
    cases.append(Case("t_leaving_en", f"Customer {cid} is thinking about leaving. Any ideas?", {
        "mentions_all_recommended": lambda a, t, rec=rec: set(rec) <= codes_in(a),
        "no_non_recommended_codes": lambda a, t, rec=rec: codes_in(a) <= set(rec),
        "offer_details_faithful": lambda a, t, rec=rec: offer_details_faithful(a, rec),
        "states_probability": has_pct(pct),
    }))
    cid = "6362-QHAFM"; rec, pct = expected(cid)
    cases.append(Case("t_info_tr", f"{cid} hakkında bilgi verir misin?", {
        "states_probability": has_pct(pct),
        "no_non_recommended_codes": lambda a, t, rec=rec: codes_in(a) <= set(rec),
        "mentions_competitor_condition": has(r"rakip|competitor"),
    }))
    cases.append(Case("t_eligible_yes_tr", "9237-HQITU müşterisine RET-SECURE verilebilir mi?", {
        "says_yes": has(r"evet|verilebilir|sunulabilir|uygun(dur)?\b|can be offered|yes"),
        "offer_details_faithful": lambda a, t: offer_details_faithful(a, ["RET-SECURE"]),
    }))
    cases.append(Case("t_paper_bill_en", "Is there a fee for paper bills?", {
        "states_2_dollars": has(r"\$\s?2\b|\b2\s?\$|2 dollars"),
    }))
    cases.append(Case("t_fiber1000_price_tr", "Fiber 1000 paketinin aylık ücreti ne kadar?", {
        "states_85": has(r"\$\s?85\b|\b85\s?\$|85 dolar"),
    }))
    cases.append(Case("t_escalation_tr", "İki teklifi de reddeden ve iptal isteyen müşteri için ne yapmalıyım?", {
        "mentions_specialist": has(r"Retention Specialist|uzman"),
    }))
    top5 = [c["customer_id"] for c in registry.find_high_risk_customers(limit=5, contract="Two year")["customers"]]
    cases.append(Case("t_top5_twoyear_en", "Show me the 5 riskiest customers on a two-year contract.", {
        "lists_correct_ids": lambda a, t, ids=top5: all(i in a for i in ids),
    }))
    cases.append(Case("t_streaming_dsl_tr", "Müşteri DSL kullanıyor ve Streaming TV sürekli takılıyor, ne önerirsin?", {
        "mentions_fiber_or_speed": has(r"fiber|25\s?Mbps"),
        "no_invented_codes": lambda a, t: codes_in(a) <= ALL_CODES,
    }))
    return _add_language_checks(cases)


def run(model: str, tag: str, repeats: int, only: set[str] | None, case_set: str = "dev") -> dict:
    registry = ToolRegistry(get_predictor(), get_store(), VectorStore.load(SentenceTransformerEmbedder()))
    agent = RetentionAgent(OllamaChat(model=model), registry)
    builder = {"dev": build_cases, "heldout": build_heldout_cases, "test": build_test_cases}[case_set]
    cases = [c for c in builder(registry) if not only or c.id in only]

    records = []
    for case in cases:
        for rep in range(repeats):
            t0 = time.time()
            try:
                res = agent.run(case.question)
                answer, trace, err = res.answer, res.trace, None
            except Exception as e:  # noqa: BLE001
                answer, trace, err = "", [], f"{type(e).__name__}: {e}"
            secs = time.time() - t0
            # Grade only what the assistant says: the code-appended source excerpt quotes the
            # document, and grading it inflated KB scores from iteration 14 to 19.
            graded = answer.split("\n\n> **")[0]
            checks = {}
            for name, fn in case.checks.items():
                try:
                    checks[name] = bool(fn(graded, trace))
                except Exception:  # noqa: BLE001 - a crashing check counts as a failure
                    checks[name] = False
            records.append({
                "case": case.id, "repeat": rep, "seconds": round(secs, 1), "error": err,
                "checks": checks, "passed": all(checks.values()),
                "question": case.question, "tools": [t.name for t in trace], "answer": answer,
                "tool_results": [t.result for t in trace],
            })
            mark = "PASS" if records[-1]["passed"] else "FAIL"
            failed = [k for k, v in checks.items() if not v]
            print(f"[{mark}] {case.id} #{rep} {secs:.0f}s tools={records[-1]['tools']} {failed or ''}")

    n_checks = sum(len(r["checks"]) for r in records)
    summary = {
        "model": model, "tag": tag, "set": case_set, "repeats": repeats,
        "case_pass_rate": round(sum(r["passed"] for r in records) / len(records), 3),
        "check_pass_rate": round(sum(sum(r["checks"].values()) for r in records) / n_checks, 3),
        "median_seconds": statistics.median(r["seconds"] for r in records),
        "prompt_temperature": config.LLM_TEMPERATURE,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{tag}__{case_set}__{model.replace(':', '-')}.json"
    out.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.OLLAMA_MODEL)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--cases", help="comma-separated case ids")
    ap.add_argument("--set", default="dev", choices=["dev", "heldout", "test"])
    a = ap.parse_args()
    run(a.model, a.tag, a.repeats, set(a.cases.split(",")) if a.cases else None, a.set)
