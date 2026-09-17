"""The retention-assistant agent: a bounded tool-calling loop."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .. import config
from .llm import ChatModel
from .tools import ToolRegistry

SYSTEM_PROMPT = """You are TelcoCare AI, an assistant for the customer retention team of NovaTel, a telecom operator.

You help agents understand why a customer may churn and what to do about it.

Pick the tool by question type:
- "Why is customer X at risk?" -> predict_churn(X)
- "What should I offer customer X?" / campaign for X -> recommend_campaigns(X)
- "Can I give campaign C to customer X?" -> check_campaign_eligibility(X, C)
- General rules, tariffs, billing, troubleshooting (no customer ID) -> search_knowledge_base
- "Which customers are riskiest?" -> find_high_risk_customers

How to answer:
- Do not state the risk level, probabilities or prices yourself: they are added by the system.
- Explain drivers in plain business language.
- Campaigns: mention ONLY codes returned by a tool. Never list campaigns that were not recommended or are not eligible.
- If check_campaign_eligibility says NO, clearly say it cannot be offered and why.
- Cite documents as [retention_campaigns.md] etc.
- If a tool returns an error, read it and try the right tool. Never guess.
- Reply in the user's language (Turkish or English). Be concise: at most ~8 bullet points.

Turkish terminology (use exactly these words):
- churn -> "müşteri kaybı" (churn risk -> "kayıp riski", churn probability -> "ayrılma olasılığı")
- high / medium / low risk -> "yüksek / orta / düşük risk"
- month-to-month contract -> "aylık sözleşme"; one-year / two-year contract -> "1 yıllık / 2 yıllık sözleşme"
- electronic check -> "elektronik çek"; campaign -> "kampanya"; add-on -> "ek hizmet"
"""


@dataclass
class ToolTrace:
    name: str
    arguments: dict
    result: dict | str
    duration_ms: int


@dataclass
class AgentResponse:
    answer: str
    trace: list[ToolTrace] = field(default_factory=list)
    steps: int = 0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "steps": self.steps,
            "trace": [t.__dict__ for t in self.trace],
        }


class RetentionAgent:
    def __init__(self, llm: ChatModel, tools: ToolRegistry, max_steps: int = config.AGENT_MAX_STEPS):
        self.llm, self.tools, self.max_steps = llm, tools, max_steps

    def run(self, question: str, history: list[dict] | None = None) -> AgentResponse:
        # The answer language is decided by code, not left to the model
        language = ("\n\nLANGUAGE: Answer ONLY in Turkish (Türkçe). Do not use English sentences."
                    if _TURKISH.search(question) else "\n\nLANGUAGE: Answer ONLY in English.")
        messages = [{"role": "system", "content": SYSTEM_PROMPT + language}]
        messages += [m for m in (history or []) if m.get("role") in ("user", "assistant")]
        messages.append({"role": "user", "content": question})

        trace: list[ToolTrace] = []
        nudged = empty_retry = script_retry = False

        # Router: when the question's structure already tells us the required tool,
        # run it up front instead of hoping the model picks it (saves 1-2 LLM calls).
        expected, args = expected_tool(question)
        if expected:
            messages.append({"role": "assistant", "content": "",
                             "tool_calls": [{"function": {"name": expected, "arguments": args}}]})
            messages.append({"role": "tool", "content": self._execute(expected, args, trace),
                             "tool_name": expected})
            canned = not_found_answer(question, trace)
            if canned:
                return AgentResponse(canned, trace, 0)
            if expected == "check_campaign_eligibility" and "eligible" in (trace[0].result or {}):
                # Verdict, rule, offer and conditions are all known exactly; LLM prose only
                # added errors ("only this campaign can be offered"), so skip the LLM.
                return AgentResponse(finalize(question, "", trace), trace, 0)

        for step in range(1, self.max_steps + 1):
            reply = self.llm.chat(messages, tools=self.tools.schemas())
            tool_calls = reply.get("tool_calls") or []
            messages.append({
                "role": "assistant",
                "content": reply.get("content", ""),
                **({"tool_calls": tool_calls} if tool_calls else {}),
            })

            if not tool_calls:
                nudge = self._grounding_nudge(question, reply.get("content", ""), trace, nudged)
                if not nudge and not reply.get("content", "").strip() and not empty_retry:
                    # Small models occasionally return an empty message
                    empty_retry = True
                    nudge = "Your reply was empty. Answer the user's question now using the tool results."
                if not nudge and _NON_LATIN.search(reply.get("content", "")) and not script_retry:
                    # Qwen occasionally drifts into Chinese mid-answer; ask once for a clean redo
                    script_retry = True
                    lang = "Turkish" if _TURKISH.search(question) else "English"
                    messages.pop()  # drop the drifted answer so it doesn't prime the retry
                    nudge = (f"Answer the question again ONLY in {lang}, using Latin letters. "
                             f"Do not use Chinese or any other language.")
                if nudge:
                    nudged = nudged or nudge.startswith("Your answer is not grounded")
                    messages.append({"role": "user", "content": nudge})
                    continue
                return AgentResponse(finalize(question, reply.get("content", ""), trace), trace, step)

            for call in tool_calls:
                fn = call.get("function", {})
                name, args = fn.get("name", ""), fn.get("arguments") or {}
                messages.append({"role": "tool", "content": self._execute(name, args, trace), "tool_name": name})

        # Step budget exhausted: ask for a final answer without tools
        messages.append({
            "role": "user",
            "content": "Tool budget reached. Answer now with the information you have.",
        })
        reply = self.llm.chat(messages, tools=None)
        return AgentResponse(finalize(question, reply.get("content", ""), trace), trace, self.max_steps + 1)

    def _execute(self, name: str, args, trace: list[ToolTrace]) -> str:
        t0 = time.perf_counter()
        output = self.tools.call(name, args)
        trace.append(ToolTrace(
            name=name,
            arguments=args if isinstance(args, dict) else {"raw": args},
            result=_safe_json(output),
            duration_ms=int((time.perf_counter() - t0) * 1000),
        ))
        return output

    @staticmethod
    def _grounding_nudge(question: str, answer: str, trace: list[ToolTrace], already_nudged: bool) -> str | None:
        """Guardrail / router: small models often pick the wrong tool or skip tools.

        The question's structure tells us which tool *must* have succeeded
        before an answer is acceptable. If it hasn't, send the model back once.
        """
        if already_nudged:
            return None
        expected, args = expected_tool(question)
        if expected is None:
            return None
        # Already called (even if it returned "not found"): repeating it won't help
        if expected in {t.name for t in trace}:
            return None
        call = f"{expected}({', '.join(f'{k}={v!r}' for k, v in args.items())})"
        return (f"Your answer is not grounded yet. Ignore previous guesses and call {call} now, "
                f"then answer using only its result.")


def expected_tool(question: str) -> tuple[str | None, dict]:
    """Deterministic routing from the question's *structure*, not from topic keywords.

    Keyword routing ("campaign", "offer"...) failed on held-out phrasings such as
    "how do I keep this customer?", so the rules below only look at entities:
    - customer ID + campaign code -> eligibility check
    - customer ID                 -> recommend_campaigns (includes risk + drivers)
    - no ID, list/ranking request -> find_high_risk_customers (count + contract parsed)
    - no ID, anything substantive -> knowledge-base search
    """
    ids = _CUSTOMER_ID.findall(question)
    codes = _CAMPAIGN_CODE.findall(question.upper())
    if ids and codes:
        return "check_campaign_eligibility", {"customer_id": ids[0], "campaign_code": codes[0]}
    if ids:
        return "recommend_campaigns", {"customer_id": ids[0]}
    if _LIST_INTENT.search(question):
        return "find_high_risk_customers", _list_args(question)
    if len(_WORDS.findall(question)) < 3:
        return None, {}
    return "search_knowledge_base", {"query": question}


def _list_args(question: str) -> dict:
    """Extract 'how many' and contract filter from a ranking request."""
    from .tools import _normalize_contract

    n = re.search(r"\b(\d{1,2})\b", question)
    args: dict = {"limit": int(n.group(1)) if n else 10}
    q = question.lower()
    for hint in ("month-to-month", "monthly", "aylık", "aylik", "two-year", "two year", "2 yıllık",
                 "iki yıllık", "one-year", "one year", "1 yıllık", "bir yıllık"):
        if hint in q:
            args["contract"] = _normalize_contract(hint)
            break
    return args


_RISK_TR = {"high": "yüksek", "medium": "orta", "low": "düşük"}
KB_MAX_SENTENCES = 2
# Turkish letters, or common Turkish words for questions typed without them
# (no global re.I: it folds "İ" to "i" and would flag every English sentence)
_TURKISH = re.compile(
    r"[çğıöşüÇĞİÖŞÜ]|(?i:\b(ve|mi|mu|ile|icin|neden|hangi|nasil|musteri|verilebilir|birlikte|nedir|var|yok)\b)")


def finalize(question: str, answer: str, trace: list[ToolTrace]) -> str:
    """Post-process the LLM answer so facts that code already knows cannot be wrong.

    Division of labour: the LLM explains *why*; code renders *what*.
    - Campaign lines: when a campaign tool ran, lines of LLM text that mention campaign
      codes are removed (the small model mixed up codes and details), and the exact
      list is rendered from the rules instead.
    - Risk line: prepended if the answer lacks the probability.
    - Turkish terminology: known mistranslations are normalised.
    - Sources: appended if documents were used but none is cited.
    """
    # Last resort for script drift that survived the retry: drop non-Latin lines
    answer = "\n".join(ln for ln in answer.strip().splitlines() if not _NON_LATIN.search(ln)).strip()
    turkish = bool(_TURKISH.search(question))
    results = [(t.name, t.result) for t in trace if isinstance(t.result, dict) and "error" not in t.result]
    rec = next((r for n, r in results if n == "recommend_campaigns"), None)

    if rec is not None:
        # Customer card: the LLM contributes reasons only. Drop its lines that carry campaign
        # codes, percentages or prices (it mixed up levels, e.g. "High risk (47%)" for a
        # medium-risk customer, and described campaigns without codes).
        kept = [ln for ln in answer.splitlines()
                if not _CAMPAIGN_CODE.search(ln) and not _CAMPAIGN_HEADER.search(ln)
                and not _NUMERIC_CLAIM.search(ln)]
        answer = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
        if len(_WORDS.findall(answer)) < 5 and rec.get("driver_features"):
            # Nothing useful survived the filter: state the model's drivers from code
            from .campaigns import describe_driver_tr
            head = "Risk nedenleri:" if turkish else "Main risk drivers:"
            items = [describe_driver_tr(d["feature"], d["value"]) for d in rec["driver_features"]] if turkish \
                else rec.get("drivers_increasing_risk", [])
            answer = head + "\n" + "\n".join(f"- {i}" for i in items)

    ranked = next((r for n, r in results if n == "find_high_risk_customers"), None)
    if ranked is not None and ranked.get("customers"):
        answer = _strip_lines_with_ids(answer)
        head = ("| Müşteri | Ayrılma olasılığı | Sözleşme | Süre (ay) | Aylık ücret |" if turkish
                else "| Customer | Churn probability | Contract | Tenure (mo) | Monthly charge |")
        rows = [f"| {c['customer_id']} | {c['churn_probability']:.0%} | {c['contract']} | "
                f"{c['tenure']} | ${c['monthly_charges']:.2f} |" for c in ranked["customers"]]
        answer = (answer + "\n\n" if answer else "") + "\n".join([head, "|---|---|---|---|---|", *rows])

    if turkish:
        for pattern, replacement in _TR_FIXES:
            answer = pattern.sub(replacement, answer)

    # Eligibility verdict comes from the rules; the LLM's wording may be vague
    verdict = next((r for n, r in results if n == "check_campaign_eligibility"), None)
    if verdict is not None and "eligible" in verdict:
        # The LLM tends to restate offer details under a NO verdict; keep only its prose
        answer = "\n".join(ln for ln in answer.splitlines() if not _NUMERIC_CLAIM.search(ln)).strip()
        code = verdict["campaign"]
        rule = verdict.get("rule_tr") if turkish and verdict.get("rule_tr") else verdict["rule"]
        if turkish:
            line = (f"**Karar: EVET.** {code} bu müşteriye verilebilir." if verdict["eligible"]
                    else f"**Karar: HAYIR.** {code} bu müşteriye verilemez.")
            line += f" (Kural: {rule})"
        else:
            line = (f"**Verdict: YES.** {code} can be offered to this customer." if verdict["eligible"]
                    else f"**Verdict: NO.** {code} cannot be offered to this customer.")
            line += f" (Rule: {rule})"
        if verdict["eligible"]:
            from .campaigns import BY_CODE
            c = BY_CODE[code]
            line += f"\n\n- **{code}**: {c.offer_tr if turkish and c.offer_tr else c.offer}"
            conditions = c.conditions_tr if turkish and c.conditions_tr else c.conditions
            if conditions:
                line += f"\n  - {'Teyit edilecek' if turkish else 'Confirm'}: {'; '.join(conditions)}"
        answer = f"{line}\n\n{answer}".strip()

    # Remove lead-in lines left dangling after filtering ("...kampanyalar önerilmiştir:")
    lines = answer.splitlines()
    answer = "\n".join(ln for i, ln in enumerate(lines)
                       if not (ln.rstrip().endswith(":") and (i + 1 == len(lines) or not lines[i + 1].strip())))
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()

    scored = next((r for n, r in results if n in ("predict_churn", "recommend_campaigns")), None)
    if scored and scored.get("churn_probability_pct"):
        level, pct = scored["risk_level"], scored["churn_probability_pct"].rstrip("%")
        # Always rendered by code (whole-number match avoids "2" inside "1293-BSEUN")
        if rec is not None or not re.search(rf"(%\s?{pct}(?!\d)|(?<!\d){pct}\s?%)", answer):
            line = (f"**Risk:** {_RISK_TR.get(level, level)} (%{pct} ayrılma olasılığı)" if turkish
                    else f"**Risk:** {level} ({pct}% churn probability)")
            answer = f"{line}\n\n{answer}"

    if rec and rec.get("recommended_campaigns"):
        from .campaigns import BY_CODE, describe_driver_tr

        header = "**Önerilen kampanyalar**" if turkish else "**Recommended campaigns**"
        rows = []
        for c in rec["recommended_campaigns"]:
            campaign = BY_CODE[c["code"]]
            offer = campaign.offer_tr if turkish and campaign.offer_tr else campaign.offer
            row = f"- **{campaign.code}**: {offer}"
            if turkish and c.get("why_features"):
                why = ", ".join(describe_driver_tr(w["feature"], w["value"]) for w in c["why_features"])
                row += f"\n  - Neden: {why}"
            elif c.get("why_it_fits"):
                row += f"\n  - Why: {', '.join(c['why_it_fits'])}"
            if c.get("conditions_to_confirm"):
                label = "Teyit edilecek" if turkish else "Confirm"
                conditions = campaign.conditions_tr if turkish and campaign.conditions_tr else c["conditions_to_confirm"]
                row += f"\n  - {label}: {'; '.join(conditions)}"
            rows.append(row)
        answer += "\n\n" + header + "\n" + "\n".join(rows)

    # KB prose answers: the 3B model gets the first sentences right and then appends
    # misattributed extras ("milyon yıllık sözleşme..."). Keep the first 2 sentences unless
    # the answer is a step list (troubleshooting, escalation procedures).
    is_kb_only = (any(n == "search_knowledge_base" for n, _ in results) and rec is None
                  and verdict is None and not any(n == "find_high_risk_customers" for n, _ in results))
    procedure = any(n == "search_knowledge_base" and r.get("answer_format") == "numbered_steps"
                    for n, r in results)
    if is_kb_only and not procedure and not re.search(r"^\s*(\d+[.)]|[-*•])\s", answer, re.M):
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ\[(])", answer.strip())
        answer = " ".join(sentences[:KB_MAX_SENTENCES]).strip()

    # No tool ever returns URLs, so any link is invented ("https://retail-docs.novatel.com/...")
    answer = re.sub(r"\[([^\]]+)\]\((?:https?://|#|www\.)[^)]*\)", r"[\1]", answer)
    answer = re.sub(r"(?<![\[(])https?://\S+", "", answer)

    # Currency repair: the knowledge base only has $ prices; a 3B model sometimes writes
    # "4 TL" for "$4". Rewrite only when "$N" really appears in a tool output.
    tool_text = json.dumps([r for _, r in results], ensure_ascii=False)
    answer = re.sub(r"(\d+(?:[.,]\d+)?)\s?TL\b",
                    lambda m: f"${m.group(1)}" if f"${m.group(1)}" in tool_text else m.group(0), answer)

    sources: list[str] = []
    for n, r in results:
        if n == "search_knowledge_base":
            sources += [h["source"] for h in r.get("results", [])[:2]]
        elif n in ("recommend_campaigns", "check_campaign_eligibility"):
            sources.append("retention_campaigns.md")
    sources = list(dict.fromkeys(sources))
    if sources and not any(src in answer for src in sources):
        label = "Kaynak" if turkish else "Sources"
        answer += f"\n\n_{label}: {', '.join(f'[{src}]' for src in sources)}_"

    # Evidence: quote the top passage so the agent can verify a KB answer at a glance
    kb = next((r for n, r in results if n == "search_knowledge_base"), None)
    if kb and kb.get("results") and rec is None and verdict is None:
        top = kb["results"][0]
        excerpt = re.sub(r"\s+", " ", top["text"]).strip()
        excerpt = excerpt[:280].rsplit(" ", 1)[0] + "…" if len(excerpt) > 280 else excerpt
        label = "Kaynak metin" if turkish else "Source excerpt"
        answer += f"\n\n> **{label}** ({top['source']} › {top['section'].split(' > ')[-1]}): {excerpt}"
    return answer


def not_found_answer(question: str, trace: list[ToolTrace]) -> str | None:
    """If the routed customer lookup failed, answer from code: never let the LLM improvise."""
    if not trace or not isinstance(trace[0].result, dict) or "not found" not in trace[0].result.get("error", ""):
        return None
    cid = trace[0].arguments.get("customer_id", "?")
    if _TURKISH.search(question):
        return (f"**{cid}** numaralı müşteri sistemde bulunamadı. Müşteri numarasını kontrol edin "
                f"(biçim: 1234-ABCDE). Müşteri bilgisi olmadan kampanya önerisi yapılamaz.")
    return (f"Customer **{cid}** was not found. Please check the ID (format: 1234-ABCDE). "
            f"No campaign can be recommended without a valid customer.")


# Numbers the LLM must not state on customer cards: percentages, prices and offer durations
# ("6 aylık ... ücretsiz" was written for an offer that is 6 months free + 6 months at 50%).
_NUMERIC_CLAIM = re.compile(
    r"\d\s?%|%\s?\d|\$\s?\d|\d\s?\$|\d\s?(TL|USD|dolar)\b|\b(high|medium|low) risk\b"
    r"|\b\d+\s?(ay|aylık|aylik|months?|yıl|yıllık|years?)\b", re.I)
_ID_IN_LINE = re.compile(r"\b\d{4}-[A-Z]{5}\b")


def _strip_lines_with_ids(text: str) -> str:
    """Ranked lists are rendered as a table by code; drop the LLM's own listing."""
    kept = [ln for ln in text.splitlines() if not _ID_IN_LINE.search(ln) and not _NUMERIC_CLAIM.search(ln)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


_CAMPAIGN_HEADER = re.compile(r"^\W*(önerilen kampanya|recommended campaign|kampanyalar\s*:|campaigns\s*:)", re.I)
# Known small-model mistranslations -> glossary terms (Turkish answers only)
_TR_FIXES = [
    (re.compile(r"(ayaklanma|tahliye|çürük|kayb[ıi]n?)\s+olas[ıi]l[ıi][ğg][ıi]", re.I), "ayrılma olasılığı"),
    (re.compile(r"elektronik kontrol", re.I), "elektronik çek"),
    (re.compile(r"mekanik kontrol|posta kontrol", re.I), "posta çeki"),
    (re.compile(r"\b(ayaklı|ay-ağrı|saatlik) sözleşme", re.I), "aylık sözleşme"),
]


_CUSTOMER_ID = re.compile(r"\b\d{4}-[A-Z]{5}\b")
# Cyrillic, Arabic, Japanese kana, CJK ideographs, Hangul: never valid in a TR/EN answer
_NON_LATIN = re.compile(r"[Ѐ-ӿ؀-ۿ぀-ヿ㐀-鿿가-힯]")
_CAMPAIGN_CODE = re.compile(r"RET-[A-Z0-9]+")
_LIST_INTENT = re.compile(
    r"riskiest|highest[- ]risk|most at risk|top \d+|\blist\b|en riskli|en yüksek risk|listele|sırala", re.I)
_WORDS = re.compile(r"\w+")


def _safe_json(s: str):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


def build_default_agent() -> RetentionAgent:
    """Wire the production dependencies (trained model, FAISS index, Ollama)."""
    from ..rag.vector_store import SentenceTransformerEmbedder, VectorStore
    from ..serving.predictor import get_predictor, get_store
    from .llm import OllamaChat

    vector_store = VectorStore.load(SentenceTransformerEmbedder())
    registry = ToolRegistry(get_predictor(), get_store(), vector_store)
    return RetentionAgent(OllamaChat(), registry)
