# Agent iteration log

Hardware: GTX 1050 Ti (4 GB VRAM), i7-7700HQ, 16 GB RAM, Ollama 0.34.1.
Eval harness: `python -m eval.eval_agent --model <m> --tag <iterN>` (6 cases, automatic checks).
Expected answers come from the deterministic parts (predictor + campaign rules), so the
eval scores only what the LLM adds: tool use, faithfulness, hallucination.

---

## Iteration 0: baseline prompt (manual test, CPU only)
- Setup: generic system prompt, tools = predict_churn / get_customer_profile / search_knowledge_base / find_high_risk_customers. NVIDIA driver 527.56 crashed CUDA discovery, so everything ran on CPU.
- qwen2.5:7b: timed out (>180 s) on the first call.
- qwen2.5:3b (~50 s): called predict_churn only, **never searched the KB**, wrote the probability as "%8383", broken Turkish, no campaign recommended.
- Learning: on CPU, 7B is unusable. Small models mangle raw floats.

## Iteration 1: workflow prompt + formatted probability (manual, CPU)
- Change: numbered workflow in the system prompt (predict → search KB → answer); tool returns `churn_probability_pct` and `risk_increasing_drivers`.
- qwen2.5:3b (163 s): followed the workflow and searched the KB, but **recommended RET-FIBERVALUE, which the customer is ineligible for** ($70.7 < $80), listed 3 campaigns (max 2), and skipped the probability.
- Learning: the LLM retrieves the right document but cannot apply numeric eligibility or stacking rules.

## Iteration 2: GPU enabled (manual)
- Change: NVIDIA driver updated to 582.66 (the first attempt failed: C: drive was full). The 3B model now runs 100% on GPU at 28 tok/s (vs 7.6 on CPU). 7B is split 55/45 CPU/GPU at 4.6 tok/s.
- Both models **skipped the KB and invented campaigns** ("Retention Campaign Code 12345"). The prompt alone is not reliable: the same prompt behaved differently from iteration 1.
- Learning: tool use must not depend on the model's goodwill.

## Iteration 3: deterministic campaign engine + grounding guardrail (manual, GPU)
- Change: `src/agent/campaigns.py` encodes the eligibility and stacking rules and ranks campaigns by the customer's SHAP drivers. New tool `recommend_campaigns`. The agent loop sends the model back once if it talks about offers without calling a grounding tool.
- qwen2.5:3b (78 s): correct codes (RET-LOCK12, RET-SECURE), no invented codes, but **invented campaign contents** ("1 month free trial"), printed SHAP impacts as percentages, repeated bullet points.
- qwen2.5:7b (211 s): correct contents for RET-LOCK12, but **omitted RET-SECURE** and the probability.
- Learning: codes are now correct; remaining errors are in *describing* the offers.

## Iteration 3b: generation cap (eval harness, qwen2.5:3b, 2 repeats)
- Root cause found for the "hangs": the model got into a repetition loop and generated **>14,000 tokens** for one answer (no `num_predict` limit). The earlier timeouts and the repeated bullets were the same bug.
- Change: `num_predict=700`, `repeat_penalty=1.1`.
- **Result: case pass 25% | check pass 61.9% | median 22 s**. No more runaway generations.
- Failures:
  1. `recommend_campaigns` also returned `other_eligible` and `not_eligible`, and the model listed those codes (offer and low-risk cases).
  2. "Can I give RET-FIBERVALUE?": the model ignored the question and printed generic recommendations; no tool answers "is X eligible".
  3. Policy question: the model called `recommend_campaigns(customer_id="RET-LOCK")`, got "not found" and gave up. The guardrail counted that failed call as grounded.
  4. English case: probability omitted.

## Iteration 4: slim tool outputs + eligibility tool + routing prompt (qwen2.5:3b, 2 repeats)
- Changes:
  - `predict_churn` returns only pre-formatted fields (`churn_probability_pct`, plain-language drivers), with no raw floats or SHAP numbers.
  - `recommend_campaigns` returns only the recommended campaigns with a one-line `customer_gets`, plus an explicit instruction. `other_eligible` and `not_eligible` were removed.
  - New tool `check_campaign_eligibility(customer_id, code)` returns a YES/NO verdict with the rule.
  - Not-found errors tell the model which tool to use instead. Failed tool calls no longer count as grounding.
  - The system prompt maps question types to tools.
- **Result: case pass 41.7% (+16.7 pts) | check pass 71.4% (+9.5 pts) | median 13 s (−9 s)**.
- All 4 offer runs pass (they were 1/4).
- **Eval harness bugs found** (not model errors): the `grounded` check didn't include `check_campaign_eligibility`, and the negation regex missed Turkish verb forms like "kullanamaz". Fixed for the next iteration, so iter5 numbers are not strictly comparable on those two checks.
- Real model failures:
  1. Low-risk question (no code in it): the model invented RET-LOCK24, called `check_campaign_eligibility`, got NO, and **offered it anyway**.
  2. Policy question (no customer ID): the model called `check_campaign_eligibility` with a customer ID, probably copied from the example in the prompt, instead of `search_knowledge_base`.
  3. Top-risk list: once answered "not found", likely because the contract filter was not in English.

## Iteration 5: deterministic router guardrail + contract normalization (qwen2.5:3b, 2 repeats)
- Changes:
  - The guardrail became a router. `expected_tool(question)` derives the required tool from the question's structure (customer ID + campaign code → eligibility check; ID + offer words → recommend; no ID + campaign/offer words → KB search). The model is sent back once if that tool hasn't succeeded.
  - `find_high_risk_customers` accepts Turkish or loose contract names ("Aylık" → Month-to-month).
  - Eval harness fixes from iter4 (grounded check, Turkish negation suffixes).
- **Result: case pass 75% (+33.3 pts) | check pass 92.9% (+21.5 pts) | median 14 s**.
- Eligibility question 2/2, top-risk list 2/2, English offer case 2/2.
- Remaining failures:
  1. Probability omitted once (Turkish offer case).
  2. Low-risk customer: said "no campaign recommended", then **invented vague discounts** ("25% off") and didn't say "low risk".
  3. Policy question: correct answer but no source citation once. Also, before the router corrected it, the model **wasted 2 tool calls** (`check_campaign_eligibility` with made-up IDs), so the question took 31 s.
- Learning: the router fixes *which* tool, but only after the model has already wasted steps. Facts that code already knows (risk line, sources) are still at the mercy of the model.

## Iteration 6: route first, finalize deterministically (qwen2.5:3b, 2 repeats)
- Changes:
  - The router **executes the expected tool before the first LLM call** and injects the result as a tool message, so the model no longer has to pick it.
  - `finalize()`: if a scoring tool ran and the answer lacks the probability, code prepends a risk line. If documents were used and none is cited, code appends the sources.
  - Stronger "no campaign" instruction in `recommend_campaigns`.
  - Bugs found by unit tests along the way: (a) language detection missed Turkish written without special letters ("... verilebilir mi?"); (b) `re.I` folds "İ" to "i", so every English sentence was detected as Turkish.
- **Result: case pass 83.3% (+8.3 pts) | check pass 95.2% (+2.3 pts) | median 10 s (−4 s)**.
- No wrong tool choices at all. Policy question 31 s → 17 s. Offer questions need a single tool call.
- Remaining failure (low-risk, 2/2): the behaviour is right (no campaign offered), but:
  1. **Bug in finalize**: the probability check was a substring test. The probability is "2%" and the customer ID "1293-BSEUN" contains "2", so the risk line was never added.
  2. Turkish terminology: "churn" was translated as "tahliye" (eviction) and "çürük" (rotten).

## Iteration 7: fix finalize bug + Turkish glossary (qwen2.5:3b, 3 repeats)
- Changes:
  - Probability check in `finalize()` matches whole numbers with a % sign, so "2" inside "1293-BSEUN" no longer counts (regression test added).
  - Turkish terminology glossary in the system prompt (churn → "müşteri kaybı / ayrılma olasılığı", etc.). Risk-line wording aligned.
  - New eval check `no_bad_turkish_terms` on Turkish questions. Rescored saved answers: iter5 had 3 answers with such terms, iter6 had 2 (both in the already-failing low-risk case), so the iter6 score is unchanged.
- **Result: case pass 94.4% (17/18) | check pass 98.7% | median 8.6 s**.
- 0 bad Turkish terms. The low-risk case now gets the risk line from code.
- The only failure is a **grader false negative**: the answer was "yüksek bir kaybı riski yoktur (kayıp riski: 2%)", which is correct, but the check looks for the word "düşük". I did not loosen the check to fit this one answer.
- **Eval gap found by reading answers**: offer *details* are not checked. RET-SECURE was described as "6 months Tech Support at 50% off" (actually: free for 6 months, then 50% off for 6 months). Low-risk answers add filler advice ("check registration procedures").

## Iteration 8: campaign details rendered by code (qwen2.5:3b, 3 repeats)
- Changes:
  - New eval check `offer_details_faithful`: every recommended campaign must be described with its key facts (e.g. RET-SECURE: 6 months, free, 50%). Rescoring saved answers: iter6 4/4, iter7 5/6. It automatically caught the RET-SECURE mistake found by reading. Grader fix: facts may appear in any passage about the code, not only the first mention.
  - Campaigns got `offer_tr`. `finalize()` appends a "Recommended campaigns" list rendered from the rules in the user's language. The tool now tells the LLM to explain *why* and not to write discounts or durations.
- **Result: case pass 100% (18/18) | check pass 100% | median 10.4 s**.
- **The eval set is now saturated. This does not mean the agent is perfect**:
  - New mistranslation: "84% **ayaklanma** olasılığı" ("uprising"). It isn't in the bad-terms list, so it passed.
  - Low-risk answers still contain filler ("başka bir yolculuk yapabilirsiniz").
  - The same 6 questions were used to tune 8 iterations, so there is a real risk of overfitting to them.
- Next: build a **held-out eval set** (new customers, new phrasings, not used for tuning) to measure generalization.

## Held-out eval of the iteration 8 system (qwen2.5:3b, 3 repeats, 8 new cases)
- New set `--set heldout`: new customers, new phrasings (some deliberately avoid the router's keywords), and new question types (billing FAQ, troubleshooting, unknown customer, a policy number in English).
- **Result: case pass 66.7% (16/24) | check pass 85.2% | median 10 s**, vs **100%** on the dev set. **This confirms overfitting to the 6 dev cases.**
- The grader crashed on its first run: the model invented a campaign code "RET-HIGHREVENUE", which the facts table didn't know. Fixed: unknown codes fail the check, and a crashing check counts as FAIL.
- Failures:
  1. "Müşteriyi kaybetmemek için ne yapabilirim?" (0/3): no router keyword, so the model only called `predict_churn` and never recommended campaigns (one run invented RET-HIGHREVENUE). **Keyword routing is brittle.**
  2. Billing question: one run returned an **empty answer** after 3 s with no tool calls.
  3. Troubleshooting: the model called `get_customer_profile` with no customer ID and asked for one (2/3 failed).
  4. "Maximum discount without approval" (1/3): **retrieval miss**. The "Stacking rules" chunk (35%) is not in the top 4. Dense search ranks tariff "contract discount" chunks higher. The one pass happened because the playbook's Escalation chunk also mentions 35%.

## Iteration 9: structural routing + hybrid retrieval + empty-reply retry
- Changes (general fixes, not per-case patches):
  - **Routing by entities, not topic keywords**: customer ID + campaign code → eligibility check; customer ID → `recommend_campaigns`, which now also returns risk drivers; no ID → KB search, except list/ranking requests and very short messages (greetings).
  - **Hybrid retrieval**: BM25 (pure Python) + dense, fused with weighted Reciprocal Rank Fusion. New `eval/eval_retrieval.py` with 15 labelled TR/EN queries:

    | method | hit@4 | MRR |
    |---|---|---|
    | dense | 0.87 | 0.787 |
    | BM25 | 0.73 | 0.573 |
    | hybrid, BM25 weight 0.25 | 0.93 | 0.832 |
    | **hybrid, BM25 weight 0.5 (chosen)** | **0.93** | **0.832** |
    | hybrid, BM25 weight 1.0 | 0.93 | 0.793 |

  - Empty LLM replies are retried once.
- The held-out set has now influenced this design, so it is no longer truly unseen. A **final `test` set** (8 new cases) was written before running iter9 and will only be run at the end, with no tuning on it.
- **Result (dev, 3 repeats): case pass 83.3% | check pass 95.2% | median 10.5 s**. All 3 failures are `offer_9237` ("ayaklanma olasılığı" 3/3, plus one run inventing "RET-LOCK12 (RET-FIBERVALUE)").
- **Result (held-out, 3 repeats): case pass 95.8% (23/24, was 66.7%) | check pass 97.5% | median 9.7 s**. The only failure: unknown customer. After "not found", the model searched the KB and listed campaigns anyway.
- Diagnosis: "ayaklanma" comes from translating the English string "84% churn probability" that `recommend_campaigns` returned. The LLM keeps writing its own campaign list even when told a list is appended, and it reintroduces errors there.

## Iteration 10: LLM explains *why*, code renders *what*
- Changes:
  - `recommend_campaigns` no longer returns an English risk sentence. The LLM is asked only for 2-4 bullets on *why* the customer is at risk.
  - `finalize()` **strips LLM lines that mention campaign codes** when the campaign tool ran, then renders the campaign list from the rules, with reasons in Turkish (`describe_driver_tr`) and conditions to confirm.
  - Turkish terminology normaliser for known mistranslations ("ayaklanma/tahliye/çürük olasılığı" → "ayrılma olasılığı", "elektronik kontrol" → "elektronik çek"...). The eval's `no_bad_turkish_terms` check now partly measures this normaliser; noted here for transparency.
  - **Unknown customer → canned answer from code**, and the LLM is not called at all.
- **Result (dev): 100% (18/18) | median 9.7 s. Held-out: case pass 87.5% (21/24) | check pass 96.3% | median 9.4 s.** Unknown customer now 3/3.
- **Reading the answers exposed real errors that the checks missed**:
  1. Medium-risk customer (47%) described as **"High risk (47%)"**. The model copied the prompt's example "High risk (84% churn probability)". The held-out check `says_medium` caught it 3/3.
  2. Mixed-language output: "High risk (84% müşteri kaybı probability)". Partly my own normaliser's fault: it replaced "churn" inside an English phrase.
  3. The model described campaigns *without* codes ("%25 indirim ... 6 ay %50"), with wrong combinations, so the code-line filter didn't catch them.

## Iteration 11: code owns every number; language is set by code
- Changes:
  - The risk line is **always** rendered by code on customer questions. LLM lines containing percentages, prices, currency, or "high/medium/low risk" are dropped from customer cards, so the LLM supplies reasons only. The prompt example with a risk level was removed.
  - The answer language is decided by code (a LANGUAGE line appended to the system prompt), and the "churn → müşteri kaybı" rewrite was removed.
  - Ranking requests are routed too: `_list_args` parses the count and contract ("en riskli 3 ... aylık", "5 riskiest ... two-year"). Code renders the result table and drops the LLM's own listing. **The model no longer needs tool calling for any routed question**, so models without good tool calling become candidates.
- **Result (dev): 100% (18/18) | median 8.9 s. Held-out: case pass 91.7% (22/24) | check pass 97.5% | median 9.7 s.**
- Medium-risk customer now correct ("medium (47% churn probability)"). The ranking answer is a clean code-rendered table.
- Remaining failures:
  1. Eligibility NO answered vaguely (didn't clearly say "cannot").
  2. Billing: "$7 late fee" became **"7 TL"**, a currency hallucination. This is a language-model capability limit; a regex can't safely fix it.

## Iteration 12: verdict rendered by code + model comparison
- Changes: the eligibility verdict line ("**Karar: HAYIR.** RET-AUTOPAY bu müşteriye verilemez. (Kural: ...)") is rendered from the rules. Dangling lead-in lines ("...önerilmiştir:") left after filtering are removed. "new customer" is no longer listed as a risk-*decreasing* driver. `think: false` is sent for reasoning models (qwen3).
- Candidates, all fitting the GTX 1050 Ti's 4 GB at least partially: qwen2.5:3b (current), **qwen3:4b** (newer, stronger multilingual), qwen2.5:7b (CPU/GPU split, slow).
- **Result (3 repeats each)**:

  | model | dev | held-out | median latency |
  |---|---|---|---|
  | qwen2.5:3b | 100% | 100% | 9 s |
  | qwen3:4b | 100% | 100% | **91–97 s** |

- **Both 100% scores were misleading**. Reading the answers showed:
  - qwen3:4b: `think: false` was not honoured, so answers were English reasoning dumps ("Okay, let me try to figure out..."). The model also spilled onto the CPU (4.1 GB with an 8k context, 42/58 CPU/GPU split). The regex checks passed because the right numbers appeared *inside the reasoning*. **Disqualified.**
  - qwen2.5:3b: billing answer correct, then drifted into unrelated contract details mixed with the $7 fee. Troubleshooting answer talked about campaigns and invented "RET-FIBERVALUE $80 discount" with no steps. The retention card lost all its reasons because the filter removed every LLM line.
- **Main lesson: the checks were too lenient.** They tested "is the right fact present", not "is anything wrong or off-language".

## Iteration 13: stricter grader + retrieval coverage scaling + focused KB answers
- **Grader (applied to every case)**:
  - `no_reasoning_leak`
  - `language_matches` (stop-word ratio of the prose)
  - `numbers_grounded`: every $ / % / days / months / TL number in the answer must appear in a tool output or in the campaign texts; any "TL" amount fails
  - Records now store the question and tool outputs.
  - Rescoring saved iter12 answers: qwen3:4b reasoning leak 39/42, language mismatch 27/42; qwen2.5:3b 0/42 on both. (`numbers_grounded` needs tool outputs, so it can't be rescored.)
- **Retrieval**: the Turkish troubleshooting query matched only "fiber" in BM25, which pushed RET-FIBERVALUE to rank 1. The BM25 weight is now scaled by *query coverage* (share of query tokens in the corpus vocabulary). Retrieval eval: **hit@4 0.93, MRR 0.865** (best fixed weight: 0.832).
- **KB answers**: results carry a rank and an instruction: answer only the question, from the most relevant passage, give steps for problems, copy numbers and currencies exactly.
- **Customer card fallback**: if no LLM reason survives the filter, code lists the risk drivers (Turkish labels).
- **Result (stricter grader): dev case pass 72.2% (13/18) | check 96.4% | median 8.5 s. Held-out case pass 95.8% (23/24) | check 99.3% | median 7.4 s.**
- Troubleshooting now gives the real steps (status dashboard → restart ONT/router → remote line test → technician). The retention card keeps its reasons.
- Failures, classified by reading:
  - **Grader false negatives**: the code-rendered ranking table (0.9551 shown as "96%") failed `numbers_grounded` 3/3; "yasaklanmıştır" (forbidden) wasn't recognised as a negation.
  - **Real errors**: "$4 mesh extender" written as **"4 TL"** (caught). Under a NO eligibility verdict the model restated offer details ("$15/maliyet indi") and the English rule text tripped `language_matches`. Billing: "service cut after **10** days" (it's 30) and an invented "%25 fee" copied from an unrelated chunk. **Not caught**, because 25 exists in another passage, so `numbers_grounded` can't detect misattribution.
- Diagnosis: the 3B model **mixes facts across passages**. Retrieval hit@k: @1 0.80, @2 0.87, @3 0.93, @4 0.93, so k=3 loses nothing.

## Iteration 14: fewer distractors, currency repair, evidence quote
- Changes:
  - `RAG_TOP_K` 4 → 3 (same hit rate, one fewer distractor).
  - **Currency repair**: "N TL" → "$N" only when "$N" exists in a tool output (the KB has no TL prices).
  - **Source excerpt**: KB answers end with a quote of the rank-1 passage, so the retention agent can verify the answer at a glance. This is a UX safety net for the misattribution errors the grader can't catch.
  - Eligibility answers: Turkish rule text (`eligibility_tr`); LLM lines with numbers are dropped under the verdict.
  - Grader fixes: rounded percentages and normalised decimals count as grounded; "yasak" counts as negation; quoted source lines are excluded from the language check. Verified on saved iter13 answers: the table now passes, "4 TL" is still caught.
- **Result (strict grader, 3 repeats): dev 100% (18/18) | held-out 100% (24/24) | median 8.8 s / 7.9 s.**
- Reading answers: eligibility answers are clean ("**Karar: HAYIR.** ... (Kural: Aylık ücreti $80 üzerinde olan ve rakip teklif almış fiber müşterisi)"). Troubleshooting gives correct ordered steps with "$4/month".
- **Still imperfect (not caught by any check)**: the billing answer gets the three facts right, then adds a contradictory summary ("10 günlük ödemeden sonra kapanması") and a misattributed "25% x ay fee". The discount-cap answer is right but adds irrelevant tariff text. **The source excerpt now shows the correct passage under each such answer.**
- Tuning stops here. Next: run the untouched final `test` set once.

## Final test set (untouched until now, run once)

| model | case pass | check pass | median latency |
|---|---|---|---|
| **qwen2.5:3b** (3 repeats, 24 runs) | **87.5%** (21/24) | 97.7% | **9.6 s** |
| qwen2.5:7b (1 repeat, 8 runs) | 87.5% (7/8) | 97.7% | 39 s |

- **Both models failed the same case for the same reason, and it was a bug in my code, not in the model**: the Turkish campaign card rendered RET-FIBERVALUE's "conditions to confirm" in English ("Customer must have received a competitor offer"). `language_matches` caught it.
- Fix: `conditions_tr` plus a regression test. Re-rendering the 4 saved failing answers with the fix: all pass `language_matches`, and their other checks had already passed.
- **Honest reporting**: the untouched test score is **87.5%**. After this localization fix the saved answers would pass, but the test set has now influenced the code, so it no longer counts as unseen.
- **Decision: `qwen2.5:3b` is the default.** Same accuracy as 7b on the test set, 4x faster, and it fits entirely in the 4 GB GPU.

## Summary of the journey

| stage | eval | case pass | median latency |
|---|---|---|---|
| iter3b (first harness run) | dev | 25% | 22 s |
| iter5 router guardrail | dev | 75% | 14 s |
| iter8 code-rendered campaigns | dev | 100% (lenient checks) | 10 s |
| iter8 on held-out | held-out | 67% ← overfitting exposed | 10 s |
| iter9 structural routing + hybrid RAG | held-out | 96% | 10 s |
| iter13 strict grader | dev / held-out | 72% / 96% | 8 s |
| iter14 | dev / held-out | 100% / 100% (strict) | 8 s |
| **final test (unseen)** | test | **87.5%** (one localization bug) | **9.6 s** |

**Key lessons**
1. A 3B model can't be trusted with numbers, eligibility rules or tool choice, but it *can* explain. Architecture: **code decides and renders facts; the LLM explains**.
2. Route by entities (customer ID, campaign code), not topic keywords. Keywords overfit to the dev phrasings.
3. An eval saturates quickly. A held-out set exposed overfitting (100% → 67%), and reading answers repeatedly exposed lenient checks. **Keep a final test set you never tune on.**
4. Several "model failures" were grader bugs or my own bugs (substring match, `re.I` + "İ", English conditions). Classify every failure before fixing it.
5. Remaining known weakness: in free-text KB answers the 3B model sometimes adds a misattributed sentence after the correct facts. Mitigated with fewer passages (k=3) and a quoted source excerpt under each answer.

## Iteration 15: adaptive relevance cut + short KB answers (after the final test)
- An end-to-end API check showed the known KB weakness clearly: "Geç ödeme ücreti ne kadar...?" got the right first sentence ($7, 10 days), then invented "4 months" and a "$5 coupon" taken from the RET-AUTOPAY chunk. **This is not acceptable for "answers must be correct".**
- Measurement (15 retrieval queries), with passages kept if cosine ≥ best − margin:

  | margin | recall of correct section | passages sent |
  |---|---|---|
  | none (k=3) | 0.93 | 3.00 |
  | 0.20 | 0.93 | 2.53 |
  | **0.15 (chosen)** | **0.93** | **2.20** |
  | 0.12 | 0.87 | 2.00 |
  | 0.08 | 0.80 | 1.73 |

- Changes: `RAG_RELEVANCE_MARGIN = 0.15` in `search_knowledge_base`. The KB instruction now also says "at most 4 sentences, don't add information that wasn't asked".
- Evaluated on dev + held-out only (the test set is no longer unseen).
- **Result: dev 100% (18/18) | held-out 100% (24/24) | test (seen) 100% (24/24) | median 8.2 / 5.9 / 8.0 s.**
- Billing answers now get exactly one passage and are correct and concise ($7 after 10 days, suspension after 30, no reconnection fee within 60).

## Iteration 16: end-to-end findings → no invented links, LLM-free eligibility answers
- An end-to-end API check after iter15 found:
  1. An **invented URL** ("https://retail-docs.novatel.com/billing_faq"). No tool returns URLs.
  2. A YES eligibility answer where the LLM added "only this campaign can be offered" (false) and repeated bullets.
- Changes:
  - `finalize()` removes every link and bare URL, keeping the link text.
  - **Eligibility questions are answered entirely by code** (verdict, rule, offer, conditions to confirm) with no LLM call, which makes them faster too.
- **Result (3 repeats): dev 100% | held-out 100% | test (seen) 100% | median 8.2 / 5.8 / 7.5 s.** Eligibility answers now take 0 s (no LLM call).
- **But a scan of all answers found script drift**: in 1 of 24 test runs, "Fiber 1000 paketinin aylık ücreti ne kadar?" was answered partly in **Chinese** ("请用中文回答。Fiber 1000套餐的月费是85美元..."), with details taken from the wrong chunk (RET-FIBERVALUE). **Every check passed it**: `language_matches` only counted Latin stop words, and $85 was correct. Scanning all saved runs found non-Latin text in **5 answers** in total (including one from qwen2.5:7b and one from qwen3:4b).

## Iteration 17: script-drift guard + better multilingual embeddings
- Grader: `language_matches` fails on any Cyrillic, Arabic, CJK or Hangul text.
- Agent: if the reply contains non-Latin script, the drifted reply is removed from context and the model is asked once to answer only in TR/EN. As a last resort, `finalize()` drops non-Latin lines. Regression test added.
- Retrieval eval grew to 20 queries (5 price/tariff/billing queries added; one comes from the test-set failure above). Embedding comparison (`eval/compare_embeddings.py`, hybrid search as in production):

  | embedding model | hit@1 | hit@3 | MRR | ms/query |
  |---|---|---|---|---|
  | paraphrase-multilingual-MiniLM-L12-v2 (previous) | 0.65 | 0.90 | 0.786 | 21 |
  | multilingual-e5-small | 0.70 | 0.95 | 0.833 | 23 |
  | **paraphrase-multilingual-mpnet-base-v2 (chosen)** | **0.85** | 0.90 | **0.885** | 53 |

  hit@1 matters most because the relevance cut usually sends only the top 1-2 passages.
- The relevance margin was re-tuned for mpnet's score scale:

  | margin | MiniLM recall / passages | mpnet recall / passages |
  |---|---|---|
  | none | 0.90 / 3.00 | 0.90 / 3.00 |
  | 0.15 | 0.90 / 2.25 | 0.90 / 2.30 |
  | 0.10 | 0.80 / 1.95 | 0.90 / 1.95 |
  | **0.08 (chosen)** | 0.75 / 1.85 | **0.90 / 1.85** |
  | 0.05 | 0.70 / 1.70 | 0.90 / 1.60 (too close to the edge) |

- "Fiber 1000" query now ranks "Internet plans" first (0.564).
- **Result (3 repeats): dev 100% | held-out 95.8% (23/24) | test (seen) 95.8% (23/24) | median 8.0 / 5.8 / 5.9 s.** A scan of all 66 answers: **0** non-Latin text, **0** URLs, **0** "TL", **0** empty.
- Remaining error pattern, from reading: KB prose answers are right in their first sentence(s), then **append misattributed extras** ("Fiber 1000 ... $85'dir. Bu paket, milyon yıllık sözleşme olmadan ... yüksek risk segmente dahildir"). The escalation answer mixes all playbook bullets.

## Iteration 18: trim KB prose answers to 2 sentences (step lists kept)
- Change: for knowledge-base-only answers that are not step lists, keep the first `KB_MAX_SENTENCES = 2` sentences. Numbered or bulleted procedures (troubleshooting, escalation) are kept whole. The quoted source excerpt still follows.
- **Result: dev 100% | held-out 91.7% (22/24) | test (seen) 100% | median 8.5 / 5.7 / 6.2 s.** Answer scan: 0 non-Latin, 0 URL, 0 TL, 0 empty.
- The trim fixed the prose answers (billing, discount cap, paper bill are now clean 1-2 sentence answers) but **broke troubleshooting 2/3**. When the model wrote the steps as prose instead of a list, only the first step survived ("İlk adım olarak ... outage durumunu kontrol edin."). A remaining 2nd sentence can still be nonsense ("mülteciliğin olup olmadığına bağlı olarak değişebilir").

## Iteration 19: procedures get numbered steps and are never trimmed
- Change: `search_knowledge_base` detects when the rank-1 passage is a procedure (≥ 2 numbered or bulleted lines) and sets `answer_format = "numbered_steps"`. The model is told to answer as a numbered list, and `finalize()` doesn't trim such answers. Short-fact passages get "Answer in 1-2 sentences". Regression test added.
- **Reported result: dev 100% | held-out 100% | test (seen) 100% | median 8.2 / 6.5 / 6.5 s**, with 0 non-Latin / URL / TL / empty answers.
- **Grader flaw found, and some earlier numbers were inflated.** Reading answers again: the escalation answer lacked its key step ("transfer to the Retention Specialist queue") but passed, because the checks graded the **code-appended source excerpt**, which quotes the document. Rescoring saved answers on the assistant's text only (excerpt removed, current checks):

  | run | dev | held-out | test |
  |---|---|---|---|
  | iter14 | **15/18** (reported 18/18) | 24/24 | – |
  | iter17 | 18/18 | 23/24 | 23/24 |
  | iter18 | 18/18 | 22/24 | 23/24 |
  | iter19 | 18/18 | 24/24 | **21/24** (reported 24/24) |

  In iter19 the escalation case fails 3/3: the model garbled the proper noun ("4. Eşletme sekmesine transfer edilir"). Troubleshooting also still mixes an English line and even a German word ("bezahllos").

## Iteration 20: grade the assistant's words only + keep proper names untranslated
- Grader: checks run on the answer with the source excerpt removed. **Numbers from iter14 to iter19 in this log should be read with the rescoring table above.**
- KB instruction: keep names of teams, queues, documents, products and campaign codes exactly as written (e.g. "Retention Specialist queue", "Compliance team").
- **Result (assistant text only): dev 100% (18/18) | held-out 100% (24/24) | test (seen) 95.8% (23/24) | median 8.6 / 6.0 / 5.5 s.** Answer scan: 0 non-Latin, 0 URL, 0 TL, 0 empty.
- The escalation answer now keeps "Retention Specialist queue" and "Compliance". The one failure: an answer with too much English in it.
- Remaining issues are **Turkish fluency of the 3B model**, not facts: odd words ("komplimantasyon"), an unrequested trailing paragraph after the troubleshooting steps, "Fiber 1000" suggested where the doc says "Fiber upgrade".

## Where it stands (after 20 iterations)

| component | how it is made reliable | status |
|---|---|---|
| Churn score, risk level, drivers | XGBoost + SHAP, rendered by code | exact |
| Campaign recommendation / eligibility / stacking | rules engine, rendered by code; eligibility needs no LLM | exact |
| Ranking lists | parsed and rendered by code | exact |
| Unknown customer | canned answer | exact |
| Knowledge-base Q&A | hybrid retrieval (mpnet + BM25, coverage-scaled RRF) → relevance cut → 3B LLM → trimming/format, source excerpt | facts correct in evals; Turkish wording sometimes awkward |

**Next steps if higher free-text quality is needed:** a stronger generator (e.g. a 7-8B model on a GPU with ≥ 8 GB, or a hosted API) for KB answers only, or an LLM-as-judge with a stronger model to grade fluency and misattribution, which regex checks can't see.

## Post-iteration checks (end-to-end)
- API `/chat` on "9237-HQITU neden riskli ve ne önermeliyim?": the LLM described the campaigns without codes or percentages ("6 aylık Online Güvenlik ve 6 aylık Tech Destek ücretsiz"; the offer is actually 6 months free, then 6 months at 50%). The numeric filter only covered %/$. **Fix**: offer durations ("6 aylık", "12 months", "2 yıl") are also dropped from LLM lines on customer cards. Regression test added.
- **Streamlit UI fails to start** with `SessionStateStatProvider() takes no arguments`. Root cause: the venv runs **Python 3.10.0rc1**, a release candidate with a CPython bug (`@dataclass` on a `typing.Protocol` subclass loses its `__init__`). Reproduced in 5 lines. API, model, RAG and evals are unaffected. Fix: rebuild the venv on a stable Python (3.10.x final or newer).
