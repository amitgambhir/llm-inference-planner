# Quality-Aware Benchmarking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend llm-inference-bench with an offline quality evaluation pipeline and a deployment advisor that recommends the best deployment by balancing latency, cost, and quality.

**Architecture:** `evaluate/run_eval.py` sends a small JSONL dataset at an inference endpoint, scores responses with DeepEval, and writes a quality sidecar JSON. `analyze/deployment_advisor.py` merges latency + quality + cost data for multiple tags into a `DeploymentProfile` contract and outputs a recommendation card. The existing pipeline is untouched.

**Tech Stack:** Python 3.8+ stdlib, `aiohttp` (existing), `deepeval` (new), pytest (test runner, installed as DeepEval transitive dep).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `evaluate/__init__.py` | Create (empty) | Makes `evaluate` importable in tests |
| `evaluate/run_eval.py` | Create | Quality evaluator CLI |
| `analyze/__init__.py` | Create (empty) | Makes `analyze` importable in tests |
| `analyze/deployment_advisor.py` | Create | Deployment decision engine |
| `datasets/chat.jsonl` | Create | 15 eval prompts, chat workload |
| `datasets/rag.jsonl` | Create | 15 eval prompts, RAG workload |
| `datasets/long_context.jsonl` | Create | 15 eval prompts, long-context workload |
| `results/quality/.gitkeep` | Create | Output directory |
| `tests/__init__.py` | Create (empty) | Test package root |
| `tests/test_run_eval.py` | Create | Unit tests for run_eval utilities |
| `tests/test_deployment_advisor.py` | Create | Unit tests for deployment_advisor |
| `pytest.ini` | Create | Pytest config: pythonpath = . |
| `requirements.txt` | Modify | Add `deepeval` |
| `README.md` | Modify | Add quality-aware benchmarking section |

---

## Task 1: Project Scaffold

**Files:**
- Create: `evaluate/__init__.py`
- Create: `analyze/__init__.py`
- Create: `tests/__init__.py`
- Create: `pytest.ini`
- Create: `results/quality/.gitkeep`
- Modify: `requirements.txt`

- [ ] **Step 1: Create package init files and test directory**

```bash
mkdir -p evaluate analyze tests results/quality
touch evaluate/__init__.py analyze/__init__.py tests/__init__.py results/quality/.gitkeep
```

- [ ] **Step 2: Create pytest.ini**

Create `pytest.ini` at repo root:

```ini
[pytest]
pythonpath = .
```

- [ ] **Step 3: Update requirements.txt**

Current content: `aiohttp>=3.8.0`

New content:
```
aiohttp>=3.8.0
deepeval>=0.21.0
```

- [ ] **Step 4: Verify pytest is available**

```bash
pip install -r requirements.txt
pytest --version
```

Expected: `pytest X.Y.Z` (installed as a DeepEval transitive dependency).

- [ ] **Step 5: Commit scaffold**

```bash
git add evaluate/__init__.py analyze/__init__.py tests/__init__.py results/quality/.gitkeep pytest.ini requirements.txt
git commit -m "feat: scaffold quality-aware benchmarking pipeline"
```

---

## Task 2: Eval Datasets

**Files:**
- Create: `datasets/chat.jsonl`
- Create: `datasets/rag.jsonl`
- Create: `datasets/long_context.jsonl`

All prompts follow the insurance/enterprise domain already established in `collect/run_bench.py`.

- [ ] **Step 1: Create `datasets/chat.jsonl`**

Each line is a JSON object. Create `datasets/chat.jsonl` with this content (15 rows):

```jsonl
{"schema_version": 1, "id": "chat_001", "workload": "chat", "prompt": "A customer reports their smart thermostat shows error code E7 and won't connect to WiFi after two restarts. They have a Home Tech Protection plan. What are the next two troubleshooting steps?", "expected": "Perform a factory reset on the thermostat, then check for a firmware update via the manufacturer app. If both steps fail, initiate a warranty replacement under the Home Tech Protection plan."}
{"schema_version": 1, "id": "chat_002", "workload": "chat", "prompt": "A policyholder wants to add a new driver aged 19 to their auto policy mid-term. What information do we need from them?", "expected": "We need the new driver's full legal name, date of birth, driver's license number, state of issue, and the date they will begin driving the insured vehicle."}
{"schema_version": 1, "id": "chat_003", "workload": "chat", "prompt": "A customer asks why their homeowner's premium increased by 18% at renewal with no claims. What are the most common reasons?", "expected": "Common reasons include general rate increases due to rising construction costs and reinsurance prices, updated property valuations, changes in the regional risk index for weather or wildfire, and inflation adjustments to dwelling coverage limits."}
{"schema_version": 1, "id": "chat_004", "workload": "chat", "prompt": "An agent asks whether a commercial property policy covers business interruption losses from a cyberattack that shuts down operations but causes no physical damage.", "expected": "Standard commercial property policies typically require physical damage to trigger business interruption coverage. A cyber incident without physical damage is generally excluded. The customer would need a standalone cyber policy with a business interruption rider."}
{"schema_version": 1, "id": "chat_005", "workload": "chat", "prompt": "A claimant says their adjuster hasn't responded in five business days. What is the standard SLA and what action should we take?", "expected": "Our standard SLA for adjuster response is three business days. At five days we are out of SLA. Escalate to the adjuster's supervisor, log the escalation in the claim file, and send the claimant a written acknowledgement within 24 hours."}
{"schema_version": 1, "id": "chat_006", "workload": "chat", "prompt": "Does our commercial general liability policy cover a contractor who injures a subcontractor on the job site?", "expected": "It depends on whether the subcontractor is listed as an additional insured and whether they have their own policy. Without additional insured status, the subcontractor's injury may fall under workers' compensation for the contractor's employees only. Bodily injury to non-employees on site is typically covered under CGL premises liability, subject to exclusions."}
{"schema_version": 1, "id": "chat_007", "workload": "chat", "prompt": "A small business owner asks what the difference is between occurrence and claims-made professional liability policies.", "expected": "An occurrence policy covers incidents that happen during the policy period regardless of when the claim is filed. A claims-made policy covers claims filed during the policy period regardless of when the incident occurred, and requires a retroactive date and optional tail coverage after cancellation."}
{"schema_version": 1, "id": "chat_008", "workload": "chat", "prompt": "Customer reports water damage after a pipe burst during a winter freeze. Their policy has a Section 4.2 maintenance exclusion. Is the claim likely covered?", "expected": "Coverage depends on whether the insured can show they maintained adequate heat. If the pipe froze due to an unforeseeable maintenance failure outside their control, Section 4.3 reinstatement may apply. The adjuster should document HVAC logs, weather records, and contractor schedules before making a coverage determination."}
{"schema_version": 1, "id": "chat_009", "workload": "chat", "prompt": "An underwriter wants to know the loss ratio target for our commercial healthcare book. What is the standard target and what does the current trailing twelve-month ratio indicate if it's at 71%?", "expected": "The target loss ratio for the commercial healthcare book is 62%. A trailing twelve-month ratio of 71% is 9 points above target, indicating the book is underperforming. Underwriting should review pricing adequacy and consider rate actions or tighter conditions at renewal."}
{"schema_version": 1, "id": "chat_010", "workload": "chat", "prompt": "A cyber policyholder asks if ransomware recovery costs are covered if they never paid a ransom and recovered from backups.", "expected": "Yes. Ransomware recovery costs including forensic investigation, system restoration, and business interruption are typically covered even when no ransom is paid. The absence of a ransom payment does not affect coverage for recovery expenses under a standard cyber policy."}
{"schema_version": 1, "id": "chat_011", "workload": "chat", "prompt": "A customer's roof was damaged in a hailstorm. Their policy pays actual cash value for roofs over 20 years old. The roof is 22 years old and replacement cost is $18,000. How is the payout calculated?", "expected": "Actual cash value applies depreciation to the replacement cost. A 22-year-old roof near end of its useful life (typically 25–30 years) may receive 70–80% depreciation, resulting in an ACV payout in the range of $3,600–$5,400. The exact amount depends on the depreciation schedule in the policy."}
{"schema_version": 1, "id": "chat_012", "workload": "chat", "prompt": "Can a policyholder assign their insurance benefits to a contractor directly without our consent?", "expected": "Assignment of benefits without insurer consent is prohibited under most policy terms. Some states have AOB restriction laws that limit or ban direct contractor assignments. The policyholder should direct the contractor to submit an invoice, and we pay the policyholder directly unless we have separately authorized a direct payment arrangement."}
{"schema_version": 1, "id": "chat_013", "workload": "chat", "prompt": "What documentation is required to initiate a large loss review for a commercial property claim above $750,000?", "expected": "Claims above $750,000 require: a formal coverage opinion memorandum, senior claims officer approval, a detailed adjuster's report with photo documentation, an independent appraisal if disputed, and notification to reinsurance if the loss exceeds the treaty retention."}
{"schema_version": 1, "id": "chat_014", "workload": "chat", "prompt": "A property inspector's report from 2024 rated our HVAC contractor relationship as satisfactory. Does that affect our coverage analysis for a 2026 freeze-related pipe claim?", "expected": "Yes. A satisfactory loss control survey from 2024 that identified no exclusion-triggering risks supports the insured's argument that the maintenance failure was unforeseeable. It strengthens the case for applying the Section 4.3 reinstatement clause over the Section 4.2 maintenance exclusion."}
{"schema_version": 1, "id": "chat_015", "workload": "chat", "prompt": "A new commercial account wants to know the difference between a blanket and scheduled property limit on their CPP.", "expected": "A blanket limit applies a single aggregate coverage amount across all covered locations, allowing flexibility when one location suffers a total loss. A scheduled limit assigns a specific coverage amount to each listed location. Blanket is better when location values vary or shift; scheduled gives more precise premium allocation for stable, well-valued properties."}
```

- [ ] **Step 2: Create `datasets/rag.jsonl`**

Create `datasets/rag.jsonl`:

```jsonl
{"schema_version": 1, "id": "rag_001", "workload": "rag", "prompt": "Based on the claim file for CLM-2026-04-19388, does Section 4.2 or Section 4.3 of the policy control coverage for Acme Industrial's water damage loss?", "expected": "Section 4.3 reinstatement likely controls. The four-day repair delay was caused by contractor parts supply failure, which the insured argues was unforeseeable and outside their reasonable control. The 2025 loss control survey rated the contractor relationship as satisfactory with no exclusion risks noted, supporting the unforeseeable-event argument. The adjuster should obtain written confirmation of the parts delay from the contractor before finalizing coverage."}
{"schema_version": 1, "id": "rag_002", "workload": "rag", "prompt": "The portfolio review identifies three metals-fabrication accounts sharing the same HVAC contractor. What is the concentration risk and what action is required?", "expected": "The shared contractor (NorthernPlex Mechanical) creates vendor-level concentration risk across three accounts totaling part of the $3.1M metals book. NorthernPlex has been linked to two prior maintenance-related claims in 2024. The January action item to review shared-vendor exposure is overdue and must be completed this quarter. Risk mitigation should include requiring accounts to diversify HVAC contractors or obtain additional sublimits."}
{"schema_version": 1, "id": "rag_003", "workload": "rag", "prompt": "DataMesh Cloud Services was hit by ransomware. Their policy has a $5M aggregate and $250K SIR. The total claim is $4.2M. What is the net insurer liability?", "expected": "Net insurer liability is $3.95M. The $4.2M total claim minus the $250K self-insured retention leaves $3.95M, which is within the $5M aggregate. The regulatory fines sub-limit of $1M applies only if regulatory penalties are assessed, which depends on the outcome of the state AG investigation into notification timing."}
{"schema_version": 1, "id": "rag_004", "workload": "rag", "prompt": "Does the DataMesh policy's Section 7.4 regulatory exclusion apply given that DataMesh notified customers on day 9 and disputes whether exfiltration was confirmed?", "expected": "The exclusion is not clear-cut. DataMesh argues exfiltration was never confirmed, only encryption, so the 7-day notification clock may not have started. If the AG investigation determines exfiltration was confirmed earlier, the notification delay could trigger the exclusion. Coverage counsel should be engaged before any reservation of rights is issued."}
{"schema_version": 1, "id": "rag_005", "workload": "rag", "prompt": "Riverside Specialty Group has a loss ratio of 87% over rolling three years with three plaintiff demands above $1M. What renewal action did underwriting flag?", "expected": "Underwriting flagged Riverside in December for evaluation of non-renewal versus a rate increase with tighter policy conditions. The October 2026 renewal is the decision point. Given the 87% loss ratio against no stated target loss ratio, the account warrants a formal pricing review and assessment of whether sub-limits or exclusions can reduce exposure."}
{"schema_version": 1, "id": "rag_006", "workload": "rag", "prompt": "Cascade Pipeline Operations reported a near-miss corrosion event. What was the estimated avoided loss and what does this signal for the energy book?", "expected": "The estimated avoided loss is $8M–$22M depending on rupture location sensitivity. Cascade's proactive inspection tightening following the 2024 NTSB advisory demonstrates good risk management. However, two other transmission-line accounts in the energy book remain on older 36-month inspection cycles, which now represent unaddressed exposure that should be flagged for loss control review."}
{"schema_version": 1, "id": "rag_007", "workload": "rag", "prompt": "What are the three concentration risks the quarterly compliance review must identify?", "expected": "The top three concentration risks from this review are: (1) vendor concentration — three accounts sharing NorthernPlex Mechanical HVAC contractor with two prior related claims; (2) managed-service provider concentration — three cyber accounts sharing Aegis IT Partners, representing $1.9M aggregate premium; (3) inspection cycle risk — two energy transmission-line accounts on outdated 36-month inspection schedules post-NTSB advisory."}
{"schema_version": 1, "id": "rag_008", "workload": "rag", "prompt": "The compliance framework requires coverage of pending regulatory matters. Which case has an open regulatory matter and what is the coverage risk?", "expected": "DataMesh Cloud Services has an open state AG investigation into notification timing. If the AG finds exfiltration was confirmed before day 9, the Section 7.4 regulatory exclusion could apply and eliminate coverage for a significant portion of the $1.4M forensic and notification costs. This is a material coverage risk that should be tracked until the AG's determination."}
{"schema_version": 1, "id": "rag_009", "workload": "rag", "prompt": "The Acme claim is $1.7M and a settlement above $750K requires senior officer approval. What additional process step is required?", "expected": "Any settlement above $750K requires both senior claims officer approval and a formal coverage opinion memorandum. For Acme's $1.7M claim, this means the adjuster's preliminary 60% partial-coverage recommendation must be elevated to the senior claims officer and a written coverage opinion must be prepared before any payment or reservation of rights letter is issued."}
{"schema_version": 1, "id": "rag_010", "workload": "rag", "prompt": "Based on the portfolio review, which account is flagged for potential non-renewal and what is the upcoming decision date?", "expected": "Riverside Specialty Group is flagged for potential non-renewal or significant restructuring. The renewal date is October 2026. Underwriting flagged this in December and the decision has not yet been made. The 87% loss ratio, three high-demand suits in 24 months, and a healthcare book loss ratio of 71% against a 62% target all support expedited renewal review."}
{"schema_version": 1, "id": "rag_011", "workload": "rag", "prompt": "How does Acme Industrial's multi-line renewal value affect the claims settlement strategy?", "expected": "Acme is in renewal discussions for a multi-line package worth approximately $1.2M annually, and underwriting wants to preserve the relationship. This creates a relationship risk consideration in the claims posture. However, claims decisions must be legally grounded in policy language — the settlement posture should be driven by the coverage analysis, not retention objectives, to avoid bad faith exposure."}
{"schema_version": 1, "id": "rag_012", "workload": "rag", "prompt": "The healthcare book loss ratio is 71% trailing twelve months. What is the target and what does this suggest for the book?", "expected": "The target is 62%. A 71% actual ratio is 9 points above target, indicating systemic underperformance — not just Riverside's individual account. The book-level review should assess whether rate adequacy, case mix, or geographic concentration is driving the gap. The Q1 portfolio memo should recommend book-wide pricing action, not just account-level remediation."}
{"schema_version": 1, "id": "rag_013", "workload": "rag", "prompt": "Which prior action item from a previous review remains open as of Q1 2026 and what should happen to it?", "expected": "The January action item to 'review shared-vendor exposure across metals book' remains open and has not been completed. The Q1 review must formally reprioritize this item and assign a responsible owner with a completion date. The NorthernPlex concentration discovery makes this item time-sensitive — it should be completed within 30 days."}
{"schema_version": 1, "id": "rag_014", "workload": "rag", "prompt": "What is the cyber book's shared managed-service provider exposure and which accounts does it affect?", "expected": "Three cyber accounts besides DataMesh share Aegis IT Partners as their managed-service provider, creating a concentration of $1.9M in aggregate premium under a single MSP. If Aegis suffers a systemic failure or breach, multiple simultaneous cyber claims could result. The portfolio review should document this concentration and consider whether sublimits or MSP-diversity requirements should be imposed at renewal."}
{"schema_version": 1, "id": "rag_015", "workload": "rag", "prompt": "Summarize the recommended posture for each of the four accounts reviewed in the Q1 portfolio memo.", "expected": "Acme Industrial: partial pay with reservation pending coverage opinion on Section 4.3; investigate contractor delay documentation. DataMesh Cloud Services: pay BI and forensic costs within SIR/aggregate; monitor AG investigation before addressing regulatory exposure. Riverside Specialty Group: defend malpractice claim; escalate renewal to pricing review with non-renewal as a live option. Cascade Pipeline Operations: no open claim; commend proactive inspection; flag other energy accounts on 36-month cycles for loss control review."}
```

- [ ] **Step 3: Create `datasets/long_context.jsonl`**

Create `datasets/long_context.jsonl`:

```jsonl
{"schema_version": 1, "id": "lc_001", "workload": "long_context", "prompt": "Review the following four commercial accounts and identify portfolio-level concentration risks: (1) Acme Industrial — metals fabrication, shared HVAC contractor NorthernPlex with two prior claims; (2) DataMesh Cloud — cyber, shares Aegis IT Partners MSP with three other accounts; (3) Riverside Specialty — healthcare, 87% loss ratio, three high-value demands in 24 months; (4) Cascade Pipeline — energy, near-miss resolved, two peer accounts on old inspection cycles. What are the top three concentration risks?", "expected": "Top three: (1) HVAC vendor concentration — NorthernPlex across three metals accounts with documented prior claims; (2) MSP concentration — Aegis IT Partners across four cyber accounts representing $1.9M aggregate; (3) Inspection cycle gap — two energy pipeline accounts still on 36-month cycles following NTSB advisory that prompted Cascade to tighten to 18 months."}
{"schema_version": 1, "id": "lc_002", "workload": "long_context", "prompt": "A senior underwriter is reviewing a commercial property policy (Section 4.2 maintenance exclusion, Section 4.3 unforeseeable reinstatement) for a freeze-related pipe loss. Facts: pipe burst during -8F night; loading bay HVAC offline for four days due to contractor parts delay; loss control survey 16 months prior rated contractor as satisfactory; total loss $1.7M. Write a structured coverage analysis memo with Issue, Facts, Analysis, and Recommendation sections.", "expected": "Issue: Whether Section 4.2 maintenance exclusion bars coverage for Acme's pipe burst, or whether Section 4.3 unforeseeable reinstatement applies. Facts: -8F recorded temperatures; HVAC offline April 11–15 due to parts delay (WO-2026-2841); adjuster confirmed pipe failure consistent with freeze; 2025 loss control survey rated contractor satisfactory. Analysis: 4.2 exclusion requires failure to maintain heat; 4.3 reinstates where failure was unforeseeable. Contractor parts delay causing a four-day overrun supports unforeseeable-event argument. Survey finding no exclusion-triggering risk corroborates. Recommendation: Apply Section 4.3, recommend full pay subject to senior officer approval and coverage opinion memorandum; obtain contractor confirmation of parts delay in writing."}
{"schema_version": 1, "id": "lc_003", "workload": "long_context", "prompt": "A compliance officer is preparing the Q1 2026 portfolio review memo. The framework requires: (a) loss ratio trends vs targets, (b) concentration risks, (c) pending regulatory matters, (d) open prior-period action items, (e) renewal recommendations. Draft the loss ratio section for the healthcare and cyber lines given: healthcare book $14.2M premium, 71% TTM loss ratio, 62% target; cyber book led by DataMesh $4.2M claim against $612K premium, three other Aegis MSP accounts.", "expected": "Healthcare: TTM loss ratio of 71% is 9 points above the 62% target on $14.2M premium, representing approximately $1.28M of excess loss. The book is materially underperforming. Recommended action: book-wide rate adequacy review and individual account pricing on all healthcare renewals in H2 2026. Cyber: DataMesh's $4.2M claim against $612K premium produces an account-level loss ratio exceeding 600%. The broader cyber book's Aegis MSP concentration ($1.9M aggregate premium) creates correlated catastrophic exposure not captured in individual account pricing."}
{"schema_version": 1, "id": "lc_004", "workload": "long_context", "prompt": "An energy account (Cascade Pipeline) voluntarily reported a corrosion near-miss that was caught and repaired before any release. The account proactively tightened inspection cycles from 36 to 18 months following a 2024 NTSB advisory. Two peer accounts in the same energy book remain on 36-month cycles. Write a risk assessment note covering: avoided loss range, Cascade's risk posture, and recommended action for the peer accounts.", "expected": "Avoided Loss: Estimated $8M–$22M depending on environmental sensitivity of the rupture location. Cascade Risk Posture: Positive. Prompt adoption of NTSB advisory and voluntary near-miss disclosure demonstrate strong risk management culture. No adverse action recommended. Peer Account Action: Both transmission-line accounts on 36-month cycles should receive loss control notices within 30 days requiring confirmation of NTSB advisory review and a written inspection tightening plan by Q3 2026. Continued non-compliance should trigger underwriting review at next renewal."}
{"schema_version": 1, "id": "lc_005", "workload": "long_context", "prompt": "DataMesh argues that a 7-day customer notification statute does not apply because exfiltration was never confirmed — only encryption. The insurer's Section 7.4 excludes losses arising from regulatory penalties for notification failures. Analyze whether the exclusion applies and what additional investigation is needed.", "expected": "The exclusion is ambiguous on current facts. The statutory clock for notification may begin at confirmation of exfiltration, not encryption; DataMesh's argument that exfiltration was unconfirmed is legally viable. However, if the AG investigation determines the threat actor exfiltrated data before or during encryption, the clock likely started earlier and the exclusion could apply to any regulatory fines. Required investigation: obtain forensic log analysis confirming or ruling out data exfiltration; engage coverage counsel before issuing any reservation of rights; monitor AG proceeding for findings."}
{"schema_version": 1, "id": "lc_006", "workload": "long_context", "prompt": "A malpractice suit has been filed against Riverside Specialty Group alleging delayed cancer diagnosis. Policy limits are $5M per claim, $7M aggregate. Plaintiff demand is $3.5M. Defense is engaged and discovery is early. Account loss ratio is 87% over three years with three prior high-value demands. Draft the underwriting decision memo for the October 2026 renewal.", "expected": "Underwriting Decision Memo — Riverside Specialty Group. Current exposure: $3.5M demand on $5M/$7M policy in early discovery; outcome uncertain. Portfolio context: 87% loss ratio over 3 years, three demands above $1M, loss ratio well above target. Options: (1) Non-renewal with 60-day statutory notice; (2) Rate increase 30–40% plus sub-limits on diagnostic liability and aggregate reduction to $5M; (3) Retain at current terms (not recommended). Recommendation: Offer renewal at Option 2 terms. Non-renewal is reserved if the current suit settles above $1.5M or if another high-value demand is filed before October. Present to senior underwriting committee for approval given account size."}
{"schema_version": 1, "id": "lc_007", "workload": "long_context", "prompt": "The Q1 portfolio review must address five open action items from prior reviews. One item — reviewing shared-vendor exposure across the metals book — was assigned in January but not completed. It is now April. Three metals accounts share NorthernPlex Mechanical, which has two prior claims. What should the memo say about this item and what immediate steps should be taken?", "expected": "The memo should explicitly flag the January action item as past due and escalate its priority. The NorthernPlex concentration discovery makes this item material — it is no longer a routine housekeeping item. Immediate steps: (1) within 30 days, complete the vendor concentration analysis across all lines, not just metals; (2) assign a named owner (suggest: senior property underwriter) with a written deadline; (3) determine whether any of the three NorthernPlex accounts are approaching renewal and flag for loss control conditions requiring vendor diversification or additional sublimits."}
{"schema_version": 1, "id": "lc_008", "workload": "long_context", "prompt": "An underwriter is assessing whether to bind a new commercial cyber account for a cloud services company that shares a managed-service provider with three existing portfolio accounts. The MSP, Aegis IT Partners, serves $1.9M of current cyber premium. The new account would add $400K of premium under the same MSP. Write a concentration risk assessment.", "expected": "Adding the new account would bring Aegis-dependent premium to $2.3M — a single point of systemic failure for a meaningful share of the cyber book. If Aegis suffered a breach or systemic failure affecting multiple clients simultaneously, the portfolio could face correlated losses across four accounts. Assessment: the concentration is material. Before binding, require the new account to document Aegis's own cybersecurity controls, obtain evidence of Aegis's cyber insurance, and confirm there are no shared infrastructure components with the existing three accounts. Consider imposing an MSP concentration sublimit on all four policies."}
{"schema_version": 1, "id": "lc_009", "workload": "long_context", "prompt": "A loss control engineer is writing a post-near-miss report for Cascade Pipeline's corrosion event. The NTSB advisory recommended tightening inspection intervals from 36 to 18 months for pipelines of this diameter and age. Cascade complied. Two peer accounts have not. Draft the engineer's recommendation section.", "expected": "Recommendations: Cascade Pipeline — no adverse action; commend proactive compliance. Issue formal acknowledgment of near-miss disclosure and document inspection tightening in loss control file. Peer Accounts — issue written loss control notices within 30 days citing NTSB advisory and Cascade's near-miss as context. Require each account to submit a written inspection acceleration plan by Q3 2026. Accounts that do not respond or refuse to tighten cycles should be referred to underwriting for renewal review, with potential premium loading or coverage restriction for inspection-cycle non-compliance."}
{"schema_version": 1, "id": "lc_010", "workload": "long_context", "prompt": "A claims officer is preparing the compliance summary table for the Q1 2026 portfolio review. It must show recommended posture for each of the four reviewed accounts. Format as: Account | Claim Status | Coverage Posture | Renewal Flag. Complete the table.", "expected": "Account | Claim Status | Coverage Posture | Renewal Flag\nAcme Industrial | $1.7M open, adjuster recommends 60% partial | Full pay recommended pending coverage opinion; senior officer approval required above $750K | Standard renewal; preserve relationship\nDataMesh Cloud | $4.2M open; AG investigation pending | Pay BI and forensic within aggregate; reserve regulatory exposure pending AG outcome | Review MSP concentration; consider sublimits\nRiverside Specialty | $3.5M demand, early discovery | Defend; no settlement recommendation yet | Non-renewal or significant rate/condition change at October 2026\nCascade Pipeline | No open claim; near-miss resolved | No coverage action needed | Positive; issue loss control commendation; flag peer accounts"}
{"schema_version": 1, "id": "lc_011", "workload": "long_context", "prompt": "Explain how Section 4.3 of the policy functions as a reinstatement clause and under what conditions it would override a Section 4.2 maintenance exclusion, using the Acme Industrial freeze loss as a concrete example.", "expected": "Section 4.3 is a carve-back to Section 4.2. Section 4.2 excludes losses from failure to maintain heat in covered premises during freezing conditions. Section 4.3 reinstates coverage when that failure was unforeseeable or beyond the insured's reasonable control. In Acme's case, the loading bay HVAC was offline because a contractor's scheduled repair ran four days over due to parts delays the insured could not have anticipated. If the contractor's parts delay is documented and corroborated, Section 4.3 would override 4.2 and restore full coverage for the pipe burst and resulting damage."}
{"schema_version": 1, "id": "lc_012", "workload": "long_context", "prompt": "The cyber policy for DataMesh has a $250K SIR, $5M aggregate, and a $1M regulatory fines sublimit. The claim breaks down as: $2.8M business interruption, $1.4M forensic and notification, and a potential AG regulatory penalty of unknown amount. What is the maximum insurer exposure under three scenarios: (a) no regulatory penalty, (b) $800K penalty, (c) $1.5M penalty?", "expected": "(a) No regulatory penalty: total loss $4.2M, minus $250K SIR = $3.95M insurer exposure, within $5M aggregate. (b) $800K penalty: total $5.0M, minus $250K SIR = $4.75M, with the $800K penalty within the $1M regulatory sublimit. Total insurer exposure $4.75M, still within $5M aggregate. (c) $1.5M penalty: regulatory exposure capped at $1M sublimit; insured absorbs $500K above sublimit. Total insurer exposure: $3.95M non-regulatory + $1M regulatory cap = $4.95M, within $5M aggregate."}
{"schema_version": 1, "id": "lc_013", "workload": "long_context", "prompt": "An underwriting committee is debating whether to offer Riverside Specialty a 30% rate increase versus non-renewal. The account has $341K current premium, 87% loss ratio over 3 years, and a $3.5M pending demand with $5M per-claim limit. Model the projected loss ratio at the 30% increased rate if incurred losses stay flat.", "expected": "At 30% rate increase: new premium = $341K × 1.30 = $443K. If incurred losses stay flat at current 3-year average implied by 87% loss ratio ($341K × 0.87 ≈ $297K/year), new loss ratio = $297K / $443K ≈ 67%. That approaches but does not meet a 62% target. The committee should note that the pending $3.5M demand — even at a 30% settlement ($1.05M) — would spike the single-year loss ratio far above target. A 30% rate increase alone is insufficient if the current suit settles unfavorably; tighter conditions (sublimits, aggregate reduction) are required to make the account viable."}
{"schema_version": 1, "id": "lc_014", "workload": "long_context", "prompt": "A reinsurance team needs to know which of the four Q1 2026 portfolio accounts may breach treaty retention thresholds. The treaty retention is $750K. Identify which claims may require reinsurance notification and at what stage.", "expected": "Acme Industrial: $1.7M claim exceeds $750K retention. Reinsurance notification is required now, prior to any settlement. A coverage opinion and senior officer approval are also required. DataMesh Cloud: $4.2M claim is well above $750K. Reinsurance has almost certainly already been notified; confirm notification is on file. Riverside Specialty: $3.5M plaintiff demand exceeds retention. Notify reinsurance even though discovery is early — late notification can jeopardize recovery. Cascade Pipeline: no open claim; no reinsurance action needed."}
{"schema_version": 1, "id": "lc_015", "workload": "long_context", "prompt": "Write a one-page executive summary of the Q1 2026 commercial lines portfolio review covering: overall loss performance, top three risks, overdue action items, and immediate recommended actions.", "expected": "Q1 2026 Executive Summary. Performance: Healthcare book 71% TTM loss ratio vs 62% target; cyber book has one account (DataMesh) with loss ratio exceeding 600%; energy book clean. Three open major claims totaling $9.4M in exposure across commercial property, cyber, and professional liability. Top Three Risks: (1) NorthernPlex HVAC vendor concentration across metals book — overdue action item; (2) Aegis IT Partners MSP concentration across four cyber accounts ($1.9M aggregate); (3) Riverside healthcare account loss ratio driving book-level underperformance. Overdue Action Items: January vendor concentration review is 90 days past due; must be completed in April. Immediate Actions: (1) Complete vendor concentration analysis and assign owner; (2) Notify reinsurance on Acme and Riverside if not already done; (3) Initiate Riverside renewal review 6 months early; (4) Issue loss control notices to two energy peer accounts."}
```

- [ ] **Step 4: Commit datasets**

```bash
git add datasets/
git commit -m "feat: add quality eval datasets (chat, rag, long_context)"
```

---

## Task 3: `evaluate/run_eval.py` — Pure Utility Functions

**Files:**
- Create: `evaluate/run_eval.py` (partial — utilities only)
- Create: `tests/test_run_eval.py` (partial)

These five functions have no external dependencies and are fully unit-testable: `load_dataset`, `normalize_score`, `select_metrics`, `derive_tag`, `write_sidecar`.

- [ ] **Step 1: Write failing tests for utility functions**

Create `tests/test_run_eval.py`:

```python
import json
import os
import pytest


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


VALID_ROW = {
    "schema_version": 1,
    "id": "test_001",
    "workload": "chat",
    "prompt": "What is 2+2?",
    "expected": "4",
}


class TestLoadDataset:
    def test_valid_rows_are_returned(self, tmp_path):
        p = tmp_path / "test.jsonl"
        write_jsonl(p, [VALID_ROW])
        from evaluate.run_eval import load_dataset
        rows = load_dataset(str(p))
        assert len(rows) == 1
        assert rows[0]["id"] == "test_001"

    def test_missing_required_field_exits(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        bad = {k: v for k, v in VALID_ROW.items() if k != "expected"}
        write_jsonl(p, [bad])
        from evaluate.run_eval import load_dataset
        with pytest.raises(SystemExit):
            load_dataset(str(p))

    def test_invalid_workload_exits(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        bad = {**VALID_ROW, "workload": "unknown_type"}
        write_jsonl(p, [bad])
        from evaluate.run_eval import load_dataset
        with pytest.raises(SystemExit):
            load_dataset(str(p))

    def test_empty_file_exits(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        from evaluate.run_eval import load_dataset
        with pytest.raises(SystemExit):
            load_dataset(str(p))

    def test_invalid_json_exits(self, tmp_path):
        p = tmp_path / "broken.jsonl"
        p.write_text("not json\n")
        from evaluate.run_eval import load_dataset
        with pytest.raises(SystemExit):
            load_dataset(str(p))


class TestNormalizeScore:
    def test_hallucination_is_inverted(self):
        from evaluate.run_eval import normalize_score
        assert normalize_score("hallucination", 0.2) == pytest.approx(0.8)

    def test_hallucination_clamps_at_zero(self):
        from evaluate.run_eval import normalize_score
        assert normalize_score("hallucination", 1.1) == pytest.approx(0.0)

    def test_relevancy_passes_through(self):
        from evaluate.run_eval import normalize_score
        assert normalize_score("answer_relevancy", 0.9) == pytest.approx(0.9)

    def test_correctness_passes_through(self):
        from evaluate.run_eval import normalize_score
        assert normalize_score("correctness", 0.75) == pytest.approx(0.75)


class TestSelectMetrics:
    def test_chat_has_relevancy_and_correctness(self):
        from evaluate.run_eval import select_metrics
        metrics = select_metrics("chat", has_contexts=False)
        assert "answer_relevancy" in metrics
        assert "correctness" in metrics
        assert "faithfulness" not in metrics

    def test_rag_without_contexts_has_no_faithfulness(self):
        from evaluate.run_eval import select_metrics
        metrics = select_metrics("rag", has_contexts=False)
        assert "faithfulness" not in metrics

    def test_rag_with_contexts_adds_faithfulness_and_hallucination(self):
        from evaluate.run_eval import select_metrics
        metrics = select_metrics("rag", has_contexts=True)
        assert "faithfulness" in metrics
        assert "hallucination" in metrics

    def test_long_context_same_as_chat(self):
        from evaluate.run_eval import select_metrics
        assert select_metrics("long_context", False) == select_metrics("chat", False)


class TestDeriveTag:
    def test_strips_json_extension(self):
        from evaluate.run_eval import derive_tag
        assert derive_tag("results/real/vllm_l4fp8_isl2k_c10.json") == "vllm_l4fp8_isl2k_c10"

    def test_works_with_nested_path(self):
        from evaluate.run_eval import derive_tag
        assert derive_tag("/some/deep/path/my_tag.json") == "my_tag"


class TestWriteSidecar:
    def test_writes_valid_json(self, tmp_path):
        from evaluate.run_eval import write_sidecar
        path = write_sidecar(
            out_dir=str(tmp_path),
            tag="test_tag",
            latency_tag="test_tag",
            evaluator="deepeval",
            model="llama-3.1-8b",
            dataset_path="datasets/rag.jsonl",
            num_samples=15,
            metrics={"answer_relevancy": 0.93, "correctness": 0.91},
            overall_score=0.92,
            cost_per_million=0.80,
            throughput_proxy=262,
        )
        assert os.path.isfile(path)
        with open(path) as f:
            data = json.load(f)
        assert data["meta"]["latency_tag"] == "test_tag"
        assert data["metrics"]["overall_score"] == 0.92
        assert data["cost"]["per_million_tokens"] == 0.80
        assert data["cost"]["throughput_proxy_tokens_per_sec"] == 262

    def test_creates_output_dir_if_missing(self, tmp_path):
        from evaluate.run_eval import write_sidecar
        nested = str(tmp_path / "deep" / "dir")
        write_sidecar(nested, "t", "t", "deepeval", "m", "d", 1, {}, 0.5, None, None)
        assert os.path.isdir(nested)
```

- [ ] **Step 2: Run tests — verify they all fail with ImportError**

```bash
pytest tests/test_run_eval.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'evaluate.run_eval'` (or similar import error).

- [ ] **Step 3: Create `evaluate/run_eval.py` with utility functions**

Create `evaluate/run_eval.py`:

```python
#!/usr/bin/env python3
"""
Offline quality evaluator for LLM inference deployments.

Sends a small evaluation dataset at an OpenAI-compatible endpoint,
scores responses with DeepEval, and writes a quality sidecar JSON
alongside the latency result.
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp is required. Install with: pip install aiohttp", file=sys.stderr)
    sys.exit(1)


def load_dataset(path):
    """Load and validate JSONL dataset. Returns list of row dicts."""
    required = {"schema_version", "id", "workload", "prompt", "expected"}
    valid_workloads = {"chat", "rag", "long_context"}
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print("ERROR: {}:{}: invalid JSON: {}".format(path, lineno, e), file=sys.stderr)
                sys.exit(1)
            missing = required - set(row.keys())
            if missing:
                print("ERROR: {}:{}: missing fields: {}".format(path, lineno, missing), file=sys.stderr)
                sys.exit(1)
            if row["workload"] not in valid_workloads:
                print("ERROR: {}:{}: unknown workload '{}'".format(path, lineno, row["workload"]), file=sys.stderr)
                sys.exit(1)
            rows.append(row)
    if not rows:
        print("ERROR: {}: no valid rows found".format(path), file=sys.stderr)
        sys.exit(1)
    return rows


def normalize_score(metric_name, raw_score):
    """Normalize metric to higher-is-better in range [0, 1].
    Inverts rate metrics where lower is better (e.g. hallucination_rate)."""
    if metric_name == "hallucination":
        return max(0.0, 1.0 - float(raw_score))
    return float(raw_score)


def select_metrics(workload, has_contexts):
    """Return list of metric names to activate for this workload."""
    metrics = ["answer_relevancy", "correctness"]
    if workload == "rag" and has_contexts:
        metrics += ["faithfulness", "hallucination"]
    return metrics


def derive_tag(latency_result_path):
    """Derive output tag from latency result filename."""
    return os.path.basename(latency_result_path).replace(".json", "")


def write_sidecar(out_dir, tag, latency_tag, evaluator, model, dataset_path,
                  num_samples, metrics, overall_score, cost_per_million, throughput_proxy):
    """Write quality sidecar JSON to <out_dir>/<tag>.json."""
    os.makedirs(out_dir, exist_ok=True)
    out = {
        "meta": {
            "tag": tag,
            "latency_tag": latency_tag,
            "evaluator": evaluator,
            "model": model,
            "dataset": dataset_path,
            "num_samples": num_samples,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "metrics": dict(metrics, overall_score=overall_score),
        "cost": {
            "per_million_tokens": cost_per_million,
            "throughput_proxy_tokens_per_sec": throughput_proxy,
        },
    }
    path = os.path.join(out_dir, tag + ".json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path
```

- [ ] **Step 4: Run tests — verify all pass**

```bash
pytest tests/test_run_eval.py -v
```

Expected: All tests pass. Output ends with `X passed`.

- [ ] **Step 5: Commit**

```bash
git add evaluate/run_eval.py tests/test_run_eval.py pytest.ini
git commit -m "feat: add run_eval utility functions with tests"
```

---

## Task 4: `evaluate/run_eval.py` — Response Collection

**Files:**
- Modify: `evaluate/run_eval.py` (add `send_prompt`, `collect_responses`)

Response collection uses `aiohttp` — same pattern as `collect/run_bench.py`. No unit test for the async HTTP call; coverage comes from the dry-run smoke test in Task 7.

- [ ] **Step 1: Add `send_prompt` and `collect_responses` to `evaluate/run_eval.py`**

Add these two functions after the `write_sidecar` function:

```python
async def send_prompt(session, endpoint, model, token, prompt, max_tokens=256):
    """Send a single prompt (non-streaming) and return the response text."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    async with session.post(endpoint, json=payload, headers=headers) as resp:
        if resp.status != 200:
            raise RuntimeError("HTTP {}".format(resp.status))
        data = await resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("empty choices in response")
        return (
            choices[0].get("text")
            or choices[0].get("message", {}).get("content")
            or ""
        )


async def collect_responses(endpoint, model, token, dataset, concurrency=5):
    """
    Send all dataset prompts to the endpoint.
    Returns (samples, errors) where samples is a list of (row, response_text) tuples.
    Concurrency is deliberately low (5) to avoid warming the KV cache or
    interfering with a parallel load test.
    """
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=120)
    samples = []
    errors = []

    async def one(session, row):
        async with sem:
            try:
                response = await send_prompt(session, endpoint, model, token, row["prompt"])
                samples.append((row, response))
            except Exception as e:
                errors.append((row["id"], repr(e)))
                print("WARN: sample {} failed: {}".format(row["id"], e), file=sys.stderr)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        await asyncio.gather(*[one(session, row) for row in dataset])

    return samples, errors
```

- [ ] **Step 2: Verify existing tests still pass**

```bash
pytest tests/test_run_eval.py -v
```

Expected: All tests still pass (new functions are not yet tested — that comes via dry-run in Task 7).

- [ ] **Step 3: Commit**

```bash
git add evaluate/run_eval.py
git commit -m "feat: add async response collection to run_eval"
```

---

## Task 5: `evaluate/run_eval.py` — DeepEval Integration

**Files:**
- Modify: `evaluate/run_eval.py` (add `run_deepeval`)

- [ ] **Step 1: Add `run_deepeval` to `evaluate/run_eval.py`**

Add after `collect_responses`:

```python
def run_deepeval(samples, eval_model, workload):
    """
    Score (row, response) samples using DeepEval metrics.
    Returns (aggregated_metrics_dict, overall_score).
    overall_score is the mean of all normalized per-metric scores.
    """
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    has_contexts = any(row.get("contexts") for row, _ in samples)
    active = select_metrics(workload, has_contexts)

    metrics = []
    if "answer_relevancy" in active:
        metrics.append(AnswerRelevancyMetric(model=eval_model, threshold=0.5))
    if "correctness" in active:
        metrics.append(GEval(
            name="Correctness",
            criteria=(
                "Does the actual output accurately answer the input question "
                "based on the expected output?"
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=eval_model,
            threshold=0.5,
        ))
    if "faithfulness" in active:
        metrics.append(FaithfulnessMetric(model=eval_model, threshold=0.5))
    if "hallucination" in active:
        metrics.append(HallucinationMetric(model=eval_model, threshold=0.5))

    # canonical_name maps each DeepEval metric object to our schema key
    def canonical_name(metric):
        n = type(metric).__name__.lower()
        if "relevancy" in n:
            return "answer_relevancy"
        if "geval" in n or "correctness" in n:
            return "correctness"
        if "faithfulness" in n:
            return "faithfulness"
        if "hallucination" in n:
            return "hallucination"
        return n

    per_metric = {m: [] for m in active}

    for row, response in samples:
        tc = LLMTestCase(
            input=row["prompt"],
            actual_output=response,
            expected_output=row["expected"],
            context=row.get("contexts"),
        )
        for metric in metrics:
            try:
                metric.measure(tc)
                key = canonical_name(metric)
                if key in per_metric:
                    per_metric[key].append(normalize_score(key, metric.score))
            except Exception as e:
                print(
                    "WARN: DeepEval {} failed on sample {}: {}".format(
                        type(metric).__name__, row["id"], e
                    ),
                    file=sys.stderr,
                )

    aggregated = {
        k: round(sum(vals) / len(vals), 4)
        for k, vals in per_metric.items()
        if vals
    }
    overall_score = (
        round(sum(aggregated.values()) / len(aggregated), 4) if aggregated else None
    )
    return aggregated, overall_score
```

- [ ] **Step 2: Verify existing tests still pass**

```bash
pytest tests/test_run_eval.py -v
```

Expected: All tests still pass.

- [ ] **Step 3: Commit**

```bash
git add evaluate/run_eval.py
git commit -m "feat: add DeepEval scoring to run_eval"
```

---

## Task 6: `evaluate/run_eval.py` — LLM-Judge Path and CLI

**Files:**
- Modify: `evaluate/run_eval.py` (add `run_llm_judge`, `main`)

- [ ] **Step 1: Add `run_llm_judge` to `evaluate/run_eval.py`**

Add after `run_deepeval`:

```python
def run_llm_judge(samples, eval_endpoint, eval_model, eval_token):
    """
    Score samples using any OpenAI-compatible chat endpoint as judge.
    Returns (aggregated_metrics_dict, overall_score).
    """
    import urllib.request

    SCORING_PROMPT = (
        "Score this response on three dimensions (1-5 each):\n"
        "  correctness: does it answer the question accurately?\n"
        "  helpfulness: is it useful and complete?\n"
        "  hallucination: 5=none, 1=severe fabrication\n\n"
        "Question: {prompt}\n"
        "Expected answer: {expected}\n"
        "Response: {response}\n\n"
        'Return JSON only: {{"correctness": N, "helpfulness": N, "hallucination": N}}'
    )

    scores = {"correctness": [], "helpfulness": [], "hallucination": []}

    for row, response in samples:
        prompt = SCORING_PROMPT.format(
            prompt=row["prompt"][:500],
            expected=row["expected"][:500],
            response=response[:500],
        )
        headers = {"Content-Type": "application/json"}
        if eval_token:
            headers["Authorization"] = "Bearer " + eval_token
        payload = json.dumps({
            "model": eval_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 64,
        }).encode()

        try:
            req = urllib.request.Request(
                eval_endpoint.rstrip("/") + "/chat/completions",
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            for k in scores:
                if k in parsed:
                    raw = float(parsed[k]) / 5.0  # normalize 1-5 → 0-1
                    scores[k].append(normalize_score(k, raw))
        except Exception as e:
            print("WARN: LLM judge failed on sample {}: {}".format(row["id"], e), file=sys.stderr)

    aggregated = {
        k: round(sum(vals) / len(vals), 4)
        for k, vals in scores.items()
        if vals
    }
    overall_score = (
        round(sum(aggregated.values()) / len(aggregated), 4) if aggregated else None
    )
    return aggregated, overall_score
```

- [ ] **Step 2: Add `main` to `evaluate/run_eval.py`**

Add at the end of the file:

```python
def main():
    ap = argparse.ArgumentParser(description="Offline quality evaluator for LLM deployments")
    ap.add_argument("--endpoint", required=True,
                    help="Inference endpoint being evaluated (OpenAI-compatible /v1/completions)")
    ap.add_argument("--model", required=True,
                    help="Model name served at --endpoint")
    ap.add_argument("--latency-result", required=True, dest="latency_result",
                    help="Path to the latency JSON produced by collect/run_bench.py")
    ap.add_argument("--dataset", required=True,
                    help="Path to eval JSONL dataset (datasets/chat.jsonl etc.)")
    ap.add_argument("--evaluator", choices=["deepeval", "llm-judge"], default="deepeval",
                    help="Scoring backend (default: deepeval)")
    ap.add_argument("--eval-model", dest="eval_model", default="gpt-4o",
                    help="Judge model name used by DeepEval or llm-judge (default: gpt-4o)")
    ap.add_argument("--eval-endpoint", dest="eval_endpoint",
                    default="https://api.openai.com/v1",
                    help="Judge model endpoint (default: https://api.openai.com/v1)")
    ap.add_argument("--token", default=os.environ.get("OPENAI_API_KEY", ""),
                    help="Bearer token for --endpoint (default: $OPENAI_API_KEY)")
    ap.add_argument("--eval-token", dest="eval_token",
                    default=os.environ.get("OPENAI_API_KEY", ""),
                    help="Bearer token for --eval-endpoint (default: $OPENAI_API_KEY)")
    ap.add_argument("--cost-per-million-tokens", type=float, dest="cost_per_million",
                    default=None,
                    help="Cost per 1M output tokens for this deployment (optional)")
    ap.add_argument("--output-dir", dest="output_dir", default="./results/quality",
                    help="Directory for quality sidecar JSON (default: ./results/quality)")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="Validate inputs and print plan without hitting the endpoint")
    args = ap.parse_args()

    if not os.path.isfile(args.latency_result):
        print("ERROR: latency result not found: {}".format(args.latency_result), file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.dataset):
        print("ERROR: dataset not found: {}".format(args.dataset), file=sys.stderr)
        sys.exit(1)

    with open(args.latency_result) as f:
        latency_data = json.load(f)
    latency_tag = latency_data.get("meta", {}).get("tag") or derive_tag(args.latency_result)
    throughput_proxy = latency_data.get("metrics", {}).get("throughput_tokens_per_sec")

    dataset = load_dataset(args.dataset)
    tag = derive_tag(args.latency_result)
    workloads = {row["workload"] for row in dataset}
    workload = workloads.pop() if len(workloads) == 1 else "mixed"

    if args.dry_run:
        print("=== Dry run ===")
        print("  latency-result : {}".format(args.latency_result))
        print("  latency-tag    : {}".format(latency_tag))
        print("  dataset        : {} ({} samples, workload={})".format(
            args.dataset, len(dataset), workload))
        print("  evaluator      : {}".format(args.evaluator))
        print("  eval-model     : {}".format(args.eval_model))
        print("  eval-endpoint  : {}".format(args.eval_endpoint))
        print("  output-tag     : {}".format(tag))
        print("  output-dir     : {}".format(args.output_dir))
        print("Would collect responses and run evaluation. Exiting (--dry-run).")
        return

    print("Collecting {} responses from {} ...".format(len(dataset), args.endpoint))
    samples, errors = asyncio.run(
        collect_responses(args.endpoint, args.model, args.token, dataset)
    )
    if not samples:
        print("ERROR: no samples collected — check endpoint and model", file=sys.stderr)
        sys.exit(1)
    if errors:
        print("WARN: {}/{} samples failed, continuing with {}".format(
            len(errors), len(dataset), len(samples)))

    print("Evaluating with {} ...".format(args.evaluator))
    if args.evaluator == "deepeval":
        try:
            import deepeval  # noqa: F401
        except ImportError:
            print("ERROR: DeepEval not installed. Run: pip install deepeval", file=sys.stderr)
            sys.exit(1)
        metrics, overall_score = run_deepeval(samples, args.eval_model, workload)
    else:
        metrics, overall_score = run_llm_judge(
            samples, args.eval_endpoint, args.eval_model, args.eval_token
        )

    out_path = write_sidecar(
        out_dir=os.path.expanduser(args.output_dir),
        tag=tag,
        latency_tag=latency_tag,
        evaluator=args.evaluator,
        model=args.model,
        dataset_path=args.dataset,
        num_samples=len(samples),
        metrics=metrics,
        overall_score=overall_score,
        cost_per_million=args.cost_per_million,
        throughput_proxy=throughput_proxy,
    )

    print("overall_score={}".format(overall_score))
    for k, v in metrics.items():
        print("  {}={}".format(k, v))
    print("wrote {}".format(out_path))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run all tests**

```bash
pytest tests/test_run_eval.py -v
```

Expected: All tests pass.

- [ ] **Step 4: Smoke test with --dry-run against a synthetic result file**

```bash
python evaluate/run_eval.py \
  --endpoint http://localhost:8000/v1/completions \
  --model llama-3.1-8b \
  --latency-result results/synthetic/vllm_l4fp8_isl2k_c10.json \
  --dataset datasets/rag.jsonl \
  --dry-run
```

Expected output:
```
=== Dry run ===
  latency-result : results/synthetic/vllm_l4fp8_isl2k_c10.json
  latency-tag    : vllm_l4fp8_isl2k_c10
  dataset        : datasets/rag.jsonl (15 samples, workload=rag)
  evaluator      : deepeval
  eval-model     : gpt-4o
  eval-endpoint  : https://api.openai.com/v1
  output-tag     : vllm_l4fp8_isl2k_c10
  output-dir     : ./results/quality
Would collect responses and run evaluation. Exiting (--dry-run).
```

- [ ] **Step 5: Commit**

```bash
git add evaluate/run_eval.py
git commit -m "feat: complete run_eval.py with LLM-judge path and CLI"
```

---

## Task 7: `analyze/deployment_advisor.py` — `load_deployment`

**Files:**
- Create: `analyze/deployment_advisor.py` (partial)
- Create: `tests/test_deployment_advisor.py` (partial)

- [ ] **Step 1: Write failing tests for `load_deployment`**

Create `tests/test_deployment_advisor.py`:

```python
import json
import pytest


def make_latency_json(tag, ttft_p50=115, ttft_p95=133, throughput=262, model="llama-3.1-8b"):
    return {
        "meta": {
            "tag": tag, "model": model, "runtime": "vllm",
            "gpu": {"name": "NVIDIA L4", "memory_mb": 23034, "util_pct": 0},
            "config": {"chunked_prefill": False, "tensor_parallel_size": 1,
                       "shared_prefix": False},
            "workload": {"isl_approx": 2048, "osl_max": 128, "concurrency": 10,
                         "duration_secs": 90},
            "synthetic": True, "timestamp": "2026-06-12T00:00:00+00:00",
        },
        "metrics": {
            "ttft_ms": {"p50": ttft_p50, "p95": ttft_p95, "p99": 200, "mean": 110},
            "total_latency_ms": {"p50": 4000, "p95": 4500, "p99": 5000},
            "throughput_tokens_per_sec": throughput,
            "throughput_req_per_sec": 2.0,
            "total_requests": 100, "successful_requests": 100, "failed_requests": 0,
        },
    }


def make_quality_json(tag, overall_score=0.93, cost=0.80, throughput=262,
                      model="llama-3.1-8b", latency_tag=None, dataset="datasets/rag.jsonl"):
    return {
        "meta": {
            "tag": tag,
            "latency_tag": latency_tag or tag,
            "evaluator": "deepeval",
            "model": model,
            "dataset": dataset,
            "num_samples": 15,
            "timestamp": "2026-06-12T00:00:00+00:00",
        },
        "metrics": {
            "answer_relevancy": 0.93,
            "correctness": 0.92,
            "overall_score": overall_score,
        },
        "cost": {
            "per_million_tokens": cost,
            "throughput_proxy_tokens_per_sec": throughput,
        },
    }


def write_json(directory, filename, data):
    path = directory / filename
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def lat_dir(tmp_path):
    d = tmp_path / "real"
    d.mkdir()
    return d


@pytest.fixture
def qual_dir(tmp_path):
    d = tmp_path / "quality"
    d.mkdir()
    return d


class TestLoadDeployment:
    def test_latency_only_profile(self, lat_dir, qual_dir):
        write_json(lat_dir, "fp16.json", make_latency_json("fp16"))
        from analyze.deployment_advisor import load_deployment
        profile = load_deployment("fp16", [str(lat_dir)], str(qual_dir))
        assert profile["tag"] == "fp16"
        assert profile["latency"]["ttft_ms_p50"] == 115
        assert profile["latency"]["ttft_ms_p95"] == 133
        assert profile["latency"]["throughput_tokens_per_sec"] == 262
        assert profile["quality"] is None

    def test_flattens_nested_latency_schema(self, lat_dir, qual_dir):
        write_json(lat_dir, "fp16.json", make_latency_json("fp16", ttft_p50=200, ttft_p95=350))
        from analyze.deployment_advisor import load_deployment
        profile = load_deployment("fp16", [str(lat_dir)], str(qual_dir))
        # raw JSON has metrics.ttft_ms.p50; profile must have latency.ttft_ms_p50
        assert profile["latency"]["ttft_ms_p50"] == 200
        assert profile["latency"]["ttft_ms_p95"] == 350

    def test_merges_quality_sidecar(self, lat_dir, qual_dir):
        write_json(lat_dir, "fp8.json", make_latency_json("fp8", ttft_p50=80))
        write_json(qual_dir, "fp8.json", make_quality_json("fp8", overall_score=0.93))
        from analyze.deployment_advisor import load_deployment
        profile = load_deployment("fp8", [str(lat_dir)], str(qual_dir))
        assert profile["quality"]["overall_score"] == 0.93
        assert profile["cost"]["per_million_tokens"] == 0.80

    def test_missing_latency_tag_exits(self, lat_dir, qual_dir):
        from analyze.deployment_advisor import load_deployment
        with pytest.raises(SystemExit):
            load_deployment("nonexistent", [str(lat_dir)], str(qual_dir))

    def test_latency_tag_mismatch_is_hard_error(self, lat_dir, qual_dir):
        write_json(lat_dir, "fp8.json", make_latency_json("fp8"))
        stale = make_quality_json("fp8", latency_tag="different_tag")
        write_json(qual_dir, "fp8.json", stale)
        from analyze.deployment_advisor import load_deployment
        with pytest.raises(SystemExit):
            load_deployment("fp8", [str(lat_dir)], str(qual_dir))

    def test_real_overrides_synthetic(self, tmp_path):
        syn_dir = tmp_path / "synthetic"
        real_dir = tmp_path / "real"
        qual_dir = tmp_path / "quality"
        syn_dir.mkdir(); real_dir.mkdir(); qual_dir.mkdir()
        write_json(syn_dir, "fp8.json", make_latency_json("fp8", ttft_p50=150))
        write_json(real_dir, "fp8.json", make_latency_json("fp8", ttft_p50=90))
        from analyze.deployment_advisor import load_deployment
        profile = load_deployment("fp8", [str(syn_dir), str(real_dir)], str(qual_dir))
        assert profile["latency"]["ttft_ms_p50"] == 90  # real wins
```

- [ ] **Step 2: Run tests — verify fail with ImportError**

```bash
pytest tests/test_deployment_advisor.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'analyze.deployment_advisor'`

- [ ] **Step 3: Create `analyze/deployment_advisor.py` with `load_deployment`**

Create `analyze/deployment_advisor.py`:

```python
#!/usr/bin/env python3
"""
Quality-aware deployment advisor.

Merges latency benchmark results with quality evaluation sidecars to
produce a deployment recommendation balancing latency, cost, and quality.
"""
import argparse
import json
import os
import sys


REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
REAL_DIR = os.path.join(REPO_ROOT, "results", "real")
SYN_DIR = os.path.join(REPO_ROOT, "results", "synthetic")
QUALITY_DIR = os.path.join(REPO_ROOT, "results", "quality")


def _find_latency_file(tag, latency_dirs):
    """Return path to latency JSON for tag. Later dirs override earlier ones."""
    found = None
    for d in latency_dirs:
        path = os.path.join(d, tag + ".json")
        if os.path.isfile(path):
            found = path
    return found


def validate_profile(profile):
    """Fail fast if required latency fields are missing."""
    required = {"ttft_ms_p50", "ttft_ms_p95", "throughput_tokens_per_sec"}
    missing = required - set(profile.get("latency", {}).keys())
    if missing:
        print(
            "ERROR: profile '{}' missing latency fields: {}".format(
                profile.get("tag"), missing
            ),
            file=sys.stderr,
        )
        sys.exit(1)


def load_deployment(tag, latency_dirs, quality_dir):
    """
    Load and merge latency + quality data into a normalized DeploymentProfile.

    Flattens the existing nested result schema (e.g. metrics.ttft_ms.p50)
    into a flat in-memory shape (latency.ttft_ms_p50) so all downstream
    functions work against a single consistent structure.
    """
    lat_path = _find_latency_file(tag, latency_dirs)
    if lat_path is None:
        print("ERROR: no latency result found for tag '{}'".format(tag), file=sys.stderr)
        sys.exit(1)

    with open(lat_path) as f:
        lat_raw = json.load(f)

    meta = lat_raw.get("meta", {})
    m = lat_raw.get("metrics", {})
    ttft = m.get("ttft_ms", {})

    profile = {
        "tag": tag,
        "model": meta.get("model", "unknown"),
        "latency": {
            "ttft_ms_p50": ttft.get("p50"),
            "ttft_ms_p95": ttft.get("p95"),
            "throughput_tokens_per_sec": m.get("throughput_tokens_per_sec"),
        },
        "quality": None,
        "cost": {
            "per_million_tokens": None,
            "throughput_proxy_tokens_per_sec": m.get("throughput_tokens_per_sec"),
        },
        "_dataset": None,
    }
    validate_profile(profile)

    qual_path = os.path.join(quality_dir, tag + ".json")
    if os.path.isfile(qual_path):
        with open(qual_path) as f:
            qual_raw = json.load(f)

        qual_latency_tag = qual_raw.get("meta", {}).get("latency_tag")
        if qual_latency_tag and qual_latency_tag != tag:
            print(
                "ERROR: quality sidecar for '{}' has latency_tag='{}'. "
                "This sidecar was generated for a different latency result. "
                "Re-run evaluate/run_eval.py with --latency-result pointing "
                "to the correct file.".format(tag, qual_latency_tag),
                file=sys.stderr,
            )
            sys.exit(1)

        qm = qual_raw.get("metrics", {})
        profile["quality"] = {
            "overall_score": qm.get("overall_score"),
            "metrics": {k: v for k, v in qm.items() if k != "overall_score"},
        }
        cost = qual_raw.get("cost", {})
        profile["cost"]["per_million_tokens"] = cost.get("per_million_tokens")
        profile["cost"]["throughput_proxy_tokens_per_sec"] = (
            cost.get("throughput_proxy_tokens_per_sec")
            or profile["cost"]["throughput_proxy_tokens_per_sec"]
        )
        profile["_dataset"] = qual_raw.get("meta", {}).get("dataset")
    else:
        print(
            "WARN: no quality sidecar for '{}' — quality metrics will be N/A".format(tag),
            file=sys.stderr,
        )

    return profile
```

- [ ] **Step 4: Run tests — verify all pass**

```bash
pytest tests/test_deployment_advisor.py::TestLoadDeployment -v
```

Expected: All 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add analyze/deployment_advisor.py tests/test_deployment_advisor.py
git commit -m "feat: add load_deployment with DeploymentProfile normalization"
```

---

## Task 8: `analyze/deployment_advisor.py` — `compute_tradeoff`

**Files:**
- Modify: `analyze/deployment_advisor.py`
- Modify: `tests/test_deployment_advisor.py`

- [ ] **Step 1: Add failing tests for `compute_tradeoff`**

Append to `tests/test_deployment_advisor.py`:

```python
def make_profile(tag, ttft=115, throughput=262, quality=0.95, cost=1.20,
                 model="llama-3.1-8b", dataset="datasets/rag.jsonl"):
    return {
        "tag": tag,
        "model": model,
        "latency": {
            "ttft_ms_p50": ttft,
            "ttft_ms_p95": ttft + 20,
            "throughput_tokens_per_sec": throughput,
        },
        "quality": {"overall_score": quality, "metrics": {}} if quality is not None else None,
        "cost": {
            "per_million_tokens": cost,
            "throughput_proxy_tokens_per_sec": throughput,
        },
        "_dataset": dataset,
    }


class TestComputeTradeoff:
    def test_latency_improvement_calculated_correctly(self):
        profiles = [
            make_profile("fp16", ttft=1200, cost=1.20, quality=0.95),
            make_profile("fp8", ttft=800, cost=0.91, quality=0.933),
        ]
        from analyze.deployment_advisor import compute_tradeoff
        table = compute_tradeoff(profiles, "fp16")
        fp8 = next(r for r in table if r["tag"] == "fp8")
        assert fp8["latency_improvement"] == pytest.approx((1200 - 800) / 1200, rel=1e-3)

    def test_quality_delta_calculated_correctly(self):
        profiles = [
            make_profile("fp16", quality=0.95),
            make_profile("fp8", quality=0.933),
        ]
        from analyze.deployment_advisor import compute_tradeoff
        table = compute_tradeoff(profiles, "fp16")
        fp8 = next(r for r in table if r["tag"] == "fp8")
        assert fp8["quality_delta"] == pytest.approx(0.933 - 0.95, rel=1e-3)

    def test_cost_reduction_uses_per_million_when_available(self):
        profiles = [
            make_profile("fp16", cost=1.20),
            make_profile("fp8", cost=0.91),
        ]
        from analyze.deployment_advisor import compute_tradeoff
        table = compute_tradeoff(profiles, "fp16")
        fp8 = next(r for r in table if r["tag"] == "fp8")
        assert fp8["cost_reduction"] == pytest.approx((1.20 - 0.91) / 1.20, rel=1e-3)

    def test_cost_reduction_falls_back_to_throughput_proxy(self):
        profiles = [
            {**make_profile("fp16", cost=None), "cost": {"per_million_tokens": None, "throughput_proxy_tokens_per_sec": 200}},
            {**make_profile("fp8", cost=None), "cost": {"per_million_tokens": None, "throughput_proxy_tokens_per_sec": 300}},
        ]
        from analyze.deployment_advisor import compute_tradeoff
        table = compute_tradeoff(profiles, "fp16")
        fp8 = next(r for r in table if r["tag"] == "fp8")
        assert fp8["cost_reduction"] == pytest.approx((300 - 200) / 300, rel=1e-3)

    def test_baseline_is_marked(self):
        profiles = [make_profile("fp16"), make_profile("fp8")]
        from analyze.deployment_advisor import compute_tradeoff
        table = compute_tradeoff(profiles, "fp16")
        baseline = next(r for r in table if r["tag"] == "fp16")
        assert baseline["is_baseline"] is True

    def test_rejects_mismatched_models(self):
        profiles = [
            make_profile("fp16", model="llama-3.1-8b"),
            make_profile("fp8", model="mistral-7b"),
        ]
        from analyze.deployment_advisor import compute_tradeoff
        with pytest.raises(SystemExit):
            compute_tradeoff(profiles, "fp16")

    def test_rejects_mismatched_datasets(self):
        profiles = [
            make_profile("fp16", dataset="datasets/rag.jsonl"),
            make_profile("fp8", dataset="datasets/chat.jsonl"),
        ]
        from analyze.deployment_advisor import compute_tradeoff
        with pytest.raises(SystemExit):
            compute_tradeoff(profiles, "fp16")

    def test_no_quality_data_leaves_delta_none(self):
        profiles = [
            make_profile("fp16", quality=None),
            make_profile("fp8", quality=None),
        ]
        from analyze.deployment_advisor import compute_tradeoff
        table = compute_tradeoff(profiles, "fp16")
        fp8 = next(r for r in table if r["tag"] == "fp8")
        assert fp8["quality_delta"] is None
```

- [ ] **Step 2: Run failing tests**

```bash
pytest tests/test_deployment_advisor.py::TestComputeTradeoff -v 2>&1 | head -20
```

Expected: All fail with `ImportError` or `AttributeError` — `compute_tradeoff` doesn't exist yet.

- [ ] **Step 3: Add `compute_tradeoff` to `analyze/deployment_advisor.py`**

Add after `load_deployment`:

```python
def compute_tradeoff(profiles, baseline_tag):
    """
    Compute relative latency, quality, and cost metrics for each profile vs baseline.
    Returns a list of tradeoff row dicts suitable for recommend() and render().
    """
    baseline = next((p for p in profiles if p["tag"] == baseline_tag), None)
    if baseline is None:
        print("ERROR: baseline tag '{}' not in profiles".format(baseline_tag), file=sys.stderr)
        sys.exit(1)

    datasets = {p["_dataset"] for p in profiles if p["_dataset"]}
    if len(datasets) > 1:
        print(
            "ERROR: profiles use different eval datasets: {}. "
            "Comparing quality scores from different datasets is invalid.".format(datasets),
            file=sys.stderr,
        )
        sys.exit(1)

    models = {p["model"] for p in profiles}
    if len(models) > 1:
        print(
            "ERROR: profiles use different models: {}. "
            "Comparing deployments of different models is invalid.".format(models),
            file=sys.stderr,
        )
        sys.exit(1)

    b_ttft = baseline["latency"]["ttft_ms_p50"]
    b_quality = baseline["quality"]["overall_score"] if baseline["quality"] else None
    b_cost = baseline["cost"]["per_million_tokens"]
    b_throughput = baseline["cost"]["throughput_proxy_tokens_per_sec"]

    rows = []
    for p in profiles:
        t_ttft = p["latency"]["ttft_ms_p50"]
        t_quality = p["quality"]["overall_score"] if p["quality"] else None
        t_cost = p["cost"]["per_million_tokens"]
        t_throughput = p["cost"]["throughput_proxy_tokens_per_sec"]

        lat_improvement = (b_ttft - t_ttft) / b_ttft if b_ttft else None

        quality_delta = None
        if b_quality is not None and t_quality is not None:
            quality_delta = t_quality - b_quality

        cost_reduction = None
        if b_cost is not None and t_cost is not None and b_cost != 0:
            cost_reduction = (b_cost - t_cost) / b_cost
        elif b_throughput and t_throughput and t_throughput != 0:
            cost_reduction = (t_throughput - b_throughput) / t_throughput

        rows.append({
            "tag": p["tag"],
            "is_baseline": p["tag"] == baseline_tag,
            "ttft_ms_p50": t_ttft,
            "throughput_tokens_per_sec": p["latency"]["throughput_tokens_per_sec"],
            "quality_score": t_quality,
            "cost_per_million": t_cost,
            "latency_improvement": lat_improvement,
            "quality_delta": quality_delta,
            "cost_reduction": cost_reduction,
        })

    return rows
```

- [ ] **Step 4: Run tests — verify all pass**

```bash
pytest tests/test_deployment_advisor.py::TestComputeTradeoff -v
```

Expected: All 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add analyze/deployment_advisor.py tests/test_deployment_advisor.py
git commit -m "feat: add compute_tradeoff with latency/quality/cost deltas"
```

---

## Task 9: `analyze/deployment_advisor.py` — `recommend` and `render`

**Files:**
- Modify: `analyze/deployment_advisor.py`
- Modify: `tests/test_deployment_advisor.py`

- [ ] **Step 1: Add failing tests for `recommend` and `render`**

Append to `tests/test_deployment_advisor.py`:

```python
def make_row(tag, is_baseline=False, lat_imp=0.0, q_delta=0.0, q_score=0.90,
             cost=1.0, ttft=115, throughput=262):
    return {
        "tag": tag,
        "is_baseline": is_baseline,
        "ttft_ms_p50": ttft,
        "throughput_tokens_per_sec": throughput,
        "quality_score": q_score,
        "cost_per_million": cost,
        "latency_improvement": lat_imp,
        "quality_delta": q_delta,
        "cost_reduction": 0.10,
    }


class TestRecommend:
    def test_picks_fastest_within_threshold(self):
        table = [
            make_row("fp16", is_baseline=True, q_score=0.95),
            make_row("fp8", lat_imp=0.33, q_delta=-0.017, q_score=0.933),
            make_row("int4", lat_imp=0.67, q_delta=-0.179, q_score=0.771),
        ]
        from analyze.deployment_advisor import recommend
        best, eliminated, warning = recommend(table, quality_threshold=0.10)
        assert best["tag"] == "fp8"
        assert len(eliminated) == 1
        assert eliminated[0]["tag"] == "int4"

    def test_recommends_baseline_when_all_alternatives_eliminated(self):
        table = [
            make_row("fp16", is_baseline=True, q_score=0.95),
            make_row("int4", lat_imp=0.67, q_delta=-0.50, q_score=0.45),
        ]
        from analyze.deployment_advisor import recommend
        best, eliminated, warning = recommend(table, quality_threshold=0.10)
        assert best["tag"] == "fp16"
        assert len(eliminated) == 1

    def test_warns_when_no_quality_data(self):
        table = [
            make_row("fp16", is_baseline=True, q_score=None, q_delta=None),
            make_row("fp8", lat_imp=0.33, q_score=None, q_delta=None),
        ]
        # Patch quality_delta to None explicitly
        for row in table:
            row["quality_delta"] = None
        from analyze.deployment_advisor import recommend
        best, eliminated, warning = recommend(table, quality_threshold=0.10)
        assert warning is not None
        assert "latency only" in warning.lower()
        assert best["tag"] == "fp8"

    def test_no_alternatives_returns_baseline(self):
        table = [make_row("fp16", is_baseline=True, q_score=0.95)]
        from analyze.deployment_advisor import recommend
        best, eliminated, warning = recommend(table, quality_threshold=0.10)
        assert best["tag"] == "fp16"


class TestRender:
    def _make_scenario(self):
        table = [
            make_row("fp16", is_baseline=True, q_score=0.95, cost=1.20, ttft=1200, throughput=210),
            make_row("fp8", lat_imp=0.333, q_delta=-0.017, q_score=0.933, cost=0.91, ttft=800, throughput=290),
            make_row("int4", lat_imp=0.667, q_delta=-0.179, q_score=0.771, cost=0.50, ttft=400, throughput=520),
        ]
        from analyze.deployment_advisor import recommend
        best, eliminated, warning = recommend(table, quality_threshold=0.10)
        return table, best, eliminated, warning

    def test_markdown_contains_recommended_tag(self):
        table, best, eliminated, warning = self._make_scenario()
        from analyze.deployment_advisor import render
        output = render(table, best, eliminated, warning, "fp16", 0.10, "markdown")
        assert "fp8" in output
        assert "RECOMMENDED" in output

    def test_markdown_shows_eliminated_tag(self):
        table, best, eliminated, warning = self._make_scenario()
        from analyze.deployment_advisor import render
        output = render(table, best, eliminated, warning, "fp16", 0.10, "markdown")
        assert "int4" in output
        assert "eliminated" in output.lower()

    def test_json_output_is_valid_json(self):
        table, best, eliminated, warning = self._make_scenario()
        from analyze.deployment_advisor import render
        output = render(table, best, eliminated, warning, "fp16", 0.10, "json")
        data = json.loads(output)
        assert data["recommended"] == "fp8"
        assert "int4" in data["eliminated"]

    def test_markdown_shows_tradeoff_table(self):
        table, best, eliminated, warning = self._make_scenario()
        from analyze.deployment_advisor import render
        output = render(table, best, eliminated, warning, "fp16", 0.10, "markdown")
        assert "Tradeoff Table" in output
        assert "fp16" in output
        assert "fp8" in output
        assert "int4" in output
```

- [ ] **Step 2: Run failing tests**

```bash
pytest tests/test_deployment_advisor.py::TestRecommend tests/test_deployment_advisor.py::TestRender -v 2>&1 | head -20
```

Expected: All fail with ImportError — `recommend` and `render` don't exist yet.

- [ ] **Step 3: Add `recommend`, `render`, and helper formatters to `analyze/deployment_advisor.py`**

Add after `compute_tradeoff`:

```python
def recommend(tradeoff_table, quality_threshold=0.10):
    """
    Filter by quality threshold then rank survivors by latency improvement.
    Returns (recommended_row, eliminated_rows, warning_str_or_None).
    """
    has_quality = any(
        row["quality_delta"] is not None
        for row in tradeoff_table
        if not row["is_baseline"]
    )

    if not has_quality:
        candidates = sorted(
            [r for r in tradeoff_table if not r["is_baseline"]],
            key=lambda r: -(r["latency_improvement"] or 0),
        )
        best = candidates[0] if candidates else next(r for r in tradeoff_table if r["is_baseline"])
        return best, [], "No quality data available. Ranked by latency only."

    eliminated = []
    survivors = []
    for row in tradeoff_table:
        if row["is_baseline"]:
            survivors.append(row)
            continue
        delta = row["quality_delta"]
        if delta is not None and delta < -quality_threshold:
            eliminated.append(row)
        else:
            survivors.append(row)

    non_baseline = sorted(
        [r for r in survivors if not r["is_baseline"]],
        key=lambda r: -(r["latency_improvement"] or 0),
    )

    if not non_baseline:
        baseline = next(r for r in survivors if r["is_baseline"])
        return baseline, eliminated, "All alternatives eliminated. Baseline is the best available."

    return non_baseline[0], eliminated, None


def _fmt_pct(v):
    return "N/A" if v is None else "{:+.1f}%".format(v * 100)


def _fmt_ms(v):
    if v is None:
        return "N/A"
    return "{:.2f}s".format(v / 1000) if v >= 1000 else "{:.0f}ms".format(v)


def _fmt_cost(v):
    return "N/A" if v is None else "${:.2f}".format(v)


def render(tradeoff_table, recommended, eliminated, warning, baseline_tag,
           quality_threshold, output_format="markdown"):
    """Format recommendation as a terminal card (markdown) or JSON."""
    eliminated_tags = {r["tag"] for r in eliminated}

    if output_format == "json":
        return json.dumps({
            "recommended": recommended["tag"],
            "baseline": baseline_tag,
            "quality_threshold": quality_threshold,
            "warning": warning,
            "eliminated": list(eliminated_tags),
            "tradeoff_table": tradeoff_table,
        }, indent=2)

    lines = ["", "=== Deployment Recommendation ===", ""]
    if warning:
        lines += ["NOTE: {}".format(warning), ""]

    lines.append("Recommended: {}".format(recommended["tag"]))
    lines.append("")

    baseline_row = next(r for r in tradeoff_table if r["is_baseline"])
    if recommended["latency_improvement"] is not None:
        lines.append("  Latency Improvement:  {}  ({} → {})".format(
            _fmt_pct(recommended["latency_improvement"]),
            _fmt_ms(baseline_row["ttft_ms_p50"]),
            _fmt_ms(recommended["ttft_ms_p50"]),
        ))
    if recommended["cost_reduction"] is not None:
        cost_detail = ""
        if baseline_row["cost_per_million"] is not None and recommended["cost_per_million"] is not None:
            cost_detail = "  ({} → {} per 1M tokens)".format(
                _fmt_cost(baseline_row["cost_per_million"]),
                _fmt_cost(recommended["cost_per_million"]),
            )
        lines.append("  Cost Reduction:       {}{}".format(
            _fmt_pct(recommended["cost_reduction"]),
            "  ({} → {} per 1M tokens)".format(
                _fmt_cost(baseline_row["cost_per_million"]),
                _fmt_cost(recommended["cost_per_million"]),
            ) if cost_detail else "",
        ))
    if recommended["quality_delta"] is not None:
        baseline_q = recommended["quality_score"] - recommended["quality_delta"]
        lines.append("  Quality Delta:        {}  ({:.3f} → {:.3f})".format(
            _fmt_pct(recommended["quality_delta"]),
            baseline_q,
            recommended["quality_score"],
        ))

    lines.append("")
    for e in eliminated:
        lines.append("Eliminated: {} — quality drop {:.1f}% exceeds threshold ({:.1f}%)".format(
            e["tag"], abs(e["quality_delta"]) * 100, quality_threshold * 100,
        ))
    if eliminated:
        lines.append("")

    lines.append("Tradeoff Table:")
    col = "{:<28} {:>9} {:>7} {:>9} {:>10}  {}"
    lines.append("  " + col.format("Tag", "TTFT p50", "Tok/s", "Quality", "Cost/1M", "Status"))
    lines.append("  " + "-" * 78)
    for row in tradeoff_table:
        if row["is_baseline"]:
            status = "baseline"
        elif row["tag"] in eliminated_tags:
            status = "eliminated"
        elif row["tag"] == recommended["tag"]:
            status = "RECOMMENDED"
        else:
            status = "ok"
        q = "{:.3f}".format(row["quality_score"]) if row["quality_score"] is not None else "N/A"
        tok = str(int(row["throughput_tokens_per_sec"])) if row["throughput_tokens_per_sec"] else "N/A"
        lines.append("  " + col.format(
            row["tag"], _fmt_ms(row["ttft_ms_p50"]), tok, q,
            _fmt_cost(row["cost_per_million"]), status,
        ))

    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests — verify all pass**

```bash
pytest tests/test_deployment_advisor.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add analyze/deployment_advisor.py tests/test_deployment_advisor.py
git commit -m "feat: add recommend and render to deployment_advisor"
```

---

## Task 10: `analyze/deployment_advisor.py` — CLI Wiring

**Files:**
- Modify: `analyze/deployment_advisor.py` (add `main`)

- [ ] **Step 1: Add `main` to `analyze/deployment_advisor.py`**

Append to the end of the file:

```python
def main():
    ap = argparse.ArgumentParser(description="Quality-aware deployment advisor")
    ap.add_argument("--tags", nargs="+", required=True,
                    help="Tags to compare (must include --baseline)")
    ap.add_argument("--baseline", required=True,
                    help="Tag to use as the reference deployment")
    ap.add_argument("--quality-threshold", type=float, default=0.10,
                    dest="quality_threshold",
                    help="Max acceptable quality drop vs baseline (default: 0.10 = 10%%)")
    ap.add_argument("--output", choices=["markdown", "json"], default="markdown",
                    help="Output format (default: markdown)")
    ap.add_argument("--latency-dirs", nargs="+", dest="latency_dirs",
                    default=[SYN_DIR, REAL_DIR],
                    help="Dirs to search for latency results; later dirs override earlier")
    ap.add_argument("--quality-dir", dest="quality_dir", default=QUALITY_DIR,
                    help="Dir containing quality sidecar JSONs")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="Validate inputs without computing recommendation")
    args = ap.parse_args()

    if args.baseline not in args.tags:
        print("ERROR: --baseline '{}' must appear in --tags".format(args.baseline),
              file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("=== Dry run ===")
        print("  tags      : {}".format(args.tags))
        print("  baseline  : {}".format(args.baseline))
        print("  threshold : {}".format(args.quality_threshold))
        for tag in args.tags:
            lat = _find_latency_file(tag, args.latency_dirs)
            qual = os.path.join(args.quality_dir, tag + ".json")
            print("  {}: latency={} quality={}".format(
                tag,
                "found ({})".format(lat) if lat else "MISSING",
                "found" if os.path.isfile(qual) else "missing (will warn)",
            ))
        return

    profiles = [
        load_deployment(tag, args.latency_dirs, args.quality_dir)
        for tag in args.tags
    ]
    tradeoff_table = compute_tradeoff(profiles, args.baseline)
    recommended, eliminated, warning = recommend(tradeoff_table, args.quality_threshold)
    output = render(
        tradeoff_table, recommended, eliminated, warning,
        args.baseline, args.quality_threshold, args.output,
    )
    print(output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test with --dry-run against synthetic data**

```bash
python analyze/deployment_advisor.py \
  --tags vllm_l4fp8_isl2k_c10 vllm_h100fp16_isl2k_c10 \
  --baseline vllm_h100fp16_isl2k_c10 \
  --dry-run
```

Expected output (no errors):
```
=== Dry run ===
  tags      : ['vllm_l4fp8_isl2k_c10', 'vllm_h100fp16_isl2k_c10']
  baseline  : vllm_h100fp16_isl2k_c10
  threshold : 0.1
  vllm_l4fp8_isl2k_c10: latency=found (...) quality=missing (will warn)
  vllm_h100fp16_isl2k_c10: latency=found (...) quality=missing (will warn)
```

- [ ] **Step 3: Smoke test latency-only recommendation (no quality sidecars)**

```bash
python analyze/deployment_advisor.py \
  --tags vllm_l4fp8_isl2k_c10 vllm_h100fp16_isl2k_c10 \
  --baseline vllm_h100fp16_isl2k_c10
```

Expected: Prints recommendation card with "NOTE: No quality data available. Ranked by latency only." and a tradeoff table with quality "N/A" for all rows.

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass. No failures.

- [ ] **Step 5: Commit**

```bash
git add analyze/deployment_advisor.py
git commit -m "feat: add CLI wiring to deployment_advisor with dry-run support"
```

---

## Task 11: README Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add quality-aware benchmarking section to README**

Open `README.md`. The table of contents currently ends at section 10 (License). Add a new section **5.8** after the existing 5.7 (`results/`) entry in the TOC and add a new top-level section **10** (renumbering License to 11) near the end. Insert these two blocks:

**In the table of contents**, add after `5.7 \`results/\` — collected and reference data`:

```markdown
   - 5.8 [`evaluate/run_eval.py` — offline quality evaluator](#58-evaluaterun_evalpy--offline-quality-evaluator)
   - 5.9 [`analyze/deployment_advisor.py` — quality-aware deployment advisor](#59-analyzedeployment_advisorpy--quality-aware-deployment-advisor)
```

**Add a new section** after the existing section 5 components block:

```markdown
### 5.8 `evaluate/run_eval.py` — offline quality evaluator

Sends a small evaluation dataset (JSONL) at an inference endpoint, scores responses
with [DeepEval](https://github.com/confident-ai/deepeval), and writes a quality sidecar
JSON alongside the latency result.

```bash
python evaluate/run_eval.py \
  --endpoint http://localhost:8000/v1/completions \
  --model llama-3.1-8b \
  --latency-result results/real/vllm_l4fp8_isl2k_c10.json \
  --dataset datasets/rag.jsonl \
  --evaluator deepeval \
  --eval-model gpt-4o \
  --cost-per-million-tokens 0.80 \
  --output-dir results/quality
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--endpoint` | — | Inference endpoint under test |
| `--model` | — | Model name served at `--endpoint` |
| `--latency-result` | — | Path to latency JSON from `run_bench.py` |
| `--dataset` | — | JSONL eval dataset (`datasets/*.jsonl`) |
| `--evaluator` | `deepeval` | `deepeval` or `llm-judge` |
| `--eval-model` | `gpt-4o` | Judge model (separate from benchmarked endpoint) |
| `--eval-endpoint` | `https://api.openai.com/v1` | Judge endpoint |
| `--cost-per-million-tokens` | none | Cost per 1M tokens; omit to use throughput proxy |
| `--output-dir` | `./results/quality` | Where to write the quality sidecar |
| `--dry-run` | — | Validate inputs without hitting endpoint |

Auth for the judge endpoint uses the `OPENAI_API_KEY` environment variable.

Eval datasets ship with the repo in `datasets/`. Each JSONL file has rows of:
```json
{"schema_version": 1, "id": "...", "workload": "rag", "prompt": "...", "expected": "..."}
```

### 5.9 `analyze/deployment_advisor.py` — quality-aware deployment advisor

Merges latency results with quality sidecars for multiple deployments and recommends the
best configuration by balancing latency improvement, cost reduction, and quality retention.

```bash
python analyze/deployment_advisor.py \
  --tags vllm_a100fp16 vllm_l4fp8 vllm_l4int4 \
  --baseline vllm_a100fp16 \
  --quality-threshold 0.10
```

**Example output:**

```
=== Deployment Recommendation ===

Recommended: vllm_l4fp8

  Latency Improvement:  +33.3%  (1.20s → 800ms)
  Cost Reduction:       +24.2%  ($1.20 → $0.91 per 1M tokens)
  Quality Delta:        -1.7%   (0.950 → 0.933)

Eliminated: vllm_l4int4 — quality drop 17.9% exceeds threshold (10.0%)

Tradeoff Table:
  Tag                          TTFT p50   Tok/s   Quality   Cost/1M  Status
  ------------------------------------------------------------------------
  vllm_a100fp16                   1.20s     210     0.950     $1.20  baseline
  vllm_l4fp8                       800ms    290     0.933     $0.91  RECOMMENDED
  vllm_l4int4                      400ms    520     0.771     $0.50  eliminated
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--tags` | — | Space-separated tags to compare |
| `--baseline` | — | Tag to use as reference (must be in `--tags`) |
| `--quality-threshold` | `0.10` | Max quality drop before elimination (10%) |
| `--output` | `markdown` | `markdown` or `json` |
| `--dry-run` | — | Validate inputs without computing |

If no quality sidecar exists for a tag, it is included in the table with quality "N/A"
and excluded from quality-gated ranking (latency-only fallback with a warning).
```

- [ ] **Step 2: Verify README renders correctly**

```bash
python -c "
import re
with open('README.md') as f:
    content = f.read()
assert 'run_eval.py' in content, 'missing run_eval section'
assert 'deployment_advisor.py' in content, 'missing advisor section'
print('README check passed')
"
```

Expected: `README check passed`

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add quality-aware benchmarking section to README"
```

---

## Task 12: Full Test Suite and Integration Smoke Test

**Files:**
- No new files

- [ ] **Step 1: Run the complete test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests pass. Note the exact count for verification:
- `tests/test_run_eval.py`: 17 tests
- `tests/test_deployment_advisor.py`: 19 tests
- Total: 36 tests, 0 failures

- [ ] **Step 2: Integration smoke test — latency-only advisor with synthetic data**

```bash
python analyze/deployment_advisor.py \
  --tags vllm_l4fp8_isl2k_c10 vllm_l4fp8_isl2k_c50 vllm_h100fp16_isl2k_c50 \
  --baseline vllm_h100fp16_isl2k_c50
```

Expected: Prints a tradeoff table with all quality fields "N/A" and a latency-only recommendation note. No errors.

- [ ] **Step 3: Integration smoke test — dry-run for run_eval**

```bash
python evaluate/run_eval.py \
  --endpoint http://localhost:8000/v1/completions \
  --model llama-3.1-8b \
  --latency-result results/synthetic/vllm_l4fp8_isl2k_c10.json \
  --dataset datasets/chat.jsonl \
  --dry-run
```

Expected:
```
=== Dry run ===
  latency-result : results/synthetic/vllm_l4fp8_isl2k_c10.json
  latency-tag    : vllm_l4fp8_isl2k_c10
  dataset        : datasets/chat.jsonl (15 samples, workload=chat)
  evaluator      : deepeval
  eval-model     : gpt-4o
  eval-endpoint  : https://api.openai.com/v1
  output-tag     : vllm_l4fp8_isl2k_c10
  output-dir     : ./results/quality
Would collect responses and run evaluation. Exiting (--dry-run).
```

- [ ] **Step 4: Commit integration smoke test results**

```bash
git add .
git status  # verify only README.md changes or nothing new
git commit -m "feat: quality-aware benchmarking pipeline complete" --allow-empty
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `evaluate/run_eval.py` with DeepEval | Tasks 3–6 |
| `analyze/deployment_advisor.py` with four pure functions | Tasks 7–10 |
| `datasets/chat.jsonl`, `rag.jsonl`, `long_context.jsonl` | Task 2 |
| `results/quality/.gitkeep` | Task 1 |
| `requirements.txt`: add deepeval | Task 1 |
| `README.md` update | Task 11 |
| DeploymentProfile normalization (flattens nested JSON) | Task 7 |
| `latency_tag` mismatch → hard error | Task 7 |
| Mismatched model → hard error | Task 8 |
| Mismatched dataset → hard error | Task 8 |
| Real overrides synthetic | Task 7 |
| `normalize_score` inverts hallucination rate | Task 3 |
| `--eval-model` / `--eval-endpoint` flags | Task 6 |
| `--dry-run` on both scripts | Tasks 6, 10 |
| LLM-judge optional path | Task 6 |
| Cost: `--cost-per-million-tokens` primary, throughput proxy fallback | Tasks 5, 8 |
| Quality threshold filter then rank by latency | Task 9 |
| `--output json` | Task 9 |
| `analyze/__init__.py`, `evaluate/__init__.py` | Task 1 |
| Existing files untouched | All tasks — confirmed no edits to run_bench.py, report.py, advisor.py |

**No gaps found.**
