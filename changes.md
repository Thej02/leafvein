# CHANGES.md — Running Change Log

> Append-only log. Every implementation decision, deviation from `implementation_plan.md`,
> calibrated threshold value, rejected data sample, or design change goes here, in
> reverse-chronological order (newest at top). `goal.md` is never changed, so nothing
> here should ever describe a change to the goal itself — only to how we get there.

## How to log an entry
Copy this template for each change:

```
## YYYY-MM-DD — <short title>
**Stage:** <which implementation_plan.md stage this touches>
**What changed:** <one or two sentences>
**Why:** <reasoning>
**Affected files:** <paths>
**Follow-up needed:** <yes/no — describe if yes>
```

---

## 2026-08-20 — Initial planning documents created
**Stage:** Stage 0 (pre-work)
**What changed:** Created `goal.md`, `implementation_plan.md`, and this `changes.md`
file based on Phase-I report review. Confirmed project will use a rule-based /
classical image-processing pipeline instead of a trained ML classifier, per explicit
user requirement.
**Why:** Phase-I report's original "AI/ML model" framing is being replaced with a
transparent, interpretable, non-ML rule engine to (a) avoid the dataset-size
limitation the report already admits to, (b) make the non-invasive/low-cost claims
in the abstract concretely true, and (c) produce explainable output tied to real
botanical signals rather than a black-box prediction.
**Affected files:** `goal.md`, `implementation_plan.md`, `changes.md`
**Follow-up needed:** Yes — Stage 2 (capture protocol) and Stage 3 (data collection)
have not started yet. Next entry should log the finalized capture protocol once
tested with a real leaf.

---

## 2026-08-20 — Novelty check performed, goal.md amended (not the core goal itself)
**Stage:** Stage 0 (pre-work) / goal validation
**What changed:** Ran a literature search to verify the project's novelty before
continuing implementation, per explicit user request. Confirmed two gaps: (1) no
existing image-based nutrient-deficiency detection work targets Hibiscus
rosa-sinensis specifically — existing studies target rice, mango, grape, citrus,
coffee, tomato, black gram, apple; (2) the field is dominated by ML/DL classifiers,
with classical vein/edge detection (e.g. Canny) used only as ML preprocessing, not
as a standalone interpretable decision engine. Added a "Novelty Positioning" section
to `goal.md` documenting this so it isn't lost and can be cited in the Phase-II
report/viva. Did not change any of the original Non-Negotiable Constraints, Success
Criteria, or Out-of-Scope items — the existing rule-based, non-ML plan already
matches what makes the project novel, so `implementation_plan.md` requires no
structural changes as a result of this check.
**Why:** User asked for the project to be verified as novel and for plans to change
if it wasn't. Research confirmed it is novel as currently scoped, so the correct
action was to document the finding, not alter the pipeline design.
**Affected files:** `goal.md` (new section added), `changes.md`
**Follow-up needed:** No immediate follow-up. When writing the Phase-I→Phase-II
literature-gap section (2.17 in the existing report) for final submission, cite this
novelty framing explicitly — it strengthens the "gap addressed" argument already
present in that section.

<!-- New entries go above this line -->
