# CHANGES.md — KsaaiP2 Pipeline

## v4_updated (April 2026) — Three gap fixes applied

### Fix A:  and  added to 

Six automated post-run self-consistency checks:

| # | Check | Pass condition |
|---|-------|---------------|
| 1 | DC beta bounds | beta in [1.90, 2.10] |
| 2 | xi_n timing dominance | xi_n median < DC median for all n >= 1K |
| 3 | HK gene count | TCGA HK gene list >= 3 genes |
| 4 | MIC CoV | CoV <= 0.30 at all n |
| 5 | SC R2 consistency | SuperConductivity log-log R2 >= 0.990 |
| 6 | INDPRO target | FRED-MD target_series == INDPRO |

### Fix B:  wired into 

Replaced hardcoded constants with  key lookups.
Pre-emptive INDPRO validation: exits code 1 if target != INDPRO.

### Fix B:  output paths wired into 

Replaced all hardcoded output CSV filenames with  lookups.

###  — new keys added

All output path keys added under  section.
Added  and  keys.

---

## v4 (April 2026) — Reviewer response patch

- MIC subprocess isolation:  +  + 
-  dict in every output table
-  in scorer registry
-  with delta-method 95% PI
- dcor -> |Pearson r| fallback
-  with pytest
- 
- 
-  (Scenarios D & E)
- Jaccard stability B=200 in 
- Open-science checklist and system state block in 

---

## v3 (March 2026)

- Initial three-script version (00_, 02_, 04_)
- config.yaml introduced (partially wired)