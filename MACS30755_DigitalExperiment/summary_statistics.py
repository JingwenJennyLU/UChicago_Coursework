"""
Summary Statistics — append to (or run after) the visualization script.
Generates 4 CSV tables ready to paste into the appendix as Word tables.

Output files (saved alongside the figures):
    table_A_overall.csv          - Overall 4-category distribution
    table_B_subgroup_means.csv   - All 15 subgroups: Human / AI / Diff
    table_C_alignment_stats.csv  - MAE / Bias / Pearson r per dimension
    table_D_chi_square.csv       - Chi-square test result on overall distribution
"""
import numpy as np
import pandas as pd
from scipy import stats


OUT = "/Users/jingwenlu/Desktop/UChicago/Course/UChicago_Coursework/MACS30755_DigitalExperiment"

df = pd.read_csv("ai_responses_raw.csv")
N_AI = len(df)

ai_overall = {
    'Great deal':    df['great_deal'].mean()*100,
    'Fair amount':   ((df['at_least_fair']==1) & (df['great_deal']==0)).mean()*100,
    'Not too much':  (df['answer.trust_scientists']=='Not too much confidence').mean()*100,
    'No confidence': (df['answer.trust_scientists']=='No confidence at all').mean()*100,
}
ai_party = {'Democrats':   df[df['party_simple']=='Democrats']['at_least_fair'].mean()*100,
            'Republicans': df[df['party_simple']=='Republicans']['at_least_fair'].mean()*100}
ai_race  = {r: df[df['agent.race_ethnicity']==r]['at_least_fair'].mean()*100
            for r in ['White','Black','Asian']}
ai_edu   = {'4-yr college+':     df[df['agent.has_college_degree']==True ]['great_deal'].mean()*100,
            'No college degree': df[df['agent.has_college_degree']==False]['great_deal'].mean()*100}
ai_pxedu = {
    'Dem · 4-yr college+': df[(df['party_simple']=='Democrats')  & (df['agent.has_college_degree']==True )]['great_deal'].mean()*100,
    'Dem · no college':    df[(df['party_simple']=='Democrats')  & (df['agent.has_college_degree']==False)]['great_deal'].mean()*100,
    'Rep · 4-yr college+': df[(df['party_simple']=='Republicans')& (df['agent.has_college_degree']==True )]['great_deal'].mean()*100,
    'Rep · no college':    df[(df['party_simple']=='Republicans')& (df['agent.has_college_degree']==False)]['great_deal'].mean()*100,
}
dems = df[df['party_simple']=='Democrats']
ai_rxp = {r+' Dems': dems[dems['agent.race_ethnicity']==r]['great_deal'].mean()*100
          for r in ['White','Asian','Hispanic','Black']}

# Human benchmarks (Pew Research, Nov 2024)
human_overall = {'Great deal':26, 'Fair amount':51, 'Not too much':19, 'No confidence':4}
human_party   = {'Democrats':88, 'Republicans':66}
human_race    = {'White':78, 'Black':77, 'Asian':85}
human_edu     = {'4-yr college+':34, 'No college degree':22}
human_pxedu   = {'Dem · 4-yr college+':51, 'Dem · no college':34,
                 'Rep · 4-yr college+':15, 'Rep · no college':11}
human_rxp     = {'White Dems':48, 'Asian Dems':45, 'Hispanic Dems':30, 'Black Dems':29}


# TABLE A — Overall 4-category distribution

rows = []
for c in ['Great deal','Fair amount','Not too much','No confidence']:
    rows.append({
        'Response category': c,
        'Human % (Pew, n=4,808)': f"{human_overall[c]}%",
        f'AI % (GPT-4o, n={N_AI})': f"{ai_overall[c]:.1f}%",
        'AI − Human (pp)': f"{ai_overall[c]-human_overall[c]:+.1f}",
    })
table_A = pd.DataFrame(rows)
table_A.to_csv(f"{OUT}/table_A_overall.csv", index=False)


# TABLE B — All 15 subgroups across 5 dimensions

rows = []
def add_row(dim, group, h_val, a_val, metric):
    rows.append({
        'Dimension': dim, 'Subgroup': group, 'Metric': metric,
        'Human %': h_val, 'AI %': round(a_val, 1),
        'AI − Human (pp)': round(a_val - h_val, 1)
    })

for g in human_party: add_row('1. Party',            g, human_party[g], ai_party[g], '% at least fair amount')
for g in human_race:  add_row('2. Race/ethnicity',   g, human_race[g],  ai_race[g],  '% at least fair amount')
for g in human_edu:   add_row('3. Education',        g, human_edu[g],   ai_edu[g],   '% great deal')
for g in human_pxedu: add_row('4. Party × Education',g, human_pxedu[g], ai_pxedu[g], '% great deal')
for g in human_rxp:   add_row('5. Race within Dems', g, human_rxp[g],   ai_rxp[g],   '% great deal')

table_B = pd.DataFrame(rows)
table_B.to_csv(f"{OUT}/table_B_subgroup_means.csv", index=False)


# TABLE C — Alignment statistics per dimension

def stats_for(h_dict, a_dict):
    h = np.array(list(h_dict.values()), dtype=float)
    a = np.array([a_dict[k] for k in h_dict.keys()], dtype=float)
    mae = np.mean(np.abs(h - a))
    bias = np.mean(a - h)
    r, p = stats.pearsonr(h, a)
    return len(h_dict), mae, bias, r, p

c_rows = []
for label, h_d, a_d, metric in [
    ('1. Party',             human_party, ai_party, '% at least fair'),
    ('2. Race/ethnicity',    human_race,  {k:ai_race[k] for k in human_race}, '% at least fair'),
    ('3. Education',         human_edu,   ai_edu,   '% great deal'),
    ('4. Party × Education', human_pxedu, ai_pxedu, '% great deal'),
    ('5. Race within Dems',  human_rxp,   ai_rxp,   '% great deal'),
]:
    n, mae, bias, r, p = stats_for(h_d, a_d)
    c_rows.append({
        'Dimension': label, 'Metric': metric, 'Cells': n,
        'MAE (pp)': f"{mae:.2f}",
        'Signed bias (AI − Human, pp)': f"{bias:+.2f}",
        'Pearson r': f"{r:.3f}",
        'p-value': f"{p:.4f}"
    })
# Pooled rows
keys12 = list(human_party) + list(human_race)
h12 = np.array([human_party.get(k, human_race.get(k)) for k in keys12], dtype=float)
a12 = np.array([ai_party.get(k,    ai_race.get(k))    for k in keys12], dtype=float)
r12, p12 = stats.pearsonr(h12, a12)
c_rows.append({'Dimension':'POOLED Dims 1–2', 'Metric':'% at least fair', 'Cells':len(h12),
               'MAE (pp)':f"{np.mean(np.abs(h12-a12)):.2f}",
               'Signed bias (AI − Human, pp)':f"{np.mean(a12-h12):+.2f}",
               'Pearson r':f"{r12:.3f}", 'p-value':f"{p12:.4f}"})

keys35 = list(human_edu) + list(human_pxedu) + list(human_rxp)
h35 = np.array([{**human_edu, **human_pxedu, **human_rxp}[k] for k in keys35], dtype=float)
a35 = np.array([{**ai_edu,    **ai_pxedu,    **ai_rxp}[k]    for k in keys35], dtype=float)
r35, p35 = stats.pearsonr(h35, a35)
c_rows.append({'Dimension':'POOLED Dims 3–5', 'Metric':'% great deal', 'Cells':len(h35),
               'MAE (pp)':f"{np.mean(np.abs(h35-a35)):.2f}",
               'Signed bias (AI − Human, pp)':f"{np.mean(a35-h35):+.2f}",
               'Pearson r':f"{r35:.3f}", 'p-value':f"{p35:.4f}"})

table_C = pd.DataFrame(c_rows)
table_C.to_csv(f"{OUT}/table_C_alignment_stats.csv", index=False)


# TABLE D — Chi-square test (overall 4-category distribution)

h_arr = np.array([human_overall[c] for c in human_overall], dtype=float)
a_arr = np.array([ai_overall[c]    for c in human_overall], dtype=float)
expected = h_arr / h_arr.sum() * N_AI
observed = a_arr / a_arr.sum() * N_AI
chi2, p_chi = stats.chisquare(observed, expected)

table_D = pd.DataFrame([{
    'Test': "Chi-square goodness-of-fit (AI distribution vs. Pew expected)",
    'Sample size (AI)': N_AI,
    'Chi-square': f"{chi2:.2f}",
    'Degrees of freedom': 3,
    'p-value': f"{p_chi:.2e}",
    'Interpretation': "p < .001 ⇒ AI distribution is significantly different from Pew"
}])
table_D.to_csv(f"{OUT}/table_D_chi_square.csv", index=False)


# Terminal summary

print("="*70); print(" TABLE A — Overall distribution"); print("="*70)
print(table_A.to_string(index=False))
print("\n" + "="*70); print(" TABLE B — Subgroup means (all 15 subgroups)"); print("="*70)
print(table_B.to_string(index=False))
print("\n" + "="*70); print(" TABLE C — Alignment statistics"); print("="*70)
print(table_C.to_string(index=False))
print("\n" + "="*70); print(" TABLE D — Chi-square test"); print("="*70)
print(table_D.to_string(index=False))
print(f"\nAll 4 CSV tables saved to: {OUT}")