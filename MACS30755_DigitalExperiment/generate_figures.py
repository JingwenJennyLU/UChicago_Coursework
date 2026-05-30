"""
Visualization script: side-by-side Human vs. AI comparisons.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size']   = 10
plt.rcParams['axes.spines.top']  = False
plt.rcParams['axes.spines.right'] = False

HUMAN_COLOR = "#1F4E79"
AI_COLOR    = "#C00000"
OUT = "/Users/jingwenlu/Desktop/UChicago/Course/UChicago_Coursework/MACS30755_DigitalExperiment"

# Load AI data
df = pd.read_csv("ai_responses_raw.csv")


# Compute AI percentages

ai_overall = {
    'Great deal':    df['great_deal'].mean()*100,
    'Fair amount':   ((df['at_least_fair']==1) & (df['great_deal']==0)).mean()*100,
    'Not too much':  (df['answer.trust_scientists']=='Not too much confidence').mean()*100,
    'No confidence': (df['answer.trust_scientists']=='No confidence at all').mean()*100,
}
ai_party = {
    'Democrats':   df[df['party_simple']=='Democrats']['at_least_fair'].mean()*100,
    'Republicans': df[df['party_simple']=='Republicans']['at_least_fair'].mean()*100,
}
ai_race  = {r: df[df['agent.race_ethnicity']==r]['at_least_fair'].mean()*100
            for r in ['White','Black','Asian']}
ai_edu   = {'4-yr college+':     df[df['agent.has_college_degree']==True]['great_deal'].mean()*100,
            'No college degree': df[df['agent.has_college_degree']==False]['great_deal'].mean()*100}
ai_pxedu = {
    'Dem · 4-yr college+': df[(df['party_simple']=='Democrats') & (df['agent.has_college_degree']==True )]['great_deal'].mean()*100,
    'Dem · no college':    df[(df['party_simple']=='Democrats') & (df['agent.has_college_degree']==False)]['great_deal'].mean()*100,
    'Rep · 4-yr college+': df[(df['party_simple']=='Republicans')&(df['agent.has_college_degree']==True )]['great_deal'].mean()*100,
    'Rep · no college':    df[(df['party_simple']=='Republicans')&(df['agent.has_college_degree']==False)]['great_deal'].mean()*100,
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


# FIGURE 1: Overall 4-category distribution

fig, ax = plt.subplots(figsize=(7.8, 4.2))
cats = ['Great deal', 'Fair amount', 'Not too much', 'No confidence']
h = [human_overall[c] for c in cats]
a = [ai_overall[c]    for c in cats]
x = np.arange(len(cats)); w = 0.36
ax.bar(x-w/2, h, w, color=HUMAN_COLOR, label='Human (Pew, n=4,808)', edgecolor='white')
ax.bar(x+w/2, a, w, color=AI_COLOR,    label='AI (GPT-4o/EDSL, n=800)', edgecolor='white')
for i,(hv,av) in enumerate(zip(h,a)):
    ax.text(i-w/2, hv+1, f"{hv}%",   ha='center', fontsize=9)
    ax.text(i+w/2, av+1, f"{av:.1f}%", ha='center', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel("% of respondents"); ax.set_ylim(0, 90)
ax.legend(frameon=False, loc='upper right')
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_overall.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.close()


# FIGURE 2: Party + Race comparison (% at least fair amount)

groups = list(human_party) + list(human_race)
h_vals = [human_party[g] if g in human_party else human_race[g] for g in groups]
a_vals = [ai_party[g]    if g in ai_party    else ai_race[g]    for g in groups]

fig, ax = plt.subplots(figsize=(8.5, 4.4))
y = np.arange(len(groups))[::-1]
ax.barh(y-0.2, h_vals, 0.38, color=HUMAN_COLOR, label='Human (Pew)', edgecolor='white')
ax.barh(y+0.2, a_vals, 0.38, color=AI_COLOR,    label='AI (GPT-4o)', edgecolor='white')
for i,(hv,av) in enumerate(zip(h_vals,a_vals)):
    ax.text(hv+1, y[i]-0.2, f"{hv}%",   va='center', fontsize=9)
    ax.text(av+1, y[i]+0.2, f"{av:.1f}%", va='center', fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(groups)
ax.set_xlabel("% expressing 'great deal' or 'fair amount' of confidence in scientists")
ax.set_xlim(0, 115)
ax.axhline(len(groups)-2-0.5, color='#cccccc', linewidth=0.6, linestyle='--')
ax.legend(frameon=False, loc='lower right', bbox_to_anchor=(1.0, -0.30), ncol=2)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_party_race.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.close()

# FIGURE 3: Education + interactions (% great deal)
groups = list(human_edu) + list(human_pxedu) + list(human_rxp)
def hv(g):
    return human_edu.get(g, human_pxedu.get(g, human_rxp.get(g)))
def av(g):
    return ai_edu.get(g, ai_pxedu.get(g, ai_rxp.get(g)))
h_vals = [hv(g) for g in groups]
a_vals = [av(g) for g in groups]

fig, ax = plt.subplots(figsize=(8.5, 5.6))
y = np.arange(len(groups))[::-1]
ax.barh(y-0.2, h_vals, 0.38, color=HUMAN_COLOR, label='Human (Pew)', edgecolor='white')
ax.barh(y+0.2, a_vals, 0.38, color=AI_COLOR,    label='AI (GPT-4o)', edgecolor='white')
for i,(hvv,avv) in enumerate(zip(h_vals,a_vals)):
    ax.text(hvv+0.4, y[i]-0.2, f"{hvv}%",   va='center', fontsize=8.5)
    ax.text(avv+0.4, y[i]+0.2, f"{avv:.1f}%", va='center', fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(groups, fontsize=9)
ax.set_xlabel("% expressing 'a great deal' of confidence in scientists")
ax.set_xlim(0, 65)
ax.axhline(len(groups)-2-0.5, color='#cccccc', linewidth=0.6, linestyle='--')
ax.axhline(len(groups)-6-0.5, color='#cccccc', linewidth=0.6, linestyle='--')
ax.legend(frameon=False, loc='lower right', bbox_to_anchor=(1.0, -0.20), ncol=2)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_education.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.close()


# FIGURE 4: Signed errors, sorted
all_keys = list(human_party) + list(human_race) + list(human_edu) + \
           list(human_pxedu) + list(human_rxp)
all_h = ([human_party[k]    for k in human_party] +
         [human_race[k]     for k in human_race] +
         [human_edu[k]      for k in human_edu] +
         [human_pxedu[k]    for k in human_pxedu] +
         [human_rxp[k]      for k in human_rxp])
all_a = ([ai_party[k]    for k in human_party] +
         [ai_race[k]     for k in human_race] +
         [ai_edu[k]      for k in human_edu] +
         [ai_pxedu[k]    for k in human_pxedu] +
         [ai_rxp[k]      for k in human_rxp])
errs = np.array(all_a) - np.array(all_h)
order = np.argsort(errs)
labs = [all_keys[i] for i in order]
errs_sorted = errs[order]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
colors = [AI_COLOR if e>0 else HUMAN_COLOR for e in errs_sorted]
ax.barh(range(len(labs)), errs_sorted, color=colors, edgecolor='white')
for i,e in enumerate(errs_sorted):
    ax.text(e + (0.6 if e>=0 else -0.6), i, f"{e:+.1f}",
            va='center', ha='left' if e>=0 else 'right', fontsize=8.5)
ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=9)
ax.axvline(0, color='black', linewidth=0.7)
ax.set_xlabel("AI − Human  (percentage points)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_errors.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.close()


# FIGURE 5: Scatter — human vs AI alignment
fig, ax = plt.subplots(figsize=(6.2, 6.2))
hv = np.array(all_h); av = np.array(all_a)
ax.scatter(hv, av, s=70, color=AI_COLOR, alpha=0.75, edgecolor='black', linewidth=0.5)
for k, h_, a_ in zip(all_keys, hv, av):
    ax.annotate(k, (h_, a_), textcoords='offset points', xytext=(5,4), fontsize=7.2)
ax.plot([0,100], [0,100], 'k--', linewidth=1, label='Perfect agreement (y = x)')
slope, intercept, r_val, _, _ = stats.linregress(hv, av)
xx = np.linspace(0, 100, 50)
ax.plot(xx, slope*xx + intercept, color=HUMAN_COLOR, linewidth=1.5,
        label=f'Best fit (r = {r_val:.2f})')
ax.set_xlabel("Human % (Pew)"); ax.set_ylabel("AI % (GPT-4o/EDSL)")
ax.set_xlim(0, 105); ax.set_ylim(-5, 105)
ax.legend(frameon=False, loc='lower right')
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_scatter.png", dpi=180, bbox_inches='tight', facecolor='white')
plt.close()

print("All 5 figures saved successfully to", OUT)