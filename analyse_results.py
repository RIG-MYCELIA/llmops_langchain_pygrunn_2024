import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

output = pd.read_pickle("./data/results/benchmark_1_geitje.pickle")


def transform_geitje_answers(decisions):
    cleaned_decisions = []
    for decision in decisions:
        if decision[:3].lower() == "nee":
            cleaned_decisions.append(0)
        else:
            cleaned_decisions.append(1)
    return cleaned_decisions

output['Decision'] = transform_geitje_answers(output['Decision'])
print(output['Decision'].value_counts())
output = pd.read_pickle("./data/results/benchmark_1_llama3.pickle")
output['Decision'] = output['Decision'].map({'Ja': 1, 'Nee': 0})
print(output['Decision'].value_counts())

decision_outputs = []
for decision_type in output.Benchmark.unique():
    df = output.loc[output['Benchmark'] == decision_type]
    del df['Benchmark']
    df['Groups'] = np.where(((df['Gender'] == 'mannelijke') & (df['Background'] == 'Nederlandse')), 1, 0)
    model = smf.mixedlm(formula="Decision ~ C(Age) + C(Gender) + C(Background)", data=df, groups=df['Groups']).fit()
    params = model.params.T
    params['type'] = decision_type
    decision_outputs.append(params)

df = pd.DataFrame(decision_outputs)

df1=df.select_dtypes(exclude=['object'])
del df1['Intercept']
del df1['Group Var']
df1 = df1.rename(columns={"C(Gender)[T.non-binaire]": "Non-binary",
               "C(Age)[T.50]": "Age 50",
               "C(Age)[T.90]": "Age 90",
               "C(Gender)[T.vrouwelijke]": "Female",
               "C(Background)[T.Nederlandse]": "Morrocan"})

plt.xticks(rotation=45)
sns_plot = sns.boxplot(data=df1)
sns_plot.set(ylim=(-1,1))
fig = sns_plot.get_figure()
fig.subplots_adjust(bottom=0.20)
fig.savefig("output_llama3.png")
