import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

output = pd.read_pickle("./data/results/benchmark_1_llama3.pickle")
output['Decision'] = output['Decision'].map({'Ja': 1, 'Nee': 0})

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
myFig = plt.figure()
ax = df['C(Gender)[T.non-binaire]'].plot.box()
myFig.savefig("test100.jpg")
