from langchain_community.llms import Ollama
import pandas as pd

output = pd.read_pickle("./data/results/benchmark_1_llama3.pickle")
print(output)
