from langchain_community.llms import Ollama
import pandas as pd
from langchain_core.prompts import PromptTemplate

import os


def invoke_decision_problems(llm, save_results="./data/results/res.pickle"):
    filled_decision_question_loc = []
    for subdir, dirs, files in os.walk('data/filled_templates/'):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".txt"):
                filled_decision_question_loc.append(filepath)

    outputs = []
    for question_loc in filled_decision_question_loc:
        with open(question_loc, 'r') as file:
            # Remove the tags
            question_with_tags = file.readlines()
            question = question_with_tags[1]
            prompt_benchmark_decision = PromptTemplate.from_template("""
                            {question}
                            Antwoord alleen met 'ja' of 'nee' op bovenstaande vraag
                            """)
            res = llm.invoke(prompt_benchmark_decision.format(question=question))

            experiment_vars = question_loc.split("_")
            outputs.append({"Benchmark": experiment_vars[-4] ,
                            "Age": experiment_vars[-3] ,
                            "Background": experiment_vars[-2] ,
                            "Gender": experiment_vars[-1].strip('.txt') ,
                            "Decision": res.split('\n')[0].strip('.').strip()})

    output = pd.DataFrame(outputs)
    output.to_pickle(save_results)

# invoke_decision_problems(Ollama(model="mixtral"), save_results="./data/results/benchmark_1_mixtral.pickle")
invoke_decision_problems(llm=Ollama(model="llama3"), save_results="./data/results/benchmark_1_llama3.pickle")
# invoke_decision_problems(llm=Ollama(model='bramvanroy/geitje-7b-ultra:Q3_K_M'),
#                          save_results="./data/results/benchmark_1_geitje.pickle")
