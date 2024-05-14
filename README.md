# Building LLM Benchmarks with Langchain and Ollama
This is the repository for the demo at the PyGrunn conference in 2024.

You need to have ollama installed locally, to be able to run LLMs on mac do the following:
`brew install ollama`.

Otherwise, visit https://Ollama.com

Make sure that Ollama is also running on your PC, before starting any of the scripts, this needs to happen by 
`ollama serve` in the command line or starting the application itself. 

Just using the ollama package from python is not enough, that is just an interface to the ollama running locally.

# Data used
For filling in the templates data on the Dutch population is used from the
[Dutch Census](https://www.cbs.nl/nl-nl/achtergrond/2016/47/bevolking-naar-migratieachtergrond) published by the CBS.

On the most popular surnames in Dutch, Frisian, Marroco, Turkey, and Suriname these websites have been traversed:
- [familyeducation](https://www.familyeducation.com/baby-names/surname/origin/dutch)
- [globalsuernames](https://globalsurnames.com/nl)
