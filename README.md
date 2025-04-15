# HistoriChat
### This has not been converted into an app yet so the scripts can be run directly for experiments
`source venv/bin/activate` <br />
`pip install -r requirements.txt` <br />
`python3 VanillaRAG.py` runs Vanilla RAG on ther testset <br />
`python3 TemporalAwareRAG.py` runs Temporal RAG on the testset <br />
<br />

### The testset consists of 916 QnA pairs (916/1000 generations was successful)
`python3 dataGenerator.py` utilizes RAGAS framework to create a synthetic dataset for evaluation. <br />

### Main Ideas
![screenshot](https://github.com/akashkashyaps/HistoriChat/blob/main/data/comparison.png)
