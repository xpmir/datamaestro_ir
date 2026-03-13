[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) [![PyPI version](https://badge.fury.io/py/datamaestro-ir.svg)](https://badge.fury.io/py/datamaestro-ir)

# Information Retrieval Datasets

This [datamaestro](https://github.com/bpiwowar/datasets) plugin provides easy and systematic access to information retrieval datasets. It handles automated downloading and preparation of standard IR collections, exposes them through a typed Python API, and includes efficient document stores for fast text access (file, mmap, or in-memory).

Full documentation: [datamaestro-ir.readthedocs.io](https://datamaestro-ir.readthedocs.io/)

## Available Datasets

### Ad-hoc Retrieval

- **TREC Ad-hoc (1–8), Robust 2004/2005** — classic TREC test collections over TIPSTER/AQUAINT corpora
- **BEIR Benchmark** — 15+ datasets: TrecCovid, NQ, ArguAna, Touché, ClimateFever, SciDocs, NFCorpus, HotpotQA, FiQA, Quora, DBpedia-Entity, FEVER, SciFact, CQADupStack (12 sub-forums)
- **LoTTE** — domain-specific retrieval across 6 domains (lifestyle, recreation, science, technology, writing, pooled) × dev/test × search/forum queries
- **MS MARCO Passage & Document** — passage ranking (8.8M passages) and document ranking (v1: 3.2M, v2: 12M documents)
- **CORD-19 / TREC-COVID** — COVID-19 research article retrieval (192K documents)

### Conversational Search

- **TREC CaST 2019–2022** — conversational passage retrieval with decontextualized queries, tree-structured conversations (2022), and segmented passages
- **iKAT 2023–2025** — interactive knowledge-seeking over ClueWeb22

### Query Rewriting

- **CANARD** — context-aware query rewriting (train/dev/test)
- **QReCC** — question rewriting in conversational context (14K conversations, 81K QA pairs)
- **OrConvQA** — open-retrieval conversational QA over 11M Wikipedia passages

### Knowledge Distillation & Training Data

- **MS MARCO Ensemble/BERT Teacher** — 40M triples with teacher scores
- **rank-distillm** — BM25/ColBERTv2/RankZephyr annotated passages
- **MS MARCO Hard Negatives** — hard negatives mined from multiple retrieval models
- **Neural Ranking KD** — knowledge distillation teacher scores

### Base Document Collections

- **TIPSTER** (AP, FT, WSJ, ZIFF, …), **AQUAINT**, **TREC CAR** (29.8M paragraphs), **WAPO** v2/v4, **KILT** (42M Wikipedia articles)

