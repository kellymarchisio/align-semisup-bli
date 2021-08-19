An Alignment-Based Approach to Semi-Supervised Bilingual Lexicon Induction with Small Parallel Corpora (2021)
===

The code in this directory implements:
- Kelly Marchisio, Conghao Xiong, and Philipp Koehn. 2021. **[An Alignment-Based Approach to Semi-Supervised Bilingual Lexicon Induction with Small Parallel Corpora](https://aclanthology.org/2021.mtsummit-research.24/)**. In *Proceedings of the 18th Biennial Machine Translation Summit (Volume 1: Research Track)*.

Quickstart
---------
To download required data and scripts, run: `sh setup.sh`
Running `sh run-all.sh` will run all main experiments from Table 3 of the published work.
---------
Note: This version of the code does not require a GPU. Parts of the embedding
mapping and evaluation can be sped up with a GPU by using the --cuda flag.

Publications
--------
Please cite our MT Summit paper if using this software for your research:
```
@inproceedings{marchisio-etal-2021-alignment,
    title = "An Alignment-Based Approach to Semi-Supervised Bilingual Lexicon Induction with Small Parallel Corpora",
    author = "Marchisio, Kelly and Koehn, Philipp and Xiong, Conghao",
    booktitle = "Proceedings of the 18th Biennial Machine Translation Summit (Volume 1: Research Track)",
    month = aug,
    year = "2021",
    address = "Virtual",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://aclanthology.org/2021.mtsummit-research.24",
    pages = "293--304",
}
```

Errata
-------
The published paper states, "For the IBM Model 2 step detailed in 5.1, we use
N=3000". Note that for the first mapping with VecMap, there was no size limit.
N=3000 was used for the second mapping only. The script run.sh reflects the
experiments from the paper accurately. Using N=3000 for the first mapping
weakens performance slightly on average. 
