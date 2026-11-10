---
title: BERTopic with Framework-Seeded Topics
description: "Embedding-based topic modeling of 1,630 student responses about TA support with BERTopic (MiniLM sentence embeddings, UMAP, HDBSCAN, c-TF-IDF), then seeded with Chickering & Gamson's Seven Principles so the topics map onto an established teaching framework."
coverHtml: /projects/ta-feedback-bertopic/cover.html
keywords: [Python, BERTopic, Sentence Transformers, HDBSCAN, UMAP, Topic Modeling, NLP, Plotly]
minorTags: ["EE 400", "Natural Language Processing", "all-MiniLM-L6-v2", "Embeddings", "c-TF-IDF", "Seeded Topic Modeling", "Guided Topic Modeling", "Intertopic Distance Map", "Hierarchical Clustering", "Outlier Analysis", "Chickering & Gamson", "Community of Inquiry", "Hugging Face"]
order: 6
lastUpdated: 2026-11-09
previews:
  - { label: Notebook (rendered), href: /projects/ta-feedback-bertopic/notebook.html }
  - { label: Notebook (.ipynb), href: /projects/ta-feedback-bertopic/ta-feedback-bertopic.ipynb }
---

Classical topic models like LDA only count words. To them, "the TA never answers emails" and "replies take forever" have almost nothing in common. BERTopic starts from meaning instead: it turns every response into a sentence embedding first, so responses that say the same thing in different words end up next to each other. This project uses it on the same TA-feedback responses as my [LDA/LSA/NMF comparison](/projects/ta-feedback-lda-lsa-nmf), then steers it with an education framework.

**The pipeline.**
1. **Embed.** Each response becomes a 384-dimensional vector from the `all-MiniLM-L6-v2` sentence transformer.
2. **Reduce.** UMAP squeezes those vectors down to a few dimensions while keeping neighbors close together.
3. **Cluster.** HDBSCAN groups dense regions into topics with a minimum cluster size of 10. Unlike k-Means, it doesn't need k up front, and it is allowed to say "this point doesn't belong anywhere" (an outlier).
4. **Label.** c-TF-IDF finds the words that set each cluster apart from the rest, after English stopwords are removed.

Preprocessing is deliberately light: lowercasing, expanding contractions, and removing punctuation. The transformer reads context, so stripping stopwords or lemmatizing beforehand would take away the very information it uses.

**What it found.** The unseeded model produced 33 fine-grained topics plus 500 outliers, about 31% of the responses. On the intertopic distance map and the hierarchy, those topics collapse into four or five big groups, led by office hours and email, labs and quiz sections, feedback and grading speed, and worked examples and practice.

**Why so many outliers?** Reading through them shows two patterns. Some are too short or general to share vocabulary with any cluster. Others contain two separate requests, and because BERTopic gives each response exactly one topic, a response pulled toward two themes ends up stranded between them.

**Seeding with a framework.** Chickering & Gamson's *Seven Principles for Good Practice in Undergraduate Education* (student–faculty contact, cooperation, active learning, prompt feedback, time on task, high expectations, and respect for diverse ways of learning) was turned into seven seed-word lists. Seeding nudges the model toward those themes without forcing them. The seeded run produced 29 topics, and outliers dropped from 500 to 463. Practice problems and worked examples separated cleanly from general explanation requests, and exam review merged with quiz sections where it belongs.

**Where the framework fits and where it doesn't.** Contact, prompt feedback, active learning, and diverse ways of learning all show up clearly. Cooperation among students and high expectations barely appear at all. Students asking TAs for help don't talk about either. Meanwhile, logistics topics like Canvas and scheduling fit no principle, but they do fit the "design and organization" part of teaching presence in the Community of Inquiry framework, which works as a useful second lens.

**A note on the numbers.** BERTopic's UMAP step wasn't given a fixed random seed, so every rerun shifts the topic count and outlier total a little. The figures above come from the run saved in the published notebook.

*Dataset not included: it was provided for course use only and contains sensitive responses. Notebook outputs that showed raw data have been removed.*

Built on the EE 400 starter notebook.

**Stack:** Python, pandas, BERTopic, sentence-transformers, HDBSCAN, UMAP, scikit-learn, Plotly
