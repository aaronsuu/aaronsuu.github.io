---
title: "Topic Modeling Student Feedback: LDA vs LSA vs NMF"
description: "Three classical topic models applied to 1,630 open-ended student responses about TA support. The number of topics is chosen with UMass coherence, singular-value spectra, and NMF reconstruction error, and a top-two probability margin separates confident topic assignments from ambiguous ones."
coverHtml: /projects/ta-feedback-lda-lsa-nmf/cover.html
keywords: [Python, LDA, LSA, NMF, Topic Modeling, scikit-learn, Gensim, NLP]
minorTags: ["EE 400", "Natural Language Processing", "UMass Coherence", "Perplexity", "SVD", "Singular Values", "Reconstruction Error", "TF-IDF", "Document-Term Matrix", "Assignment Margin", "Word Clouds", "Model Comparison", "Human-in-the-Loop", "Matplotlib"]
order: 5
lastUpdated: 2026-11-09
previews:
  - { label: Notebook (rendered), href: /projects/ta-feedback-lda-lsa-nmf/notebook.html }
  - { label: Notebook (.ipynb), href: /projects/ta-feedback-lda-lsa-nmf/ta-feedback-lda-lsa-nmf.ipynb }
---

Ask over a thousand students "what could your TAs do better?" and you get a pile of short answers no one has time to read one by one. Topic models compress that pile into a handful of recurring themes. This project runs three different ones on the same responses and compares what each sees.

**Three ways to find a topic.**
- **LDA (Latent Dirichlet Allocation)** is probabilistic. It assumes each response is a mix of topics and each topic is a mix of words, then works backwards to find the mixes that best explain the text. Its output is a probability for every topic in every response.
- **LSA (Latent Semantic Analysis)** is linear algebra. It runs a truncated SVD on the TF-IDF matrix and keeps the strongest directions of variation.
- **NMF (Non-negative Matrix Factorization)** also factors the matrix, but with no negative weights allowed. That tends to produce topics that read like additive parts ("office hours" + "email").

**Choosing how many topics.** There's no single right answer, so three signals were weighed alongside a read-through of a random sample of responses:
- UMass coherence, which checks how often a topic's top words actually show up together, swept from k = 2 to 10 for each model.
- The singular values from LSA, which drop steeply and level off after about 7 components.
- NMF reconstruction error, which falls in an almost straight line. That means no single k explains the data much better than its neighbors, because the responses are short and share so much vocabulary.

Coherence alone favors k = 2, which is too coarse to be useful. The sample read-through showed at least five distinct kinds of requests, so k = 6 was used for the final models.

**How confident is each assignment?** For every response, the gap between its top two LDA topic probabilities measures how cleanly it belongs to one theme. 76% of responses are "strong" (gap of 0.6 or more) and about 5% are "weak" (gap under 0.1). The weak ones are almost always responses with two ideas in them, typically a request followed by the reason for it. That points to a practical fix: split responses into sentences before modeling, so each idea gets its own topic.

**Cleaner word clouds.** A few words ("help", "question", "lab", and similar) sat near the top of most topics and drowned out the differences. Removing words shared by four or more topics made each LDA topic's cloud read as its own theme: flexible office hours, fast email replies, feedback on assignments, worked examples, and exam review.

**What the comparison shows.** All three models agree on the big themes: office hours, answering questions, lab help, and how TAs teach. They differ in granularity. LSA and NMF surface the dominant keywords. LDA, because it models which words occur together within a response, separates more specific requests like feedback and worked examples. Where the models disagree is informative too: it shows which themes students tend to mention together.

*Dataset not included: it was provided for course use only and contains sensitive responses. Notebook outputs that showed raw data have been removed.*

Built on the EE 400 starter notebook.

**Stack:** Python, pandas, scikit-learn, Gensim, spaCy, NLTK, WordCloud, Matplotlib
