---
title: Unsupervised Topic Discovery on 20 Newsgroups
description: Cleans raw Usenet posts, compares count and TF-IDF embeddings through PCA, picks k with silhouette and WCSS, and scores 4-cluster k-Means against the known labels using Hungarian matching. Result: 68.5% overall accuracy and 0.703 macro F1, stable across random seeds.
coverHtml: /projects/newsgroups-kmeans/cover.html
keywords: [Python, scikit-learn, k-Means, PCA, TF-IDF, Clustering, NLP, Regex]
minorTags: ["EE 400", "Natural Language Processing", "20 Newsgroups", "Bag-of-Words", "Silhouette Score", "WCSS", "Elbow Method", "Hungarian Algorithm", "Macro F1", "Precision/Recall", "Hyperparameter Sweep", "spaCy", "Lemmatization", "Yellowbrick", "Data Cleaning"]
order: 7
lastUpdated: 2026-11-09
previews:
  - { label: Notebook (rendered), href: /projects/newsgroups-kmeans/notebook.html }
  - { label: Notebook (.ipynb), href: /projects/newsgroups-kmeans/newsgroups-kmeans.ipynb }
---

If you dump a few thousand forum posts on a table with no labels, can an algorithm sort them back into the conversations they came from? This project tests that with four topics from the classic 20 Newsgroups benchmark: autos, baseball, gun politics, and medicine. k-Means never sees the labels. They are only used at the end to grade how well the clusters line up with reality.

**Cleaning Usenet.** Raw newsgroup posts are mostly packaging. Each one carries email headers, quoted replies from earlier in the thread ("In article ... writes:" followed by lines starting with `>`), signature blocks, PGP keys, encoded attachments, and ASCII separator lines. A sequence of regular expressions strips each of those, then removes email addresses, phone numbers, duplicates, blank and near-empty posts, and length outliers. That takes 2,330 posts down to 2,178 usable documents.

**Preprocessing.** Lowercasing, contraction expansion, punctuation and number removal, stopwords, and spaCy lemmatization, so "running" and "ran" land in the same bin.

**Turning words into vectors.** Each post becomes a row in a document-term matrix, scored two ways. Plain counts reward words that appear often. TF-IDF downweights words that appear everywhere and boosts words that are distinctive to a few posts. Both are compressed to 10 principal components with PCA. Even so, those 10 components hold only about 2.5–6% of the total variance, depending on the embedding: text is extremely high-dimensional, and most of its variation is spread thinly across thousands of rare words.

**Tuning instead of guessing.** Rather than picking settings by feel, a sweep ran 48 configurations: both embeddings, six `min_df` values, and four `max_df` values, each clustered and scored. TF-IDF with `min_df = 20` came out on top. Rerunning it with ten different random seeds moved accuracy by at most 0.001, so the result isn't a lucky initialization.

**How many topics does the data think it has?** The silhouette score peaks at k = 3, not 4, and the WCSS curve falls smoothly with no clear elbow. In other words, the geometry of the data honestly suggests three groups, because two of the four topics overlap heavily in this space.

**Scoring the clusters.** With k = 4, each cluster is matched to exactly one topic using the Hungarian algorithm, the assignment that maximizes total agreement. Three clusters come out very clean: the baseball, gun politics, and medicine clusters are 92–100% pure. The fourth acts as a catch-all. It collects almost every autos post along with the leftovers from the other three topics, so autos gets high recall (0.94) but low precision (0.45). Overall accuracy is 68.5% with a macro F1 of 0.703.

The takeaway: k-Means does well when a topic has its own vocabulary (box scores, firearms law, symptoms) and struggles when a topic's words are general enough to blend into everything else.

*Dataset not included: it was provided for course use only and contains sensitive responses. Notebook outputs that showed raw data have been removed.*

Built on the EE 400 starter notebook.

**Stack:** Python, pandas, scikit-learn, SciPy, spaCy, NLTK, Yellowbrick, Matplotlib
