---
title: Mental-Health Text Cleaning & Visualization
description: "A human-in-the-loop cleaning and preprocessing pipeline for 2,231 machine-translated reflections tagged anxiety, depression, stress, or normal. It covers encoding repair, de-identification with spaCy NER, a protected clinical vocabulary, and per-theme word clouds tested on human readers."
coverHtml: /projects/mental-health-text-cleaning/cover.html
keywords: [Python, NLP, spaCy, NLTK, pandas, Data Cleaning, Text Preprocessing, Word Clouds]
minorTags: ["EE 400", "Natural Language Processing", "ftfy", "LanguageTool", "Named Entity Recognition", "De-identification", "Lemmatization", "Stopwords", "POS Filtering", "Contractions", "Mojibake", "Human-in-the-Loop", "Matplotlib"]
order: 8
lastUpdated: 2026-11-09
previews:
  - { label: Notebook (rendered), href: /projects/mental-health-text-cleaning/notebook.html }
  - { label: Notebook (.ipynb), href: /projects/mental-health-text-cleaning/mental-health-text-cleaning.ipynb }
---

Before a model can learn anything from text, someone has to decide what counts as noise. This project is that step for a set of short personal reflections on mental health, each tagged with one of four affective states: anxiety, depression, stress, or normal. The posts were machine-translated from social media, so they arrive with broken characters, repeated posts, empty rows, and real names mixed in with the feelings.

One rule guided every decision: fix what is broken, but don't clean away the signal.

**Repairing the text.** About 400 rows had mangled characters from a bad encoding round-trip, where a single curly apostrophe turns into three random symbols. `ftfy` reverses those in place, and URLs and HTML tags are stripped next.

**Removing what isn't a response.** 21 duplicate posts and 5 non-answers (blank rows, "idk", filler) are dropped, taking the set from 2,231 rows to 2,205. Unusually long posts, more than three standard deviations above the mean word count, are flagged for a human to look at but kept. A long post about depression is still a real post.

**De-identifying without erasing.** spaCy's named-entity recognizer finds people and places and replaces them with placeholder tokens, which are later dropped as stopwords. Off-the-shelf NER and spell-checkers don't know medication names, though, so a word like "lexapro" can get "corrected" into an ordinary English word. A small protected vocabulary of medication and condition terms is excluded from both steps so the subject matter survives. Grammar is left as written on purpose: awkward, translated phrasing is part of how people describe what they are going through.

**Preprocessing.** Lowercasing, contraction expansion, removal of punctuation, special characters, and numbers, then spaCy lemmatization. Standard stopwords are removed along with a few custom ones like "want" and "get" that appear everywhere and carry no theme. A part-of-speech filter keeps only nouns, verbs, adjectives, and adverbs. Every step writes to its own column, so each transformation can be checked side by side against the original.

**Does cleaning change the picture?** Three versions of the pipeline (no cleaning, full cleaning, and only the steps that mattered most) were compared through word clouds. Then each theme got its own word cloud and top-20 bar chart. The themes are uneven: stress makes up almost half the responses, and depression posts average around 177 words against about 11 for normal, so raw frequency alone would let the long posts dominate.

**A small human test.** A few readers were shown the four theme clouds without labels and asked to match them. Everyone placed normal and depression correctly. "Morning", "dream", and "good" give normal away, while "life" and "depression" anchor the other. The only mistakes were stress and anxiety getting swapped, which makes sense: the two share most of their vocabulary, and the word "anxiety" shows up in both.

*Dataset not included: it was provided for course use only and contains sensitive responses. Notebook outputs that showed raw data have been removed.*

Built on the EE 400 starter notebook.

**Stack:** Python, pandas, spaCy, NLTK, ftfy, LanguageTool, WordCloud, Matplotlib
