---
title: Obesity and Heart Disease Across US States
description: "INFO 201 final project, redone as an interactive explorer. Joins CDC obesity survey data with county cardiovascular death rates for 51 states and DC (2011–2019). States with higher obesity have higher heart-disease death rates (r between 0.61 and 0.77 in every year), but the trend over time tells a more complicated story."
keywords: [R, tidyverse, ggplot2, Plotly, Quarto, Python, pandas, Data Visualization, dplyr]
minorTags: [Choropleth Maps, Correlation Analysis, CDC BRFSS, INFO 201, Public Health Data]
coverHtml: /projects/obesity-cvd/cover.html
order: 10
lastUpdated: 2026-11-19
previews:
  - { label: Interactive explorer (maps, scatter, 3D), href: /projects/obesity-cvd/explorer.html }
  - { label: Original R/Quarto report source, href: /projects/obesity-cvd/original-report.qmd }
---

Obesity and cardiovascular disease are two of the biggest public-health problems in the US, and they are often discussed as if one simply causes the other. This was the final project for INFO 201 at UW: do states with more obesity actually have more heart-disease deaths, and has that changed over time? We answered it in R and Quarto with two CDC datasets, then I rebuilt the analysis as the interactive explorer linked above, where you can drag a year slider, switch age groups, and hover any state.

**The data.** Obesity comes from the CDC's Behavioral Risk Factor Surveillance System (adult obesity prevalence by state and year). Cardiovascular deaths come from CDC/NVSS county-level, age-standardised death rates. Both had to be reshaped before they could be joined: filtering the survey to the one obesity question, collapsing hundreds of thousands of county rows to a state-year average, fixing a text `Year` column so the join would work, and keeping only the years the two sources share (2011–2019, 51 states and DC).

**What we found.** In every year, states with higher obesity also had higher cardiovascular death rates. The correlation across states ranges from r = 0.61 to 0.71 for ages 35–64 and 0.69 to 0.77 for ages 65+, and the South stands out on both maps (Mississippi, Alabama and Arkansas are highest). The three views show the same result three ways: paired maps, a scatterplot with a trend line, and the original 3D plot of year, death rate and obesity.

**A flaw I fixed when rebuilding it.** The original report averaged everything in the cardiovascular file together, including rates for two age groups, predicted rates, rate ratios and percent changes, and then claimed deaths and obesity were both rising. Separating them showed a more careful picture. For ages 35–64 the death rate rose about 7% over the decade, alongside obesity (27.6% to 32.0%). For ages 65+ it fell about 7%, even as obesity rose. So obesity is clearly associated with where heart-disease deaths are highest, but it can't explain the trend by itself. The comparison is also between states rather than people, so it shows an association, not that obesity causes the deaths.

**My role.** The original report was a group project, and I did the majority of the work on it. For this page I redid the analysis and built the interactive explorer on my own.

**Stack:** R, Quarto, tidyverse, ggplot2, usmap, plotly (original report); Python, pandas, Plotly.js (interactive explorer).
