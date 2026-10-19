# Music Recommendation System

This repository contains code for a music recommendation system that recommends 10 songs similar to the one entered by the user.

## Overview

The system is a content-based recommender. It loads the first 1,000 rows of the song dataset, keeps the chart position, artist name and song title, joins them into one text field per song and vectorizes it with scikit-learn's `CountVectorizer`. Cosine similarity between these bag-of-words vectors gives a song-to-song similarity matrix; for a title entered by the user the 10 most similar songs are printed. No user listening history is required.

## Dataset

`data/song-dataset.csv` contains Spotify chart statistics for 11,084 songs (position, artist name, song name, days on chart, times in the top 10, peak position, peak streams, total streams). Any CSV with `Artist Name` and `Song Name` columns can be substituted.

## Requirements

To run the code in this repository, you need to have the following dependencies installed:

- Python 3.x
- Pandas
- Scikit-learn

Install them with pip:

```bash
pip install pandas scikit-learn
```

The notebook was written for Google Colab; change the `read_csv` path to `data/song-dataset.csv` to run it locally, then run the cells top to bottom and enter a song title at the prompt.
