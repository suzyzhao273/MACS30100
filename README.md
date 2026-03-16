# Machine Learning Final Project

This repository contains the **code used for the final machine learning project**.

The datasets used in this project are **not included** because they are very large and uploading them would be impractically slow. Therefore, this repository only contains the **code pipeline** used to construct the dataset and run the models.

---

## Main Pipeline

### `final_pipeline.Rmd`

This file contains the **complete machine learning workflow**, including:

- data processing
- feature engineering
- model training
- hyperparameter tuning
- model comparison
- model interpretation

---

## Data Construction Scripts

### `base_table.ipynb`

Creates the base reference table of songs with identifiers and release information, including:

- `song_id`
- `song_name`
- `artist_id`
- `artist_name`
- `album_id`
- `album_name`
- `release_date`

It also defines the time windows used later for collecting Reddit discussion data.

---

### `reddit_music_artist_only_*.ipynb`

Collects discussion data from the **r/music** subreddit to construct variables such as:

- discussion **volume**
- **controversy** indicators based on Reddit activity.

---

### `merge_popularity_features.ipynb`

Merges song popularity, audio features, and other music-related metadata into the main dataset.

---

### `add_new_control_variables.ipynb`

Adds additional control variables such as artist followers and artist type.

---

### `check_data.ipynb`

Performs final data checks and minor cleaning before the dataset is used in the machine learning pipeline.
