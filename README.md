# ML / DL / AI Hub

A collection of small, self-contained machine learning, deep learning and generative-AI projects. Each folder is an independent sub-project, mostly Jupyter notebooks written for Google Colab, with its own README describing the dataset, the model and how to run it. Topics range from classical scikit-learn classifiers and a CNN for deepfake-audio detection to LLM tooling with Google Gemini and LangChain.

## Projects

| Project | What it does | Techniques / models | Dataset | Status |
|---|---|---|---|---|
| [Fake Audio Detection](./AUDIO_DETECTION_SYSTEM-main/) | Classifies short voice clips as real or AI-generated from their mel-spectrograms | librosa mel-spectrograms (128x128) and a 3-block Conv2D CNN in TensorFlow/Keras; 96.5% validation accuracy, 0.93 precision, 1.00 recall after 10 epochs | 1,000 WAV clips, 500 real / 500 fake, in `dataset.rar` | Complete (Colab notebook) |
| [Dynamic Predictor](./Dynamic_Model/) | Interactive CLI that trains 15 regression / time-series models on any CSV or XLSX, ranks them by R² and forecasts daily or monthly values with the best one | scikit-learn (linear, Ridge, Lasso, ElasticNet, Bayesian Ridge, polynomial, KNN, decision tree, random forest, extra trees, gradient boosting), XGBoost, statsmodels ARIMA and ETS, Keras LSTM | User-supplied (none bundled) | Complete (Python script) |
| [Fake News Detection](./FAKE_NEWS_DETECTION_SYSTEM-main/) | Detects unreliable news articles from author + title text; also includes two warm-up notebooks (diabetes classification, housing EDA) | Porter stemming + TF-IDF with Logistic Regression, Decision Tree, Naive Bayes, Random Forest, Gradient Boosting (best: Random Forest, 99.4% test accuracy); Embedding + LSTM on full article text (92.1%) | Kaggle "Fake News" `train.csv` (20,800 articles); Pima Indians Diabetes; Boston Housing | Complete (Colab notebooks) |
| [Gemini through LangChain](./GEMINI_THROUGH_LANGCHAIN-main/) | Walkthrough of calling Google Gemini directly and via LangChain: chat, streaming, LCEL chains, PAL math reasoning, image understanding | `google-generativeai` SDK, `langchain-google-genai`, LCEL (`prompt \| model \| parser`), `PALChain`, `gemini-pro` / `gemini-pro-vision` | None (needs a Google AI Studio API key) | Complete; model names in the notebook have since been retired by Google |
| [Image & Text Generation / Manipulation](./IMAGE%26TEXT_GENERATION_MANIPULATION/) | Cookbook of generative tasks: text-to-image through an API, background removal, informal-to-formal rewriting of text and DOCX files, local paraphrasing | StarryAI REST API, `rembg` (ONNX U2-Net), Sapling rephrase API, T5 paraphraser `humarin/chatgpt_paraphraser_on_T5_base` via Hugging Face `transformers` | None (needs StarryAI and Sapling API keys) | Complete (Colab notebook) |
| [Music Recommendation System](./MUSIC_RECOMENDATION_SYSTEM-main/) | Interactive loop that returns the 10 songs most similar to the title you enter | Content-based filtering: CountVectorizer bag-of-words on artist + title, cosine similarity (scikit-learn) | `data/song-dataset.csv`, Spotify chart statistics, 11,084 songs (first 1,000 used) | Complete (Colab notebook) |

## How to run

All projects use Python 3 (3.9 to 3.11 recommended). The notebooks were written for Google Colab and mount Google Drive for their data. To run them locally, skip the `drive.mount(...)` cell and replace the `/content/drive/...` paths with the files that live in each project folder.

```bash
git clone https://github.com/Ramim-Siddiqui/ML-DL-AI-hub.git
cd ML-DL-AI-hub
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install jupyterlab
jupyter lab                     # open the notebook inside the project folder
```

No folder ships a `requirements.txt` yet, so install the libraries each notebook imports:

| Project | Install | Notes |
|---|---|---|
| Fake Audio Detection | `pip install tensorflow numpy librosa pydub scikit-image matplotlib` | `pydub` needs [ffmpeg](https://ffmpeg.org/) on the PATH. Extract `dataset.rar` (7-Zip or WinRAR) and point `fake_directory` / `real_directory` at the `fake` and `real` folders. |
| Dynamic Predictor | `pip install pandas numpy scikit-learn xgboost statsmodels tensorflow openpyxl` | Run `python dynamic_model.py` and follow the prompts. Output files `r2scores.csv`, `predictions.csv`, `monthly_predictions.csv` are written to the working directory. |
| Fake News Detection | `pip install pandas numpy scikit-learn nltk wordcloud seaborn matplotlib scipy statsmodels wquantiles "tensorflow<2.16"` | Run `python -c "import nltk; nltk.download('stopwords')"` once. The LSTM section uses the legacy `keras.preprocessing.text` API, which was removed in Keras 3 (TensorFlow 2.16+). |
| Gemini through LangChain | `pip install langchain langchain-core langchain-experimental langchain-google-genai google-generativeai requests` | Set `GOOGLE_API_KEY` as an environment variable instead of pasting it into the notebook. Replace `gemini-pro` / `gemini-pro-vision` with a currently supported model name. |
| Image & Text Generation | `pip install requests rembg onnxruntime pillow docx2txt python-docx transformers sentencepiece torch` | Set the StarryAI and Sapling API keys in the notebook. The T5 paraphraser (~900 MB) downloads on first use and runs on CPU or GPU. |
| Music Recommendation | `pip install pandas scikit-learn` | Change the `read_csv` path to `data/song-dataset.csv`. |

## Data files

The repository tracks its datasets directly in git. Two of them are large and account for roughly half of the ~280 MB clone size (about 140 MiB of packed history).

| File | Size | Origin | Notes |
|---|---|---|---|
| `FAKE_NEWS_DETECTION_SYSTEM-main/Fake_News_Detection_System/train.csv` | 98.8 MB (94.2 MiB) | Kaggle "Fake News" competition: https://www.kaggle.com/c/fake-news/data (columns `id, title, author, text, label`; 20,800 rows) | Above the GitHub 50 MiB warning threshold and just under the 100 MiB hard limit. |
| `AUDIO_DETECTION_SYSTEM-main/dataset.rar` | 42.0 MB (40.1 MiB) | Not recorded in the repo. Contains 1,000 WAV clips in `dataset/real` and `dataset/fake`, 50 per accent-group / gender pair (`african`, `american`, `easian`, `european`, `sasian` x `m` / `w`) | RAR5 archive; needs 7-Zip or WinRAR to extract. |
| `MUSIC_RECOMENDATION_SYSTEM-main/data/song-dataset.csv` | 0.66 MB | Spotify chart statistics (position, artist, title, days on chart, peak position, peak and total streams); 11,084 rows; source not recorded | |
| `FAKE_NEWS_DETECTION_SYSTEM-main/Data_Visualization/housing.csv` | 50 KB | Boston Housing dataset (UCI), 506 rows x 14 columns | Whitespace-delimited, no header. |
| `FAKE_NEWS_DETECTION_SYSTEM-main/Data_Preprocessing_and_Machine_Learning/pima-indians-diabetes.csv` | 24 KB | Pima Indians Diabetes dataset (UCI / Kaggle), 768 rows | No header row; column names are set in the notebook. |

The root `.gitignore` now blocks new archives, model weights and pickles (`*.rar`, `*.zip`, `*.h5`, `*.pkl`, ...) so large artefacts are not added again. Files already tracked are unaffected.

## Roadmap

- Move the two large datasets out of git: link to the Kaggle page for `train.csv`, and host `dataset.rar` on Git LFS or as a release asset. Rewriting history afterwards would shrink a fresh clone from ~280 MB to a few MB.
- Split each sub-project into its own repository and keep this hub as an index.
- Add a `requirements.txt` per project and replace Colab / Google Drive paths with relative paths so the notebooks run locally without edits.
- Update the Gemini notebook to current model names and the current LangChain import paths.
- Strip bulky notebook outputs (the two fake-news notebooks are 2.3 MB and 2.6 MB because of embedded plots) while keeping the metric cells.
