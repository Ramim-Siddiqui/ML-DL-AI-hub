# Image & Text Generation / Manipulation

A Colab notebook (`Image & Text Generation_Manipulation.ipynb`) collecting small, independent recipes for generating and manipulating images and text with both hosted APIs and locally-run models.

## What the notebook covers

### Image generation (StarryAI API)
- Submits a text prompt to the StarryAI REST endpoint (`POST /creations/`, model `lyra`).
- Polls the creation until it is ready, then prints the generated image URL.

### Background removal (local)
- Uses [`rembg`](https://github.com/danielgatis/rembg) (a U2-Net model run through `onnxruntime`) to strip the background from an image file and save a transparent PNG.

### Text rephrasing (Sapling API)
- Calls the Sapling `rephrase` endpoint with the `informal_to_formal` mapping on a single sentence.
- Batch version: reads a `.docx` with `docx2txt`, rephrases it paragraph by paragraph and writes a new document with `python-docx`.

### AI text humanizer (local)
- Loads `humarin/chatgpt_paraphraser_on_T5_base` with Hugging Face `transformers` and generates several paraphrases per input with beam search (`num_beams=5`, `no_repeat_ngram_size=2`). Runs on GPU if available, otherwise CPU.

## Installation

```bash
pip install requests rembg onnxruntime pillow docx2txt python-docx transformers sentencepiece torch
```

For GPU paraphrasing install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/.

## API keys

- **StarryAI** - from https://starryai.com (requests consume credits).
- **Sapling** - from https://sapling.ai.

Both are read from variables at the top of the respective cells; set them there or load them from environment variables.

## Usage

Open the notebook in Jupyter or Colab and run the section you need; the sections are independent of each other. The background-removal cell mounts Google Drive for its input and output paths; change `input_path` / `output_path` to local files when running outside Colab.

## Project structure

- `Image & Text Generation_Manipulation.ipynb` - the notebook with all recipes.
- `README.md` - this file.

## Author

Developed by Ramim Siddiqui
