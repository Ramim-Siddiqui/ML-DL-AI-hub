# Google Gemini through LangChain

A Colab notebook (`Gemini.ipynb`) that walks through using Google's Gemini large language models from Python: first with the official `google-generativeai` SDK, then through LangChain's `langchain-google-genai` integration.

## What the notebook covers

1. **Setup** - installs `langchain`, `langchain-core`, `langchain-experimental`, `langchain-google-genai` and `google-generativeai`, configures `GOOGLE_API_KEY` and lists the models available to the key.
2. **Direct SDK call** - `genai.GenerativeModel('gemini-pro').generate_content(prompt)` rendered as Markdown.
3. **LangChain chat model** - `ChatGoogleGenerativeAI` with `invoke(...)` for a single answer and `stream(...)` for token streaming.
4. **LCEL chain** - `ChatPromptTemplate | model | StrOutputParser` to run a templated prompt (`"Tell me short facts about {topic}"`).
5. **PAL chain** - `PALChain.from_math_prompt` from `langchain_experimental` solves arithmetic word problems by having the model write and execute Python.
6. **Multimodal** - `gemini-pro-vision` describes an image fetched from a URL, passed as a `HumanMessage` with a text part and an `image_url` part.

## Requirements

- Python 3.9+
- A Google AI Studio API key: https://aistudio.google.com/app/apikey

```bash
pip install langchain langchain-core langchain-experimental langchain-google-genai google-generativeai requests
```

Set the key as an environment variable rather than pasting it into the notebook:

```bash
set GOOGLE_API_KEY=...        # Windows (PowerShell: $env:GOOGLE_API_KEY="...")
export GOOGLE_API_KEY=...     # Linux / macOS
```

Then open `Gemini.ipynb` in Jupyter or Colab and run the cells top to bottom.

## Notes

- The model names used in the notebook (`gemini-pro`, `gemini-pro-vision`) have since been retired by Google. Substitute a currently supported model from https://ai.google.dev/gemini-api/docs/models in the `GenerativeModel(...)` and `ChatGoogleGenerativeAI(model=...)` calls.
- The first cell pins `google-generativeai==0.3.1` and `google-ai-generativelanguage==0.4.0` (early-2024 versions). With current LangChain releases the imports `langchain.prompts` and `langchain.schema.output_parser` should be replaced by `langchain_core.prompts` and `langchain_core.output_parsers`.
