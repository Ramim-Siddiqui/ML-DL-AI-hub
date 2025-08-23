# 🖼️ Image & Text Generation / Manipulation

This project demonstrates how to work with **AI models for image generation and text humanization/paraphrasing**, using both **local models** and **remote APIs**.  

---

## 🚀 Features

### 🎨 Image Generation
- Run **Stable Diffusion locally** with GPU acceleration (`diffusers`).  
- Generate AI images via the **StarryAI REST API**.  
- Poll StarryAI’s API until the image is ready, then fetch the image URL.  

### ✍️ Text Generation
- Use **Hugging Face Inference API** for paraphrasing and chat completions.  
- Run a **local Transformer model** (`humarin/chatgpt_paraphraser_on_T5_base`) for offline text humanization.  
- Supports **multiple paraphrase variations**.  

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
Install dependencies:

PyTorch (with CUDA if available).

Required Python libraries:

pip install torch diffusers transformers sentencepiece huggingface_hub requests

### 🔑 API Keys

Some parts of the notebook require API keys:

StarryAI API Key → obtained from the StarryAI platform.

Hugging Face Token → only needed for private/gated models.

### 📌 Usage

Open and run the Jupyter Notebook:

Stable Diffusion (Local)

Generate AI images on your machine using GPU acceleration.

StarryAI API

Send prompts to StarryAI, poll until the image is ready, and fetch the generated image URL.

Hugging Face API (Remote)

Use Hugging Face-hosted models for chat completions and paraphrasing.

Hugging Face Transformers (Local)

Run a local paraphraser model (humarin/chatgpt_paraphraser_on_T5_base) to humanize AI text without relying on APIs.

### 📂 Project Structure

Image & Text Generation_Manipulation.ipynb → Main Jupyter Notebook with examples.

README.md → Documentation.

### ✅ Notes

⚠️ StarryAI requests consume credits, so use them sparingly.

🔑 Hugging Face token is optional, unless using private or gated models.

💾 Local models are cached in your system for offline use.

👨‍💻 Author
Developed by Ramim Siddiqui
