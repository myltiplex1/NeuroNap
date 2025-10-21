# 🧠 NeuroNap

**NeuroNap** is an EEG-based sleep analysis tool that processes EEG CSV files to analyze sleep stages, generate hypnograms, visualize frequency spectra, and provide LLM-driven insights using **Gemini**.  
Built with **Python** and **Gradio**, it uses YASA for sleep stage classification and HMM for clustering and offers an interactive interface for researchers and clinicians studying sleep patterns.

---

## 🚀 Features

- **EEG Processing:**  
  Processes raw EEG CSV files using **bandpass** and **notch filters** to remove noise and artifacts, ensuring clean, reliable signal data.

- **Sleep Stage Classification:**  
  Utilizes **YASA** (Yet Another Sleep Analysis) to automatically classify sleep stages — **W, N1, N2, N3, and REM** — and compute essential sleep metrics such as **Total Sleep Time (TST)**, **Sleep Efficiency (SE)**, and **Wake After Sleep Onset (WASO)**.

- **Clustering:**  
  Applies **Hidden Markov Models (HMM)** to cluster EEG-derived features, revealing latent patterns and transitions across sleep stages.

- **Visualizations:**  
  Generates detailed plots for **EEG signals**, **hypnograms**, **frequency spectra**, and **band-specific waveforms**, allowing for intuitive data interpretation.

- **Sleep Statistics:**  
  Displays comprehensive sleep metrics (e.g., **Total Sleep Time**, **Sleep Efficiency**) with expandable tables showing additional parameters like **Time in Bed (TIB)**, **Sleep Period Time (SPT)**, and **latencies**.

- **LLM Insights:**  
  Integrates **Gemini** via **LangChain** for natural language interaction, using a **FAISS vector store** for context-aware sleep data insights.

- **Exportable Outputs:**  
  Enables easy download of results in **CSV** format (clean EEG, extracted features, clusters) and **PDF** format (visual reports and plots).
 

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/NeuroNap.git
cd NeuroNap

# Create a virtual environment
python -m venv venv

# Activate the environment
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

🔑 **API Setup**  
Set your Gemini API key in a `.env` file:  
```bash
echo "GEMINI_API_KEY=your_api_key" > .env
```

🧩 **Knowledge Embeddings**

Before running the app, generate embeddings from your knowledge PDFs:

```bash
python rag.py
```
Place relevant research papers or notes in the knowledge/ folder before running the above command.

🖥️ **Usage**

Launch the Gradio interface:
```bash
python main.py
```
Then:

1. **Upload** an EEG CSV file (e.g., `EEGDATA.csv`).  
2. **View** sleep stage visualizations, HMM clusters, and frequency spectra.  
3. **Interact** with the **"Chat with NeuroNap"** feature to get context-aware insights.  
4. **Download** processed data and reports in **CSV** or **PDF** format.  

## 📦 Requirements

- **Python 3.9+**  
- See `requirements.txt` for the full dependency list  

