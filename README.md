# 🧠 NeuroNap

**NeuroNap** is an EEG-based sleep analysis tool that processes EEG CSV files to analyze sleep stages, generate hypnograms, visualize frequency spectra, and provide LLM-driven insights using **Gemini**.  
Built with **Python** and **Gradio**, it offers an interactive interface for researchers and clinicians studying sleep patterns.

---

## 🚀 Features

- Upload EEG CSV files for automated processing  
- Visualize EEG signals, hypnograms, and frequency spectra  
- Display detailed sleep statistics (TST, SE, REM, etc.)  
- Interactive **LLM chat** for sleep data insights  
- Downloadable reports in **CSV** and **PDF** formats  

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
