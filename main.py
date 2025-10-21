import gradio as gr
import pandas as pd
from processing import process_eeg
from visuals import plot_eeg_signals, plot_hypnogram, plot_frequency_spectra, plot_band_waveforms
from rag import setup_rag, chat_with_llm, update_vectorstore_with_user_results
import os
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neuronap.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def analyze_eeg(file):
    logger.info("Starting EEG analysis")
    if file is None:
        logger.error("No CSV file uploaded")
        raise ValueError("No CSV file uploaded.")
    file_path = file.name
    logger.info(f"Processing file: {file_path}")
    eeg_uv, eeg_uv_bp, eeg_uv_notched, hypno, stats, features_df, hmm_labels, fs, time_s = process_eeg(file_path)
    logger.info("EEG processing completed")
    signals_img = plot_eeg_signals(time_s, eeg_uv, eeg_uv_bp, eeg_uv_notched)
    hypno_img = plot_hypnogram(hypno)
    spectra_img = plot_frequency_spectra(eeg_uv_notched, fs)
    bands_img = plot_band_waveforms(eeg_uv_notched, fs)
    
    # Define key metrics with human-readable names for interface preview
    key_metrics = {
        'TST': 'Total Sleep Time (minutes)',
        'SE': 'Sleep Efficiency (%)',
        'REM': 'REM Sleep (minutes)',
        'N3': 'Deep Sleep (N3) (minutes)',
        'WASO': 'Wake After Sleep Onset (minutes)'
    }
    
    # Format stats preview for interface default view
    stats_preview = {k: stats.get(k) for k in key_metrics.keys() if k in stats}
    stats_preview_str = "\n".join([
        f"• {key_metrics[k]}: {v:.2f} {'%' if k in ['SE', '%N1', '%N2', '%N3', '%REM', '%NREM', 'SME'] else 'minutes'}"
        for k, v in stats_preview.items()
    ])
    
    # Format full stats for user_results.txt
    stats_full_str = "\n".join([
        f"• {k.replace('_', ' ').title()}: {v:.2f} {'%' if k in ['SE', '%N1', '%N2', '%N3', '%REM', '%NREM', 'SME'] else 'minutes'}"
        for k, v in stats.items()
    ])
    
    # Full stats table for interface expanded view
    stats_full = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value']).to_html(classes='stats-table', index=False)
    
    # Include full stats in user_results for saving to file
    user_results = f"Sleep Statistics:\n{stats_full_str}\nHypnogram: {hypno}\n"
    logger.info("Updating vectorstore with user results")
    vectorstore = update_vectorstore_with_user_results(user_results)
    qa_chain = setup_rag()
    
    # Generate previews and full tables for features and clusters
    features_preview = features_df.head(5).to_html(classes='compact-table')
    features_full = features_df.to_html(classes='compact-table')
    clusters_preview = pd.DataFrame({'Cluster': hmm_labels}).head(5).to_html(classes='compact-table')
    clusters_full = pd.DataFrame({'Cluster': hmm_labels}).to_html(classes='compact-table')
    
    logger.info("Analysis completed, returning outputs")
    return (
        signals_img, hypno_img, spectra_img, bands_img, stats_preview_str, stats_full,
        features_preview, features_full, clusters_preview, clusters_full,
        qa_chain,
        "clean_eeg.csv", "features.csv", "eeg_clusters.csv",
        "eeg_signal_plot.pdf", "hypnogram_plot.pdf", "eeg_fft_welch.pdf", "eeg_bands_plot.pdf"
    )

def handle_chat(query, qa_chain):
    logger.info(f"Handling chat query: {query}")
    if not query or qa_chain is None:
        logger.warning("Invalid query or no QA chain available")
        return "Please analyze an EEG file first and provide a query."
    return chat_with_llm(query, qa_chain)

with gr.Blocks(title="NeuroNap", css=".compact-table { font-size: 12px; max-height: 300px; overflow-y: auto; } .stats-table { font-size: 14px; border-collapse: collapse; width: 50%; } .stats-table th, .stats-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }") as demo:
    logger.info("Initializing Gradio interface")
    qa_chain_state = gr.State(None)
    gr.Markdown("# NeuroNap: EEG Sleep Monitoring System")
    with gr.Column():
        with gr.Row():
            file_input = gr.File(label="Upload EEG CSV (e.g., EEGDATA.CSV)")
            analyze_btn = gr.Button("Analyze")
        with gr.Row():
            signals_output = gr.Image(label="EEG Signals")
            hypno_output = gr.Image(label="Hypnogram")
        with gr.Row():
            spectra_output = gr.Image(label="Frequency Spectra")
            bands_output = gr.Image(label="Frequency Bands")
        with gr.Group():
            gr.Markdown("### Sleep Statistics")
            stats_preview_output = gr.Textbox(label="Key Sleep Statistics")
            with gr.Accordion("Show All Sleep Statistics", open=False):
                stats_full_output = gr.HTML(label="Full Statistics")
        with gr.Group():
            gr.Markdown("### Extracted Features")
            features_preview_output = gr.HTML(label="Features Preview")
            with gr.Accordion("Show Full Features Table", open=False):
                features_full_output = gr.HTML(label="Full Features")
        with gr.Group():
            gr.Markdown("### HMM Clusters")
            clusters_preview_output = gr.HTML(label="Clusters Preview")
            with gr.Accordion("Show Full Clusters Table", open=False):
                clusters_full_output = gr.HTML(label="Full Clusters")
        with gr.Group():
            gr.Markdown("### Chat with NeuroNap")
            chat_input = gr.Textbox(label="Ask about your sleep data", lines=2)
            chat_btn = gr.Button("Send")
            chat_output = gr.Textbox(label="LLM Response", lines=4)
        with gr.Group():
            gr.Markdown("### Downloads")
            with gr.Row():
                clean_csv = gr.File(label="Download clean_eeg.csv")
                features_csv = gr.File(label="Download features.csv")
                clusters_csv = gr.File(label="Download eeg_clusters.csv")
            with gr.Row():
                signals_pdf = gr.File(label="Download eeg_signal_plot.pdf")
                hypno_pdf = gr.File(label="Download hypnogram_plot.pdf")
                spectra_pdf = gr.File(label="Download eeg_fft_welch.pdf")
                bands_pdf = gr.File(label="Download eeg_bands_plot.pdf")
    
    analyze_btn.click(
        analyze_eeg,
        inputs=file_input,
        outputs=[
            signals_output, hypno_output, spectra_output, bands_output, 
            stats_preview_output, stats_full_output,
            features_preview_output, features_full_output, clusters_preview_output, clusters_full_output,
            qa_chain_state, clean_csv, features_csv, clusters_csv, signals_pdf, hypno_pdf, spectra_pdf, bands_pdf
        ]
    )
    chat_btn.click(handle_chat, inputs=[chat_input, qa_chain_state], outputs=chat_output)

logger.info("Launching Gradio interface")
port = int(os.environ.get("PORT", 7860))
demo.launch(share=True, server_name="0.0.0.0", server_port=port, pwa=True, favicon_path="assets/icon-192x192.png")
# demo.launch(share=True)