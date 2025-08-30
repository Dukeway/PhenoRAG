

<div align="right">

**English** | [**简体中文**](README.zh-CN.md)

</div>

# PhenoRAG 

**Interactive-RAG-HPO: A User-Friendly, Multilingual Web Application for Real-Time Phenotype Annotation.**

PhenoRAG is an advanced, user-friendly web application designed to bridge the gap between complex clinical narratives and standardized phenotypic data. Leveraging the power of Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs), this tool intelligently extracts clinical phenotypes from patient descriptions in any language and accurately maps them to the Human Phenotype Ontology (HPO).

I have deployed this project to Streamlit Cloud to facilitate usage by non-technical personnel.

Inspired by the foundational research of [Garcia et al. in *Genome Medicine*](https://doi.org/10.1186/s13073-025-01521-w), this project transforms the original command-line tool into an interactive, clinic-ready platform, making deep phenotyping accessible to clinicians, genetic counselors, and researchers without requiring any programming knowledge.

**It should be noted that** using RAG to reduce the hallucination rate of LLMs in HPO identification was a small part of the content envisioned in my medical doctoral dissertation. I am pleased to see that Garcia et al. were the first to implement this approach and published their research findings in a high-quality journal, conducting substantial additional work that I had not previously anticipated. Although Garcia et al. also open-sourced a similar application, their approach differs from my envisioned method. They augmented their knowledge base with extensive synonyms to improve model retrieval accuracy and primarily utilized the smaller open-source model Llama3.1-70B (believing that such  models would have advantages for local hospital deployment). However, I believe that extensive data annotation runs counter to the zero-shot reasoning capabilities of large language models. Moreover, HPO identification primarily occurs in academic communication or online diagnosis of rare diseases. Therefore, patients' phenotypic descriptions are typically not privacy or security concerns, allowing us to utilize larger parameter models like ChatGPT to reduce data annotation requirements.

More importantly, Garcia et al.'s application lacks visualization capabilities, which is highly inconvenient for many physicians who essentially lack programming skills. Given the complexity of their work, I did not study their code implementation, but inevitably drew inspiration from their published article. I would like to express my gratitude here.

![描述](./images/screenshot01.png)

![描述2](./images/screenshot02.png)

---

## ✨ Key Features

- **🌐 Multilingual Support**: Automatically translates clinical text from any language (e.g., Chinese, Spanish, French) into English before analysis, breaking down language barriers.
- **🧠 Advanced RAG-HPO Core**: Implements a robust RAG pipeline to minimize LLM hallucinations and achieve high accuracy in HPO term assignment.
- **🖥️ Interactive & User-Friendly UI**: A clean web interface built with Streamlit that provides real-time feedback, progress updates, and clear, verifiable results.
- **✅ Result Validation**: Automatically verifies each LLM-assigned HPO term against the official HPO database, providing an instant quality check (✅ Correct, ⚠️ Name Mismatch, ❌ Invalid ID).
- **🔄 Automated & Up-to-Date Knowledge Base**: The knowledge base is built automatically from the latest official HPO `hp.json` file, ensuring the tool always uses the most current phenotype data.
- **💾 Data Export**: Allows users to download analysis results as a CSV file for record-keeping, statistical analysis, or integration into other systems.
- **🔧 Flexible Backend**: Easily configurable to work with any OpenAI-compatible LLM API (e.g., Groq, Together AI, OpenAI).

## 🚀 Getting Started

Follow these steps to set up and run the application locally.

### 1. Prerequisites

- Python 3.10+
- Git

### 2. Installation

Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/dukeway/PhenoRAG.git
cd PhenoRAG

# Install dependencies
pip install -r requirements.txt
```

### 3. Data and Model Setup

- **HPO Data**: Download the latest hp.json file from the HPO Website([Human Phenotype Ontology](https://hpo.jax.org/data/ontology) )and **place it in the data/ directory.**
- **Embedding Model**: The application uses the bge-small-en-v1.5 sentence transformer model. It should be automatically downloaded on first run. Alternatively, you can pre-download it and place it in a local_models/ directory.

### 4. Configuration

Create a .env file in the root directory of the project and add your LLM API credentials:

```
# .env file
API_BASE_URL="https://api.siliconflow.cn/v1"
API_KEY="YOUR_API_KEY_HERE"
LLM_MODEL="deepseek-ai/DeepSeek-V3"
```

### 5. Running the Application

Launch the Streamlit app with the following command:

codeBashdownloadcontent_copyexpand_less

```
streamlit run app.py
```

Your web browser should automatically open a new tab with the application running.

## 📖 How to Use

1.  **Configure API**: Ensure your API credentials are correctly entered in the sidebar.
2.  **Select Language**: Choose whether your input text is "English" or "Non-English" from the dropdown menu.
3.  **Input Text**: Paste the patient's clinical description into the text area.
4.  **Analyze**: Click the "开始分析 (Start Analysis) button.
5.  **View Results**: The application will display the translated text (if applicable), the extracted phenotypic phrases, and a final table with HPO term assignments and their validation status.
6.  **Download**: Click the "Download results as CSV file" button to save the results to your computer.

## 🛠️ Technical Stack

- **Backend**: Python
- **Web Framework**: Streamlit
- **AI/ML**:
    - **LLM Interaction**: OpenAI Python library
    - **Vector Search**: FAISS (Facebook AI Similarity Search)
    - **Embeddings**: Sentence-Transformers
- **Data Handling**: Pandas, NumPy

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgements

This work is an implementation and extension of the concepts presented in the paper "Improving automated deep phenotyping through large language models using retrieval-augmented generation" by Garcia et al., published in *Genome Medicine*. We are grateful for their foundational research.