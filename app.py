# app.py (Final version with correct structure and defensive unpacking)

import os
import json
import re
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss

# --- 1. Configuration and Constants ---
load_dotenv()
st.set_page_config(page_title="PhenoRAG App", layout="wide")
HPO_JSON_PATH = Path("data/hp.json")


# --- 2. Function and Class Definitions ---

@st.cache_data(show_spinner="正在解析 HPO 数据文件 (Parsing HPO data)...")
def load_hpo_data(filepath: Path):
    # ... (This function is correct, keep as is)
    if not filepath.exists():
        st.error(f"错误: 未找到 `hp.json` 文件 (Error: hp.json not found). 请确保它位于 `data/` 文件夹中。")
        st.stop()
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    parsed_terms, hpo_validator_dict = [], {}
    nodes = data.get("graphs", [{}])[0].get("nodes", [])
    for node in nodes:
        if 'id' in node and node['id'].startswith("http://purl.obolibrary.org/obo/HP_"):
            hpo_id = node['id'].split('/')[-1].replace('_', ':')
            name = node.get('lbl', '')
            synonyms = [s.get('val', '') for s in node.get('meta', {}).get('synonyms', [])]
            if name:
                parsed_terms.append({"hpo_id": hpo_id, "name": name, "synonyms": "; ".join(synonyms)})
                hpo_validator_dict[hpo_id] = name
    return parsed_terms, hpo_validator_dict


@st.cache_resource(show_spinner="正在构建向量知识库 (Building vector knowledge base)...")
def build_knowledge_base(_hpo_data_and_validator):
    """Builds the FAISS index and mappings. This is cached as a resource."""
    model_name = "BAAI/bge-small-en-v1.5"
    embedding_model = SentenceTransformer(model_name, device='cpu')

    if isinstance(_hpo_data_and_validator, tuple) and len(_hpo_data_and_validator) == 2:
        hpo_data_list = _hpo_data_and_validator[0]
    else:
        hpo_data_list = _hpo_data_and_validator

    if not isinstance(hpo_data_list, list):
        st.error("构建知识库时发生严重错误：未能获取到正确的HPO术语列表。")
        st.stop()

    corpus, hpo_map = [], {}
    for term in hpo_data_list:
        phrases = [term['name']] + term['synonyms'].split('; ')
        for phrase in phrases:
            clean_phrase = phrase.strip()
            if clean_phrase:
                corpus.append(clean_phrase.lower())
                hpo_map[clean_phrase.lower()] = {"hpo_id": term['hpo_id'], "name": term['name']}

    embeddings = embedding_model.encode(corpus, convert_to_numpy=True).astype('float32')
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    return faiss_index, corpus, hpo_map


class RAG_HPO_Pipeline:
    def __init__(self, api_key: str, base_url: str, model: str, knowledge_base: tuple, hpo_validator: dict):
        self.llm_client = OpenAI(api_key=api_key, base_url=base_url)
        self.llm_model = model
        self.embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device='cpu')
        self.faiss_index, self.corpus, self.hpo_map = knowledge_base
        self.hpo_validator = hpo_validator


    def translate_to_english(self, text: str):
        system_prompt = (
            "You are an expert medical translator. Your task is to translate the following text into clear, "
            "concise, and accurate clinical English. Preserve all medical details, symptoms, and findings. "
            "Do not add any interpretation, just perform a direct and professional translation."
        )
        try:
            response = self.llm_client.chat.completions.create(model=self.llm_model,
                                                               messages=[{"role": "system", "content": system_prompt},
                                                                         {"role": "user",
                                                                          "content": f"Translate this text:\n\n{text}"}],
                                                               temperature=0.0)
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Translation error: {e}");
            return None

    def _extract_phenotypes_from_text(self, english_text: str):

        system_prompt = (
            "You are a clinical genetics expert. Extract all distinct phenotypic phrases describing patient abnormalities from the clinical text. Ignore family history and negations! Do not describe the phenotype of the patient's relatives! "
            "Return the output as a JSON object with a single key 'phenotypes' containing a list of strings."
        )
        try:
            response = self.llm_client.chat.completions.create(model=self.llm_model,
                                                               messages=[{"role": "system", "content": system_prompt},
                                                                         {"role": "user",
                                                                          "content": f"Clinical text:\n\n{english_text}"}],
                                                               response_format={"type": "json_object"}, temperature=0.0)
            parsed_json = json.loads(response.choices[0].message.content)
            if isinstance(parsed_json, dict):
                return parsed_json.get('phenotypes', [])
            elif isinstance(parsed_json, list):
                return parsed_json
            else:
                st.warning("Phenotype extraction returned an unknown JSON format."); return []
        except Exception as e:
            st.error(f"Error extracting phenotype phrases: {e}");
            return []

    def _assign_hpo_terms(self, phrases: list, english_text: str):
        assignments, progress_bar = [], st.progress(0, text="正在为提取的短语分配HPO术语（Assigning HPO terms）...")
        for i, phrase in enumerate(phrases):
            progress_text = f"正在处理短语（Processing phrase） ({i + 1}/{len(phrases)}): '{phrase}'..."
            progress_bar.progress((i + 1) / len(phrases), text=progress_text)
            query_embedding = self.embedding_model.encode([phrase], convert_to_numpy=True).astype('float32')
            _, indices = self.faiss_index.search(query_embedding, 10)
            retrieved_candidates, seen_hpo_ids = [], set()
            for idx in indices[0]:
                retrieved_phrase = self.corpus[idx]
                hpo_info = self.hpo_map[retrieved_phrase]
                if hpo_info['hpo_id'] not in seen_hpo_ids:
                    retrieved_candidates.append(hpo_info);
                    seen_hpo_ids.add(hpo_info['hpo_id'])
            candidate_context = "\n".join([f"- {c['hpo_id']}: {c['name']}" for c in retrieved_candidates])
            system_prompt = (
                "You are an expert in medical ontology mapping. Select the single most accurate and specific "
                "HPO term for the given clinical phrase from the provided candidate list. "
                "Use the original text for context. Your response MUST be ONLY a single JSON object with 'hpo_id' and 'hpo_name'. "
                "Do not add any explanations or introductory text."
            )
            user_prompt = f"Original Clinical Text:\n---\n{english_text}\n---\n\nClinical Phrase to Map: \"{phrase}\"\n\nCandidate HPO Terms (Choose ONE):\n{candidate_context}"
            try:
                response = self.llm_client.chat.completions.create(model=self.llm_model, messages=[
                    {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                                                                   temperature=0.0)
                raw_content = response.choices[0].message.content
                json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                if json_match:
                    assignment = json.loads(json_match.group(0))
                    if assignment.get('hpo_id'):
                        assignments.append({"原始短语 (phrase)": phrase, "HPO ID (model)": assignment['hpo_id'],
                                            "HPO name (model)": assignment['hpo_name']})
                else:
                    st.warning(f"处理短语 '{phrase}' 时，模型未返回有效的JSON（Invalid JSON）。")
            except Exception as e:
                st.warning(f"处理短语 '{phrase}' 时出错: {e}")
        progress_bar.empty();
        return assignments

    def _validate_and_format_assignments(self, assignments: list):
        # ... (keep your existing _validate_and_format_assignments code)
        if not assignments: return []
        validated_results = []
        for term in assignments:
            model_hpo_id, model_hpo_name = term.get("HPO ID (model)"), term.get("HPO 名称 (model)")
            official_name = self.hpo_validator.get(model_hpo_id)
            if official_name:
                status = "✅ Fully correct" if official_name == model_hpo_name else "⚠️ Name mismatch"
                term["验证（verify）"] = status;
                term["官方HPO（official）"] = official_name
            else:
                term["验证（verify）"] = "❌ Invalid ID";
                term["官方HPO（official）"] = "N/A"
            validated_results.append(term)
        return validated_results

    def run(self, text: str, source_language: str):
        translated_text = None
        if source_language == 'English':
            st.info("跳过翻译步骤（skipping translation）。")
            english_text = text
        else:
            with st.spinner("步骤 1/4: 正在将输入文本翻译成英文（Translating）..."):
                english_text = self.translate_to_english(text)
                translated_text = english_text
            if not english_text:
                return None

        step_offset = 1 if translated_text is not None else 0
        total_steps = 3 + step_offset
        with st.spinner(f"步骤 {1 + step_offset}/{total_steps}: 正在从英文文本中提取表型短语..."):
            phenotypic_phrases = self._extract_phenotypes_from_text(english_text)
        st.info(f"步骤 {2 + step_offset}/{total_steps}: 正在为提取的短语分配HPO术语（Assigning HPO terms）...")
        final_assignments = self._assign_hpo_terms(phenotypic_phrases, english_text)
        st.info(f"步骤 {3 + step_offset}/{total_steps}: 正在验证分配结果（Validating assignments）...")
        validated_assignments = self._validate_and_format_assignments(final_assignments)

        return {
            "translated_text": translated_text,
            "extracted_phrases": phenotypic_phrases,
            "final_assignments": validated_assignments
        }


# --- 3. Main App Execution Block ---

# First, load/build all heavy resources.
hpo_data, hpo_validator = load_hpo_data(HPO_JSON_PATH)
knowledge_base = build_knowledge_base(hpo_data)

# render the UI.
st.title("🧬 PhenoRAG: An intelligent Human Phenotype Ontology analysis tool")
st.markdown("输入任何语言的患者临床描述，本工具将使用大语言模型自动提取表型并将其映射到标准的人类表型本体术语。Enter a patient’s clinical description in any language...")

with st.sidebar:
    st.header("⚙️ 模型配置（LLM Settings）")
    st.info("Please provide your own OpenAI-compatible API credentials.")

    # 不再使用 os.getenv 作为默认值，而是提供一个常见的默认值或留空
    api_base_url = st.text_input(
        "API Base URL",
        value="https://api.siliconflow.cn/v1",  # 提供一个示例
        help="Enter the API endpoint URL. Default is Groq."
    )
    api_key = st.text_input(
        "API Key",
        value="",  # 默认留空，强制用户输入
        type="password",
        help="Enter your personal API key."
    )
    llm_model = st.text_input(
        "Model Name",
        value="deepseek-ai/DeepSeek-V3",  # 提供一个示例
        help="Enter the model name you want to use."
    )
# Check for API config.
if not all([api_key, api_base_url, llm_model]):
    st.warning("请在左侧侧边栏中配置有效的API信息以开始（Configure valid API）。")
    st.stop()

# Create the lightweight pipeline instance.
pipeline = RAG_HPO_Pipeline(api_key, api_base_url, llm_model, knowledge_base, hpo_validator)

# Render the rest of the main page UI.
language_option = st.selectbox(
    '请选择输入文本的语言（Select input text language）:',
    ('Non-English', 'English'),
    index=0 # 默认选择第一个，即“非英文”
)

# 直接判断完整的字符串，避免歧义
source_language = 'English' if language_option == 'English' else 'Non-English'

sample_text_cn = ("患儿，男，7岁，因发育迟缓就诊。家长反映其语言发育明显落后，4岁才说出第一个词。"
                  "患儿从2岁起频繁出现抽搐。体格检查发现，其头围小于同龄人正常范围，且双眼眼距过宽。"
                  "生长曲线显示其身高持续低于第3百分位，提示身材矮小。"
                  "此外，手指和肘部关节活动度过大。")
sample_text_en = (
    "A 7-year-old male presented with developmental delay. Parents reported markedly delayed speech development, with first words at age 4. "
    "He has a history of seizures since age 2. Physical examination revealed microcephaly and ocular hypertelorism. "
    "Growth charts show height below the 3rd percentile, indicating short stature. "
    "Additionally, there is hyperextensibility of finger and elbow joints.")

default_text = sample_text_en if language_option == 'English' else sample_text_cn
user_input = st.text_area(
    "请输入患者临床描述（Enter patient clinical description）:",
    value=default_text,
    height=250
)

# ---  main logic execution ---
if st.button("开始分析（Start Analysis）", type="primary"):
    if user_input.strip() and pipeline:
        # The 'pipeline.run()' call MUST be inside the button block.
        results = pipeline.run(user_input, source_language)

        if results:
            st.divider()
            st.subheader("📝 分析结果（Analysis Results）")
            if results['translated_text']:
                with st.expander("1. 翻译后的英文临床文本 (Translated English Text)"):
                    st.text(results['translated_text'])

            exp_prefix = "2." if results['translated_text'] else "1."
            with st.expander(f"{exp_prefix} 提取出的英文表型短语 (Extracted Phenotype Phrases)"):
                st.write(results['extracted_phrases'])

            sub_prefix = "3." if results['translated_text'] else "2."
            st.subheader(f"{sub_prefix} 最终HPO术语分配与验证结果 (Final Assignments & Validation)")
            if results['final_assignments']:
                df = pd.DataFrame(results['final_assignments'])
                cols_order = ["原始短语 (phrase)", "验证（verify）", "HPO ID (model)", "HPO name (model)", "官方HPO（official）"]
                df_to_save = df[cols_order]

                csv_data_bytes = df_to_save.to_csv(index=False).encode('utf-8-sig')
                timestamp = datetime.now().strftime("%Y%m%d_%HM%S")
                filename = f"hpo_analysis{timestamp}.csv"
                st.download_button(
                    label="📥 Download results as CSV",
                    data=csv_data_bytes, file_name=filename, mime='text/csv'
                )

                st.dataframe(df_to_save, use_container_width=True, hide_index=True)
            else:
                st.info("未能从文本中分配任何HPO术语（No HPO terms were assigned from the text）。")
    else:
        st.error("请输入有效的临床描述。")