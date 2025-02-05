"""Streamlit app for Azure Podcast Generator"""

import logging
import os
from datetime import datetime

import streamlit as st
from const import AZURE_HD_VOICES
from dotenv import find_dotenv, load_dotenv
from utils.identity import check_claim_for_tenant
from utils.llm import (
    document_to_english_learning_podcast,
    document_to_podcast_script,
)
from utils.speech import podcast_script_to_ssml, text_to_speech
from utils.video_scraper import scrape_video
from utils.web_scraper import scrape_webpage

# optional: only allow specific tenants to access the app (using Azure Entra ID)
headers = st.context.headers
if os.getenv("ENTRA_AUTHORIZED_TENANTS") and headers.get("X-Ms-Client-Principal"):
    authorized_tenants = os.environ["ENTRA_AUTHORIZED_TENANTS"].split(",")
    ms_client_principal = headers.get("X-Ms-Client-Principal")
    access = check_claim_for_tenant(ms_client_principal, authorized_tenants)

    if access is not True:
        st.error("Access denied.")
        st.stop()


st.set_page_config(
    page_title="Azure Podcast Generator",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("AI Podcast Generator")

# Custom CSS for the modal-like appearance
st.markdown(
    """
    <style>
    .source-container {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 30px;
        margin: 20px 0;
    }
    .source-header {
        color: white;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .source-description {
        color: #888;
        margin-bottom: 30px;
    }
    .scrollable-text {
        max-height: 200px;
        overflow-y: scroll;    /* 垂直滚动条 */
        overflow-x: hidden;    /* 隐藏水平滚动条 */
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        background-color: #f5f5f5;
        word-wrap: break-word;
        white-space: normal;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Sources Container
with st.container():
    st.markdown('<div class="source-container">', unsafe_allow_html=True)
    st.markdown('<div class="source-header">Add sources</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="source-description">Choose your content source to generate podcast.</div>',
        unsafe_allow_html=True,
    )

    # Main upload area
    upload_col, link_col, video_col = st.columns(3)

    with upload_col, st.form(key="document_form", clear_on_submit=True):
        st.markdown("### 📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["txt", "pdf", "docx"],
            help="Supported formats: TXT, PDF, DOCX",
        )
        doc_submit = st.form_submit_button("Upload")

    with link_col, st.form(key="web_form", clear_on_submit=True):
        st.markdown("### 🔗 Web Link")
        url = st.text_input("Enter website URL")
        web_submit = st.form_submit_button("Process")

    with video_col, st.form(key="bilibili_form", clear_on_submit=True):
        st.markdown("### 📺 Bilibili Video")
        video_bvid = st.text_input(
            "Enter BV ID",
            placeholder="Example: BV1GJ411x7h7",
            help="Enter the BV ID from the Bilibili video URL",
        )
        video_submit = st.form_submit_button("Process")

    if video_submit and video_bvid:
        try:
            with st.spinner("Processing Bilibili video..."):
                document_response = scrape_video(video_bvid)
                st.success("Video content processed successfully!")

                # 添加到源列表
                source_item = {
                    "type": "bilibili",
                    "url": f"BV{video_bvid}",
                    "content": document_response.markdown,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.session_state.source_list.append(source_item)

        except Exception as e:
            st.error(f"Error processing Bilibili video: {str(e)}")

# 在文件开头添加
if "source_list" not in st.session_state:
    st.session_state.source_list = []

# Process sources and show content
if web_submit and url:
    try:
        with st.spinner("Processing webpage..."):
            document_response = scrape_webpage(url)
            st.success("Web content processed successfully!")

            # 添加到源列表
            source_item = {
                "type": "web",
                "url": url,
                "content": document_response.markdown,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state.source_list.append(source_item)

    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")

# 显示源列表
if st.session_state.source_list:
    st.markdown("### Source List")
    for idx, source in enumerate(st.session_state.source_list):
        with st.expander(
            f"Source {idx + 1}: {source['type'].upper()} - {source['timestamp']}"
        ):
            st.markdown('<div class="scrollable-text">', unsafe_allow_html=True)
            st.markdown(source["content"])
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button(f"Remove Source {idx + 1}", key=f"remove_{idx}"):
                st.session_state.source_list.pop(idx)
                st.rerun()

    # 合并所有源的内容
    combined_content = "\n\n".join(
        [source["content"] for source in st.session_state.source_list]
    )
    st.session_state.input_content = combined_content

    # 根据源列表长度动态调整token数量
    base_tokens = 3000  # 基础token数调整为3000
    tokens_per_source = 1000  # 每个额外源增加的token数
    max_tokens = min(
        base_tokens + (len(st.session_state.source_list) - 1) * tokens_per_source, 6000
    )  # 上限调整为6000

    # 添加播客模式选择
    st.markdown("### Podcast Settings")
    podcast_mode = st.radio(
        "Select Podcast Mode",
        ["Standard Podcast", "English Learning (JP/EN)"],
        help="Choose between standard podcast or English learning format with Japanese explanations",
    )

    # 语音设置根据模式调整
    st.markdown("### Voice Settings")
    col1, col2 = st.columns(2)
    with col1:
        if podcast_mode == "Standard Podcast":
            voice_1_label = "Select Voice 1 (Host)"
        else:
            voice_1_label = "Select Voice 1 (English Host)"
        voice_1 = st.selectbox(voice_1_label, options=AZURE_HD_VOICES, index=0)
    with col2:
        if podcast_mode == "Standard Podcast":
            voice_2_label = "Select Voice 2 (Co-host)"
        else:
            voice_2_label = "Select Voice 2 (Japanese Host)"
        voice_2 = st.selectbox(voice_2_label, options=AZURE_HD_VOICES, index=1)

    # 显示并允许调整token数量
    max_tokens = st.slider(
        "Max Length (tokens)",
        min_value=1000,
        max_value=6000,  # 上限调整为6000
        value=max_tokens,
        help=f"Suggested length based on {len(st.session_state.source_list)} sources. Base: 3000, +1000 per additional source",
    )

    # Generate Podcast Button
    st.divider()
    if st.button("Generate Podcast", type="primary"):
        with st.spinner("Generating podcast..."):
            try:
                if podcast_mode == "Standard Podcast":
                    podcast_response = document_to_podcast_script(
                        document=st.session_state.input_content,
                        max_tokens=max_tokens,
                        voice_1=voice_1,
                        voice_2=voice_2,
                    )
                else:
                    podcast_response = document_to_english_learning_podcast(
                        document=st.session_state.input_content,
                        max_tokens=max_tokens,
                        voice_1=voice_1,
                        voice_2=voice_2,
                    )

                # Create podcast data dictionary with explicit structure
                podcast_data = {
                    "script": podcast_response.podcast["script"],
                    "voice_1": voice_1,
                    "voice_2": voice_2,
                    "config": podcast_response.podcast.get("config", {}),
                }

                # Generate audio
                ssml = podcast_script_to_ssml(podcast_data)
                # st.write("Generated SSML:", ssml)  # Debug SSML output

                audio = text_to_speech(ssml)

                # Display audio player
                st.audio(audio, format="audio/wav")

                # Display script
                with st.expander("View podcast script"):
                    for item in podcast_response.podcast["script"]:
                        st.markdown(f"**{item['name']}**: {item['message']}")
            except Exception as e:
                st.error(f"Error generating podcast: {str(e)}")

# Footer
st.divider()
st.caption(
    "Created by [Mick Vleeshouwer](https://github.com/imicknl). The source code is available on [GitHub](https://github.com/iMicknl/azure-podcast-generator), contributions are welcome."
)

if __name__ == "__main__":
    load_dotenv(find_dotenv())

if os.getenv("DEBUG_MODE") == "true":
    logging.basicConfig(level=logging.INFO)
