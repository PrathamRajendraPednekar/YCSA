

import streamlit as st
import pandas as pd
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import plotly.graph_objects as go
import tempfile


# ------------------ Page Config ------------------
st.set_page_config(layout="wide", page_title="YouTube Comment Sentiment Analysis", page_icon="📊")
# st.title("📊 YouTube Comment Sentiment Analysis")

# ------------------ Sidebar ------------------

st.sidebar.image("YCSA_logo.png", use_container_width=True)
st.sidebar.info("Upload a CSV file of YouTube comments and generate sentiment analysis charts.")
st.sidebar.markdown("✨ **Tips:**\n- Upload .csv file \n- Ensure proper columns\n- Charts may take a few minutes")

# ------------------ Custom CSS ------------------
st.markdown(
    """
           <h1 style="
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        font-family: Arial, sans-serif;
        background: linear-gradient(90deg, #00c6ff, #7a00ff, #ff00cc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    ">
        YouTube Comment Sentiment Analysis
    </h1>
    
    

    <style>
    div.stButton > button {
        color: white !important;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #764ba2, #667eea);
    }

    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.10) !important;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 25px;
        margin-top: 20px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }

    [data-testid="stFileDropzone"] {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px dashed rgba(255, 255, 255, 0.4);
        border-radius: 12px;
    }

    [data-testid="stFileUploader"] * {
        color: #ffffff !important;
    }

    label[data-testid="stFileUploaderLabel"] {
        font-size: 20px;
        font-weight: bold;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Background Gradient ------------------
def set_gradient_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(to right, #cc2b5e, #753a88);
            background-size: cover;
            background-position: center;
            color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_gradient_background()

# ------------------ Upload and Process CSV ------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded file temporarily
        csv_path = os.path.join(tmpdir, "uploaded_data.csv")
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Preview Data
        df = pd.read_csv(uploaded_file)
        st.subheader("🔎 File Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Quick stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("File Size (KB)", round(uploaded_file.size/1024, 2))

        if st.button("📈 Generate Charts"):
            with st.spinner("⏳ Running analysis... Please wait..."):
                try:
                    # ✅ Pass CSV path to notebook
                    os.environ["CSV_PATH"] = csv_path

                    with open("YCSA.ipynb", encoding="utf-8") as f:
                        nb = nbformat.read(f, as_version=4)

                    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
                    ep.preprocess(nb, {"metadata": {"path": tmpdir}})

                except Exception as e:
                    st.error(f"❌ Notebook execution failed:\n{e}")
                    st.stop()

            # ------------------ Display Charts ------------------
            st.success("✅ Analysis complete!")
            st.subheader("📊 Generated Charts:")

  
            plot_count = 0
            for cell in nb.cells:
                if cell.cell_type == "code" and "fig" in cell.source:
                    for output in cell.get("outputs", []):
                        if output.output_type == "display_data":
                            fig_json = output.get("data", {}).get("application/vnd.plotly.v1+json", None)
                            if fig_json:
                                plot_count += 1
                                with st.expander(f"📈 Chart {plot_count}", expanded=True):
                                    st.plotly_chart(go.Figure(fig_json), use_container_width=True)

            if plot_count == 0:
                st.warning("⚠️ No charts were found in the notebook. Make sure your notebook cells output Plotly figs named `fig1`, `fig2`, ...")


            # ------------------ Footer ------------------
            st.markdown(
                "<hr><p style='text-align:center;color:white;'>©2025 Pratham Pednekar & Shravan Dige | YCSA Dashboard</p>",
                unsafe_allow_html=True
            )
