"""
Data Upload Page
================
Upload data files, select target, and process through the data pipeline.

This page uses a two-step process:
1. File upload and validation
2. Target column selection
3. Pipeline execution (Transformation & ML Communication)
"""

import streamlit as st
import sys
import os
import pandas as pd
from io import BytesIO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.uploader import UploadService
from components.data_upload_pipeline import DataUploadPipeline, TransformMode


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Upload Data",
    page_icon="⬆️",
    layout="wide"
)

# ============================================================================
# SIDEBAR - ML SERVICE STATUS
# ============================================================================

with st.sidebar:
    st.markdown("### 🔧 Service Status")

    # Test ML service connection
    try:
        import requests
        response = requests.get("http://ml-model:5000", timeout=2)
        st.success("✅ ML Service: Online")
    except requests.ConnectionError:
        st.error("❌ ML Service: Offline")
        st.caption("Start with: `make dev-up`")
    except requests.Timeout:
        st.warning("⚠️ ML Service: Slow Response")
    except:
        st.info("ℹ️ ML Service: Unknown")

    st.markdown("---")
    st.caption("**Logs:** `make dev-logs-ml`")
    st.caption("**Status:** `make dev-status`")


# ============================================================================
# INITIALIZE SERVICES (SINGLETONS)
# ============================================================================

@st.cache_resource
def get_upload_service():
    """Initialize upload service with 50MB file size limit"""
    return UploadService(max_file_size_mb=50)

@st.cache_resource
def get_upload_pipeline():
    """Initialize upload pipeline with transformation disabled"""
    return DataUploadPipeline(transform_mode=TransformMode.NONE)

upload_service = get_upload_service()
pipeline = get_upload_pipeline()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def reset_session_state():
    """Clears all processing-related session state keys."""
    keys_to_clear = ['dataset', 'pipeline_result', 'upload_errors']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("📊 Data Upload")
st.markdown("Upload, select your target, and process your data.")
st.markdown("**Supported formats:** CSV, Excel (.xlsx, .xls) | **Max file size:** 50 MB")

st.markdown("---")

# ========================================================================
# STAGE 1: FILE UPLOAD & VALIDATION
# ========================================================================
st.subheader("1. Upload File")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=['xlsx', 'xls', 'csv'],
    help="Upload your data file (CSV or Excel format, max 50MB)",
    on_change=reset_session_state # Reseta o estado se um novo arquivo for enviado
)

if uploaded_file is not None:
    # Se o dataset ainda não foi validado, valide-o
    if 'dataset' not in st.session_state and 'upload_errors' not in st.session_state:
        with st.spinner("Validating file..."):
            # Apenas executa a ETAPA 1 (Validação e Leitura)
            dataset, errors = upload_service.process_upload(uploaded_file)
            
            if errors:
                st.session_state['upload_errors'] = errors
            else:
                st.session_state['dataset'] = dataset

    # Se a validação falhou, mostre os erros
    if 'upload_errors' in st.session_state:
        for error in st.session_state['upload_errors']:
            st.error(f"❌ Validation Failed: {error}")
        
    # Se a validação foi um sucesso e o dataset está no estado
    elif 'dataset' in st.session_state:
        dataset = st.session_state['dataset']
        st.success(f"✅ File '{dataset.filename}' validated successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Rows", f"{dataset.row_count:,}")
        with col2:
            st.metric("📋 Columns", dataset.column_count)
            
        st.dataframe(dataset.get_preview(5), use_container_width=True)

        # ========================================================================
        # STAGE 2: TARGET SELECTION
        # ========================================================================
        st.markdown("---")
        st.subheader("2. Select Target Column")
        
        # Pega a coluna alvo (se já foi selecionada) ou a última coluna como padrão
        default_index = len(dataset.column_names) - 1
        target_column = st.selectbox(
            "Select the column you want to predict (target)",
            options=dataset.column_names,
            index=default_index,
            help="This column will be used as the 'y' value for the machine learning model."
        )

        st.info(f"You selected **{target_column}** as the target column.")

        # ========================================================================
        # STAGE 3: PROCESS & SEND
        # ========================================================================
        st.markdown("---")
        st.subheader("3. Process and Send to ML Service")
        
        if st.button(f"Processar e Enviar '{target_column}' para ML", type="primary"):
            try:
                with st.spinner("Processing through data pipeline (Stages 2 & 3)..."):
                    # Executa o restante da pipeline (Transformação e Envio)
                    result = pipeline.execute(dataset, target_column)
                    st.session_state['pipeline_result'] = result

            except Exception as e:
                # Erro inesperado
                st.error(f"❌ An unexpected error occurred: {str(e)}")
                import traceback
                with st.expander("🔧 Technical Details"):
                    st.code(traceback.format_exc())

# ========================================================================
# STAGE 4: DISPLAY RESULTS
# ========================================================================
if 'pipeline_result' in st.session_state:
    result = st.session_state['pipeline_result']
    dataset = st.session_state['dataset'] # Recupera o dataset original
    
    st.markdown("---")
    st.subheader("4. Pipeline Status")

    if result.is_success:
        st.success(f"✅ {result.message}")
        
        ml_response = result.ml_response
        if ml_response:
            st.markdown("---")
            st.subheader("5. ML Service Response")

            # ... (O restante da lógica de exibição de resultados pode ser copiada
            #      do arquivo 'upload.py' original, pois ela já funciona) ...
            
            # Display response details
            col_ml1, col_ml2, col_ml3 = st.columns(3)
            with col_ml1:
                st.metric("Status", ml_response.get('status', 'unknown').upper())
            with col_ml2:
                st.metric("Batch ID", ml_response.get('batch_id', 'N/A')[:20] + "...")
            with col_ml3:
                st.metric("Rows Stored", f"{ml_response.get('row_count', 0):,}")

            # Show processing details
            if 'processing_summary' in ml_response:
                summary_data = ml_response['processing_summary']
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)

                with col_s1:
                    st.metric("ML Status", summary_data.get('ml_status', 'N/A'))
                with col_s2:
                    st.metric("Processing Mode", summary_data.get('processing_mode', 'N/A'))
                with col_s3:
                    st.metric("Storage Status", summary_data.get('storage_status', 'N/A'))
                with col_s4:
                    has_missing = summary_data.get('has_missing_values', False)
                    st.metric("Data Quality", "Has Missing" if has_missing else "Complete")

            with st.expander("📋 View Full ML Service Response"):
                st.json(ml_response)

    else:
        # Pipeline failed
        st.error(f"❌ {result.message}")
        if result.errors:
            st.markdown("### Error Details")
            for error in result.errors:
                st.error(f"**Stage:** {error.get('stage', 'unknown')} - **Error:** {error.get('error', 'Unknown error')}")

# Se nenhum arquivo foi enviado
elif 'dataset' not in st.session_state:
    st.info("👆 Please upload a file to begin")