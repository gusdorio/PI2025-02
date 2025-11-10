"""
Results Page
============
View ML processing results with card-based interface.

Fetches data from 3 collections:
- datasets: Raw data
- pipeline_runs: Summary of execution and dashboard metrics
- ml_results: Detailed model outputs (confusion matrix, etc.)
"""

import streamlit as st
import sys
import os
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_database_connection, check_database_health


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Results",
    page_icon="📊",
    layout="wide"
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_dataset_card(dataset_doc, pipeline_run_doc=None):
    """(Função sem alteração)"""
    batch_id = dataset_doc.get('_id', 'unknown')
    filename = dataset_doc.get('filename', 'unnamed')
    timestamp = dataset_doc.get('upload_timestamp', datetime.now())

    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except:
            timestamp = datetime.now()

    metadata = dataset_doc.get('metadata', {})
    row_count = metadata.get('row_count', 0)
    column_count = metadata.get('column_count', 0)

    status = 'uploaded'
    if pipeline_run_doc:
        status = pipeline_run_doc.get('status', 'uploaded')

    with st.container():
        card = st.container()
        with card:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"### 📁 {filename}")
                st.caption(f"Batch ID: {batch_id[:20]}...")
            with col2:
                status_emoji = {
                    'completed': '✅', 'failed': '❌',
                    'processing': '⏳', 'uploaded': '📤'
                }.get(status, '❓')
                st.markdown(f"**Status:** {status_emoji} {status.title()}")
            with col3:
                if st.button("View Details", key=f"view_{batch_id}"):
                    st.session_state['selected_dataset'] = batch_id
                    st.rerun()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{row_count:,}")
            with col2:
                st.metric("Columns", column_count)
            with col3:
                st.metric("Upload Time", timestamp.strftime('%Y-%m-%d %H:%M'))
            with col4:
                # Mostra o status de ML do pipeline_run
                ml_status = "N/A"
                if pipeline_run_doc:
                    ml_status = pipeline_run_doc.get('summary', {}).get('ml_status', 'N/A')
                st.metric("ML Status", ml_status.title())

        st.markdown("---")


def show_dataset_details(db_conn, dataset_doc, pipeline_run_doc=None):
    """
    Show detailed view of a dataset.
    Agora busca dados das 3 coleções.
    """
    filename = dataset_doc.get('filename', 'unnamed')
    batch_id = dataset_doc.get('_id', 'unknown')

    # =========================================================
    # *** BUSCA OS DADOS DETALHADOS DE ML ***
    # =========================================================
    ml_results_doc = None
    if db_conn:
        try:
            ml_results_collection = db_conn.get_collection('ml_results')
            ml_results_doc = ml_results_collection.find_one({'batch_id': batch_id})
        except Exception as e:
            st.warning(f"Could not fetch ML results: {e}")

    # Header with back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Results"):
            st.session_state.pop('selected_dataset', None)
            st.rerun()
    with col2:
        st.title(f"Dataset: {filename}")
        st.caption(f"Batch ID: {batch_id}")

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 ML Results",
        "📊 Original Data",
        "📈 Visualizations",
        "ℹ️ Metadata"
    ])

    # =========================================================
    # TAB 1: ML RESULTS (AGORA É A PRIMEIRA TAB)
    # =========================================================
    with tab1:
        st.subheader("Machine Learning Results")

        if pipeline_run_doc:
            summary = pipeline_run_doc.get('summary', {})
            ml_status = summary.get('ml_status', 'not_processed')

            if ml_status == 'skipped':
                st.info("🔄 ML processing was skipped (minimal mode)")
            
            elif ml_status == 'failed':
                st.error("❌ ML processing failed.")
                if 'ml_summary_dashboard' in summary:
                    st.json({"error": summary['ml_summary_dashboard'].get('reason')})
            
            elif ml_status == 'completed':
                st.success("✅ ML processing completed")
                
                # Pega o resumo de métricas do pipeline_run
                ml_summary = summary.get('ml_summary_dashboard', {})
                
                if ml_summary:
                    st.markdown("#### Resumo das Métricas (AutoML)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Melhor Modelo", ml_summary.get('best_model_name', 'N/A'))
                    with col2:
                        st.metric("Tipo de Problema", ml_summary.get('problem_type', 'N/A'))
                    
                    metrics_df = pd.DataFrame(ml_summary.get('summary_metrics', []))
                    if not metrics_df.empty:
                        st.dataframe(metrics_df, use_container_width=True)
                    else:
                        st.warning("No summary metrics found.")
                
                # Pega os resultados detalhados da coleção ml_results
                if ml_results_doc:
                    st.markdown("---")
                    st.markdown("#### Resultados Detalhados do Melhor Modelo")
                    
                    results_data = ml_results_doc.get('results', {})
                    best_model_name = ml_summary.get('best_model_name')
                    
                    if best_model_name and best_model_name in results_data:
                        best_model_details = results_data[best_model_name]
                        
                        # Exibe detalhes com base no tipo de problema
                        if ml_summary.get('problem_type') == 'regression':
                            st.json({
                                "CV R²": best_model_details.get('cv_r2'),
                                "Test R²": best_model_details.get('test_r2'),
                                "CV RMSE": best_model_details.get('cv_rmse'),
                                "Test RMSE": best_model_details.get('test_rmse'),
                            })
                            st.subheader("Range de Erro (Regressão)")
                            st.json(best_model_details.get('error_range', {}))
                            
                        elif ml_summary.get('problem_type') == 'classification':
                            st.json({
                                "CV Accuracy": best_model_details.get('cv_accuracy'),
                                "Test Accuracy": best_model_details.get('test_accuracy'),
                                "CV F1-Score": best_model_details.get('cv_f1'),
                            })
                            st.subheader("Matriz de Confusão (Classificação)")
                            st.code(best_model_details.get('confusion_matrix', 'N/A'))
                            st.subheader("Relatório de Classificação")
                            st.code(best_model_details.get('classification_report', 'N/A'))
                            
                        with st.expander("Ver dados brutos de resultados detalhados"):
                            st.json(best_model_details)
                    else:
                        st.error("Não foi possível encontrar os detalhes do melhor modelo.")
                else:
                    st.warning("Nenhum documento de resultados detalhados de ML encontrado.")
            
            else:
                st.warning(f"⚠️ ML Status: {ml_status}")
        else:
            st.info("📤 Dataset uploaded but not yet processed through ML pipeline")

    # =========================================================
    # TAB 2: ORIGINAL DATA (Sem alteração)
    # =========================================================
    with tab2:
        st.subheader("Original Dataset")
        data_list = dataset_doc.get('data', [])
        if data_list:
            df = pd.DataFrame(data_list)
            st.dataframe(df.head(100), use_container_width=True, height=400)
        else:
            st.warning("No data found")

    # =========================================================
    # TAB 3: VISUALIZATIONS (Sem alteração)
    # =========================================================
    with tab3:
        st.subheader("Data Visualizations")
        data_list = dataset_doc.get('data', [])
        if data_list:
            df = pd.DataFrame(data_list)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("Select X axis", numeric_cols)
                with col2:
                    y_col = st.selectbox("Select Y axis", numeric_cols[1:] + [numeric_cols[0]])
                if x_col and y_col:
                    st.scatter_chart(df[[x_col, y_col]].head(500))
            else:
                st.info("No numeric columns for visualization")
        else:
            st.warning("No data for visualization")

    # =========================================================
    # TAB 4: METADATA (Sem alteração)
    # =========================================================
    with tab4:
        st.subheader("Dataset Metadata")
        metadata = dataset_doc.get('metadata', {})
        st.json(metadata)
        
        st.markdown("### Processing Information")
        if pipeline_run_doc:
            st.json(pipeline_run_doc.get('summary', {}))
        else:
            st.info("No pipeline run information found.")


# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("📊 ML Processing Results")
st.markdown("View and explore your uploaded datasets and their ML analysis results.")
st.markdown("---")

db_conn = get_database_connection()
is_healthy, message = check_database_health(db_conn)

if not is_healthy:
    st.error("❌ Database connection is not available. Cannot load results.")
    st.stop()

# Check if user selected a specific dataset
if 'selected_dataset' in st.session_state:
    # Detail view mode
    try:
        datasets_collection = db_conn.get_collection('datasets')
        dataset_doc = datasets_collection.find_one({'_id': st.session_state['selected_dataset']})

        if dataset_doc:
            pipeline_runs_collection = db_conn.get_collection('pipeline_runs')
            pipeline_run_doc = pipeline_runs_collection.find_one({'batch_id': st.session_state['selected_dataset']})
            
            # Passa a conexão do DB para a função de detalhes
            show_dataset_details(db_conn, dataset_doc, pipeline_run_doc)
        else:
            st.error("Dataset not found")
            # ... (lógica de 'voltar' sem alteração) ...

    except Exception as e:
        st.error(f"Error loading dataset details: {e}")
        # ... (lógica de 'voltar' sem alteração) ...

else:
    # List view mode - show all datasets as cards
    # (Lógica de filtros e listagem sem alteração)
    with st.sidebar:
        st.markdown("### 🔍 Filters")
        time_filter = st.selectbox("Time Range", ["All Time", "Last 24 Hours", "Last 7 Days"])
        status_filter = st.multiselect("Status", ["uploaded", "completed", "failed", "processing"], default=["uploaded", "completed", "failed", "processing"])
        sort_order = st.radio("Sort By", ["Newest First", "Oldest First"])

    try:
        datasets_collection = db_conn.get_collection('datasets')
        pipeline_runs_collection = db_conn.get_collection('pipeline_runs')

        query = {}
        # (Lógica de filtros de tempo sem alteração)
        
        sort_direction = -1 if sort_order == "Newest First" else 1
        datasets = list(datasets_collection.find(query).sort('upload_timestamp', sort_direction).limit(50))

        if datasets:
            all_runs = list(pipeline_runs_collection.find())
            runs_by_batch = {run['batch_id']: run for run in all_runs}
            
            # (Métricas de resumo sem alteração)
            total = len(datasets)
            completed = sum(1 for d in datasets if runs_by_batch.get(d['_id'], {}).get('status') == 'completed')
            failed = sum(1 for d in datasets if runs_by_batch.get(d['_id'], {}).get('status') == 'failed')
            pending = total - completed - failed
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("📁 Total Datasets", total)
            with col2: st.metric("✅ Completed", completed)
            with col3: st.metric("❌ Failed", failed)
            with col4: st.metric("⏳ Pending", pending)
            st.markdown("---")
            
            st.subheader("📋 Datasets")
            
            # (Lógica de exibição dos cards sem alteração)
            for dataset in datasets:
                batch_id = dataset.get('_id')
                pipeline_run = runs_by_batch.get(batch_id)
                status = 'uploaded'
                if pipeline_run:
                    status = pipeline_run.get('status', 'uploaded')
                if status in status_filter:
                    create_dataset_card(dataset, pipeline_run)

        else:
            st.info("📭 No datasets found. Upload data through the Upload page.")

    except Exception as e:
        st.error(f"Error loading datasets: {e}")