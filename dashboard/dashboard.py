import streamlit as st
import polars as pl
import pymongo
import os

# 1. Configuração da Página
st.set_page_config(layout="wide", page_title="Ingestão Cosmos DB")

st.title("Ingestão de Dados para Cosmos DB 🚀")
st.write("Faça o upload do seu arquivo .csv ou .json para iniciar a ingestão.")

# 2. Carregamento de Segredos (Variáveis de Ambiente)
# Em produção (Azure Container Apps), configure estas como variáveis de ambiente.
# Localmente, crie um arquivo .streamlit/secrets.toml
try:
    # Busca a connection string principal do Cosmos DB
    MONGO_URI = st.secrets["COSMOS_CONNECTION_STRING"]
    # Define o nome do banco de dados (será criado se não existir)
    MONGO_DB_NAME = st.secrets["COSMOS_DB_NAME"]
    # Define o nome da coleção (será criada se não existir)
    MONGO_COLLECTION_NAME = st.secrets["COSMOS_COLLECTION_NAME"]
    
except KeyError:
    # Falha se as variáveis de ambiente não estiverem configuradas
    st.error("ERRO CRÍTICO: As variáveis de ambiente (secrets) do Cosmos DB não foram configuradas.")
    st.info("Configure COSMOS_CONNECTION_STRING, COSMOS_DB_NAME, e COSMOS_COLLECTION_NAME.")
    # Interrompe a execução do script se não houver credenciais
    st.stop()


# 3. Função de Ingestão de Dados 
# Encapsula a lógica de conexão e envio para o Cosmos DB
def send_to_cosmos_db(dataframe, uri, db_name, collection_name):
    """
    Conecta ao Cosmos DB (API MongoDB) e insere um DataFrame Polars.
    """
    try:
        # Converte o DataFrame Polars para uma lista de dicionários
        data_to_insert = dataframe.to_dicts()

        # Verifica se há dados a serem inseridos
        if not data_to_insert:
            st.warning("O arquivo estava vazio ou era inválido. Nada foi enviado ao banco de dados.")
            return 0 # Retorna 0 registros inseridos

        # Inicializa o cliente do MongoDB
        client = pymongo.MongoClient(uri)
        
        # Testa a conexão com o servidor (boa prática)
        client.admin.command('ping')
        
        # Seleciona o banco de dados (será criado na primeira inserção)
        db = client[db_name]
        
        # Seleciona a coleção (será criada na primeira inserção)
        collection = db[collection_name]
        
        # Insere os dados em lote (mais eficiente)
        result = collection.insert_many(data_to_insert)
        
        # Fecha a conexão com o banco de dados
        client.close()
        
        # Retorna a contagem de documentos inseridos
        return len(result.inserted_ids)

    except pymongo.errors.ConnectionFailure as e:
        # Erro de resiliência: falha na conexão
        st.error(f"Erro de conexão com o Cosmos DB: {e}")
        return -1 # Retorna -1 para indicar falha de conexão
    except Exception as e:
        # Erro de resiliência: falha geral (ex: permissão, formato de dados)
        st.error(f"Erro durante a inserção no Cosmos DB: {e}")
        return -1 # Retorna -1 para indicar falha geral

# 4. O Componente de Upload
uploaded_file = st.file_uploader(
    "Selecione seu arquivo:",
    type=["csv", "json"],
    help="Apenas arquivos .csv e .json são aceitos."
)

# 5. Lógica de Processamento e Envio
if uploaded_file is not None:
    
    with st.spinner('Processando o arquivo...'):
        dataframe = None
        
        # Lê os bytes do arquivo carregado
        file_bytes = uploaded_file.getvalue()

        try:
            # Tenta ler o CSV
            if uploaded_file.name.endswith('.csv'):
                dataframe = pl.read_csv(file_bytes)
            
            # Tenta ler o JSON
            elif uploaded_file.name.endswith('.json'):
                dataframe = pl.read_json(file_bytes)

        except Exception as e:
            st.error(f"Erro ao ler o arquivo '{uploaded_file.name}' com Polars: {e}")
            st.warning("Verifique se o formato do JSON é suportado (lista de objetos ou NDJSON).")
            dataframe = None

    # Se a leitura com Polars foi bem-sucedida
    if dataframe is not None:
        st.success(f"Arquivo '{uploaded_file.name}' lido com {dataframe.height:,} linhas.")
        
        # Exibe uma amostra dos dados
        st.dataframe(dataframe.head())
        
        st.subheader("Confirmação de Envio")
        st.write(f"Os dados serão enviados para a coleção **{MONGO_COLLECTION_NAME}** no banco de dados **{MONGO_DB_NAME}**.")

        # Adiciona um botão de confirmação para evitar envios duplicados
        # em re-execuções acidentais do Streamlit.
        if st.button("Confirmar e Enviar para o Cosmos DB"):
            with st.spinner("Conectando ao banco de dados e enviando dados..."):
                
                # Chama a função de ingestão
                inserted_count = send_to_cosmos_db(
                    dataframe, 
                    MONGO_URI, 
                    MONGO_DB_NAME, 
                    MONGO_COLLECTION_NAME
                )
                
                # Feedback baseado no resultado da ingestão
                if inserted_count > 0:
                    st.success(f"Sucesso! {inserted_count} registros inseridos no Cosmos DB.")
                    st.info("O contêiner de processamento downstream (escutando o Change Feed) deve ser ativado agora.")
                elif inserted_count == 0:
                    st.warning("Nenhum registro foi inserido (arquivo vazio).")
                else:
                    st.error("A ingestão de dados falhou. Verifique os logs de erro acima.")

else:
    st.info("Aguardando o upload de um arquivo para análise.")