from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
import os
# -----------------------------------------------------------------------------
# Databricks Embedding Model
# -----------------------------------------------------------------------------
import requests
class OpenAI_Databricks_Embedding():
    """Databricks embedding endpoint wrapped with LlamaIndex OpenAIEmbedding."""

    def __init__(self):
        load_dotenv()

        self.db_host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
        self.sp_client_id = os.getenv("DATABRICKS_CLIENT_ID", "")
        self.sp_client_secret = os.getenv("DATABRICKS_CLIENT_SECRET", "")
        self.endpoint_name = os.getenv("DATABRICKS_EMBEDDING_ENDPOINT", "")
        self.endpoint_type = os.getenv("DATABRICKS_EMBEDDING_TYPE", "text-embedding-3-large")

        if not self.db_host or not self.sp_client_id or not self.sp_client_secret or not self.endpoint_name:
            raise RuntimeError(
                "Missing env vars: DATABRICKS_HOST, SP_CLIENT_ID, "
                "SP_CLIENT_SECRET, DATABRICKS_EMBEDDING_ENDPOINT"
            )

        access_token = self.get_oauth_token()

        # model must be a valid OpenAI embedding model name
        # model_name overrides the actual engine used by LlamaIndex
        self.embedding = OpenAIEmbedding(
            model=self.endpoint_type,
            model_name=self.endpoint_name,
            api_key=access_token,
            api_base=f"{self.db_host}/serving-endpoints",
        )

    def get_oauth_token(self) -> str:
        token_url = f"{self.db_host}/oidc/v1/token"

        resp = requests.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.sp_client_id,
                "client_secret": self.sp_client_secret,
                "scope": "all-apis",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    def as_llama_embedding(self):
        return self.embedding