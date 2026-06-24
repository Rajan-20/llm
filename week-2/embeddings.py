from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from ingest import load_faq_data
import numpy as np
from rag_helper import RAGBase
from openai import OpenAI
from dotenv import load_dotenv
from minsearch import VectorSearch
import os


model = SentenceTransformer("all-MiniLM-L6-v2")
documents = load_faq_data()


texts = []

for doc in documents:
    text = doc["question"] + " " + doc["answer"]
    texts.append(text)

batch_size = 50
vectors = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    batch_vectors = model.encode(batch)
    vectors.extend(batch_vectors)



X = np.array(vectors)


class RAGVector(RAGBase):

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}

        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict
        )


# In[90]:



groq_base_url = "https://api.groq.com/openai/v1"
load_dotenv()

openai_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=groq_base_url
)


vindex = VectorSearch(keyword_fields=["course"])
vindex.fit(X, documents)


vector_assistant = RAGVector(
    embedder=model,
    index=vindex,
    llm_client=openai_client,
)


vector_assistant.rag("the program has already begun, can I still sign up?")