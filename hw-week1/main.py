from ingest import load_lessons_data, build_index
from openai import OpenAI
from dotenv import load_dotenv
from rag_helper import RAGBase
import os

groq_base_url = "https://api.groq.com/openai/v1"
load_dotenv()


def main():
    documents = load_lessons_data()
    index = build_index(documents)

    openai_client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url=groq_base_url
    )

    assistant = RAGBase(
        index=index,
        llm_client=openai_client,
    )

    answer,usage = assistant.rag("How does the agentic loop keep calling the model until it stops?")
    print(answer, usage)


if __name__ == "__main__":
    main()
