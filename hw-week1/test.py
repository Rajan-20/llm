from openai import OpenAI
from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback
from ingest import load_lessons_data
from gitsource import chunk_documents
from minsearch import Index
import os
from dotenv import load_dotenv
load_dotenv()



documents = load_lessons_data()
chunks = chunk_documents(
    documents,
    size=2000,
    step=1000
)

chunk_index = Index(
    text_fields = ["content"],
    keyword_fields = ["filename"]
)
chunk_index.fit(chunks)

tools = Tools()

search_calls = 0

def search(query: str) -> dict[str, str]:
    """
    Search the course lessons and return relevant passages.
    """
    global search_calls
    search_calls += 1

    return chunk_index.search(
        query=query,
        num_results=5
    )

tools.add_tool(search)

chat_interface = IPythonChatInterface()
groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# # Create and run chat assistant
runner = OpenAIResponsesRunner(
    tools=tools,
    developer_prompt="You are a helpful weather assistant.",
    chat_interface=chat_interface,
    llm_client=OpenAIClient(
        client=groq_client,
        model="llama-3.3-70b-versatile"
    )
)

runner.run()