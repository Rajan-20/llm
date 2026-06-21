import os
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from ingest import load_lessons_data
from gitsource import chunk_documents
from minsearch import Index



load_dotenv()

INSTRUCTIONS = """
You're a course teaching assistant.

You're given a question from a course student and your task is to answer it.

If you want to look up information, use the search function.

Use as many keywords from the user question as possible when making first requests.

Make multiple searches.

Try to expand your search by using new keywords
based on the results you get from the search.

At the end, ask if there are other areas that the user wants to explore.
""".strip()



print("Loading lessons...")

documents = load_lessons_data()

chunks = chunk_documents(
    documents,
    size=2000,
    step=1000
)

chunk_index = Index(
    text_fields=["content"],
    keyword_fields=["filename"]
)

chunk_index.fit(chunks)

print(f"Indexed {len(chunks)} chunks")




@dataclass
class AppDeps:
    index: Index
    search_calls: int = 0


deps = AppDeps(index=chunk_index)




agent = Agent(
    # model="groq:openai/gpt-oss-120b", //token limit exceeded
    model="groq:llama-3.3-70b-versatile",
    system_prompt=INSTRUCTIONS,
    deps_type=AppDeps,
)



@agent.tool
def search(ctx: RunContext[AppDeps], query: str) -> str:
    """
    Search the course lessons and return relevant passages.
    """

    ctx.deps.search_calls += 1

    results = ctx.deps.index.search(
        query=query,
        num_results=3 
    )

    if not results:
        return "No relevant results found."

    passages = []

    for r in results:
        filename = r.get("filename", "unknown")
        content = r.get("content", "")

        passages.append(
            f"FILE: {filename}\n\n{content}"
        )

    return "\n\n--------------------\n\n".join(passages)




def main():
    print("\nCourse Assistant Ready")
    print("Type 'exit' to quit.\n")

    message_history = None

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        try:
            result = agent.run_sync(
                question,
                deps=deps,
                message_history=message_history,
            )

            print("\nAssistant:\n")
            print(result.output)

            if message_history is None:
                message_history = result.new_messages()
            else:
                message_history.extend(result.new_messages())

            print(
                f"\n[Search calls so far: {deps.search_calls}]"
            )

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()