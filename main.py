from typing import List, Sequence

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import generate_chain, reflect_chain


REFLECT = "reflect"
GENERATE = "generate"


load_dotenv()


# DEFINE NODES
def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]  # Reframe the role to be the one of a human


# DEFINE GRAPH
builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


# DEFINE CONDITIONAL EDGE
def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(
    GENERATE,           # Starting node
    should_continue     # Condition function
)

# CONNECT REFLECT TO GENERATE
builder.add_edge(REFLECT, GENERATE)

# COMPILE THE GRAPH
graph = builder.compile()
print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = HumanMessage(content="""Make the following tweet better:"
                                    Goodbye, vanilla RAG.

                                    Hello, Agentic RAG!

                                    𝗩𝗮𝗻𝗶𝗹𝗹𝗮 𝗥𝗔𝗚
                                    The common vanilla RAG implementation processed the user query through a retrieval and generation pipeline to generate a response grounded in external knowledge.
                          """)
    response = graph.invoke(inputs)
