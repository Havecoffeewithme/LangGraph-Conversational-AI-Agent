from typing import TypedDict, List 
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END 
from dotenv import load_dotenv 
from langchain.chat_models import init_chat_model
import os 


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found!")

print("API key successfully loaded!")


class AgentState(TypedDict): 
    messages : List[HumanMessage]


model = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai",
    temperature=0
)

#llm = ChatOpenAI(model="gpt-4o-mini")


def process(state: AgentState) -> AgentState:
    response = model.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")


