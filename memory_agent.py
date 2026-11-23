from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END 
from dotenv import load_dotenv 
from langchain.chat_models import init_chat_model
import os 


load_dotenv()


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


model = init_chat_model(
    "gpt-4o-min",
    model_provider="openai",
    temperture=0
)


def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = model.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")

    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


conversation_history = []


user_input = input("Enter: ")

while user_input != "exit" or "EXIT":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})

    print(result["messages"])

    conversation_history = result["messages"]

    user_input = input("Enter: ")






