# -*- coding: utf-8 -*-
"""Data Analyst Agent Assignment with LangGraph, Memory, and Summarization"""

import streamlit as st
from datasets import load_dataset
import pandas as pd
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
import os
import json
import uuid
import operator

# --- Constants ---
DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
MODEL_NAME = "gpt-3.5-turbo"
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SUMMARY_MEMORY_LIMIT = 3  # Max key facts to remember about user

# --- Load dataset ---
@st.cache_data
def load_dataset_to_df() -> pd.DataFrame:
    dataset = load_dataset(DATASET_NAME, split="train")
    df = pd.DataFrame(dataset)
    df['category'] = df['category'].str.upper().str.strip()
    df['intent'] = df['intent'].str.lower().str.strip()
    return df[['instruction', 'response', 'category', 'intent']].dropna()

# --- Shared Tools ---
class ToolHandler:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    @tool
    def get_all_categories(self) -> List[str]:
        """List all unique categories in the dataset"""
        return sorted(self.df['category'].unique().tolist())
    
    @tool
    def get_all_intents(self) -> List[str]:
        """List all unique intents in the dataset"""
        return sorted(self.df['intent'].unique().tolist())
    
    @tool
    def count_category(self, category: str) -> int:
        """Count examples in a specific category"""
        return len(self.df[self.df['category'] == category.upper()])
    
    @tool
    def count_intent(self, intent: str) -> int:
        """Count examples with a specific intent"""
        return len(self.df[self.df['intent'] == intent.lower()])
    
    @tool
    def show_examples(self, n: int, category: Optional[str] = None, intent: Optional[str] = None) -> str:
        """Show examples with optional filters"""
        filtered = self.df.copy()
        if category:
            filtered = filtered[filtered['category'] == category.upper()]
        if intent:
            filtered = filtered[filtered['intent'] == intent.lower()]
        
        n = min(n, len(filtered))
        examples = filtered.sample(n).to_dict('records')
        return "\n\n".join(
            f"**Example {i+1}**\n**Customer**: {e['instruction']}\n**Agent**: {e['response']}"
            for i, e in enumerate(examples)
        )
    
    @tool
    def summarize(self, topic: str) -> str:
        """Generate summary of a category or intent"""
        # Filter relevant examples
        filtered = self.df.copy()
        if topic.upper() in self.df['category'].unique():
            filtered = filtered[filtered['category'] == topic.upper()]
        elif topic.lower() in self.df['intent'].unique():
            filtered = filtered[filtered['intent'] == topic.lower()]
        
        # Take sample for summarization
        examples = filtered.sample(min(5, len(filtered))).to_dict('records')
        text_examples = "\n\n".join(
            f"Customer: {e['instruction']}\nAgent: {e['response']}" 
            for e in examples
        )
        
        # Generate summary
        llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7, api_key=OPENAI_API_KEY)
        response = llm.invoke(
            f"Summarize key patterns from these customer service examples about {topic}:\n\n{text_examples}"
        )
        return response.content

# --- Graph State Definition ---
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    session_id: str
    user_summary: str  # Key facts about the user
    last_tool_results: List[str]

memory = MemorySaver()

# --- Classifier Node ---
def classify_query(state: GraphState):
    """Classify user query into structured, unstructured, or out-of-scope"""
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
    
    # Get last human message
    last_message = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1]
    
    system = """
    Classify the user query into one of:
    - structured: questions about counts, lists, examples, distributions
    - unstructured: questions requiring summarization or interpretation
    - out_of_scope: everything else
    
    Consider the conversation history and previous tool results.
    """
    
    messages = [
        ("system", system),
        *[(msg.type, msg.content) for msg in state["messages"] if msg.type in ["human", "ai"]],
        ("human", "Classify this query: " + last_message.content)
    ]
    
    response = llm.invoke(messages)
    classification = response.content.lower()
    
    if "structured" in classification:
        return {"query_type": "structured"}
    elif "unstructured" in classification:
        return {"query_type": "unstructured"}
    else:
        return {"query_type": "out_of_scope"}

# --- Structured Agent Node ---
def structured_agent(state: GraphState, config: RunnableConfig):
    """Handle structured queries with dedicated tools"""
    configurable = config.get("configurable", {})
    handler = configurable.get("handler")
    
    if not handler:
        # Handle missing handler gracefully
        state["messages"].append(AIMessage(content="System error: Tool handler not available"))
        return state
        
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
    
    # Structured tools
    tools = [
        handler.get_all_categories,
        handler.get_all_intents,
        handler.count_category,
        handler.count_intent,
        handler.show_examples,
    ]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Create agent
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    tool_calls = response.tool_calls
    
    # Execute tools
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]
        try:
            tool_func = next(t for t in tools if t.name == tool_name)
            result = tool_func.invoke(args)
            results.append(result)
            
            # Add tool message to state
            state["messages"].append(AIMessage(
                content=f"Tool {tool_name} called with args {args}. Result: {result}",
                tool_calls=[tool_call]
            ))
            state["messages"].append(AIMessage(
                content=f"Tool {tool_name} result: {result}",
                name=tool_name
            ))
        except StopIteration:
            # Handle unknown tool
            state["messages"].append(AIMessage(
                content=f"Unknown tool called: {tool_name}",
                name="error_handler"
            ))
    
    # Store results for follow-up queries
    state["last_tool_results"] = results
    return state

# --- Unstructured Agent Node ---
def unstructured_agent(state: GraphState, config: RunnableConfig):
    """Handle unstructured queries with summarization tools"""
    configurable = config.get("configurable", {})
    handler = configurable.get("handler")
    
    if not handler:
        # Handle missing handler gracefully
        state["messages"].append(AIMessage(content="System error: Tool handler not available"))
        return state
        
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
    
    # Unstructured tools
    tools = [handler.summarize]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Create agent
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    tool_calls = response.tool_calls
    
    # Execute tools
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]
        try:
            tool_func = next(t for t in tools if t.name == tool_name)
            result = tool_func.invoke(args)
            results.append(result)
            
            # Add tool message to state
            state["messages"].append(AIMessage(
                content=f"Tool {tool_name} called with args {args}. Result: {result}",
                tool_calls=[tool_call]
            ))
            state["messages"].append(AIMessage(
                content=f"Tool {tool_name} result: {result}",
                name=tool_name
            ))
        except StopIteration:
            # Handle unknown tool
            state["messages"].append(AIMessage(
                content=f"Unknown tool called: {tool_name}",
                name="error_handler"
            ))
    
    # Store results for follow-up queries
    state["last_tool_results"] = results
    return state

# --- Out-of-Scope Handler ---
def out_of_scope_handler(state: GraphState):
    """Handle out-of-scope queries"""
    state["messages"].append(AIMessage(
        content="I can only answer questions about customer service data. Please ask about the dataset."
    ))
    return state

# --- Summary Memory Node ---
def update_summary_memory(state: GraphState, config: RunnableConfig):
    """Update user summary with key information from conversation"""
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
    
    # Get conversation context
    conversation = "\n".join(
        f"{msg.type}: {msg.content}" 
        for msg in state["messages"] 
        if isinstance(msg, (HumanMessage, AIMessage))
    )
    
    prompt = f"""
    Extract key factual information about the user from this conversation.
    Focus on permanent characteristics, preferences, and important context.
    Limit to {SUMMARY_MEMORY_LIMIT} most important facts.
    Return as a bullet-point list.
    
    Current Summary:
    {state.get('user_summary', 'No information yet')}
    
    Conversation:
    {conversation}
    
    Updated Summary (bullet points only):
    """
    
    # Get updated summary
    new_summary = llm.invoke(prompt).content
    state["user_summary"] = new_summary
    
    # Add memory update to conversation
    state["messages"].append(AIMessage(
        content=f"Memory updated with key user facts",
        name="memory_updater"
    ))
    
    return state

# --- Recall Summary Tool ---
@tool
def recall_user_summary(user_summary: str) -> str:
    """Recall key facts about the user from memory"""
    return user_summary

# --- Build LangGraph Workflow ---
def build_workflow():
    """Create and compile LangGraph workflow with memory"""
    workflow = StateGraph(GraphState)
    
    # Define nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("structured_agent", structured_agent)
    workflow.add_node("unstructured_agent", unstructured_agent)
    workflow.add_node("out_of_scope", out_of_scope_handler)
    workflow.add_node("update_memory", update_summary_memory)
    
    # Set entry point
    workflow.set_entry_point("classify")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "classify",
        lambda state: state.get("query_type", "structured"),
        {
            "structured": "structured_agent",
            "unstructured": "unstructured_agent",
            "out_of_scope": "out_of_scope"
        }
    )
    
    # Add edges
    workflow.add_edge("structured_agent", "update_memory")
    workflow.add_edge("unstructured_agent", "update_memory")
    workflow.add_edge("out_of_scope", "update_memory")
    workflow.add_edge("update_memory", END)
    
    # Compile with checkpointing
    return workflow.compile(checkpointer=memory)


# --- Streamlit UI ---
def main():
    # Load data and tools
    df = load_dataset_to_df()
    handler = ToolHandler(df)
    
    # Build LangGraph workflow
    workflow = build_workflow()
    
    st.title("📊 Customer Support Data Analyst (LangGraph + Memory)")
    st.write(f"Dataset stats: {len(df)} rows, {df['intent'].nunique()} intents, {df['category'].nunique()}")
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []  # Initialize messages here
    
    # Session ID input
    session_id = st.sidebar.text_input("Session ID", value=st.session_state.session_id)
    if session_id != st.session_state.session_id:
        st.session_state.session_id = session_id
        st.session_state.messages = []  # Reset messages on session change
        st.experimental_rerun()
    
    st.sidebar.write(f"Active Session: `{session_id}`")
    
    # Memory display
    if st.sidebar.button("Show My Memory"):
        # Retrieve memory from current state
        config = RunnableConfig(
            configurable={
                "thread_id": session_id
            }
        )
        try:
            # Get current state from checkpoint
            current_state = memory.get(config)
            if current_state and "user_summary" in current_state:
                st.sidebar.subheader("Your Memory Summary")
                st.sidebar.markdown(current_state["user_summary"])
            else:
                st.sidebar.info("No memory stored yet")
        except Exception:
            st.sidebar.warning("Couldn't retrieve memory")
    
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])
    
    # User input
    if prompt := st.chat_input("Ask about the dataset..."):
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        
        with st.spinner("Analyzing..."):
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=prompt)],
                "session_id": session_id,
                "last_tool_results": [],
                "user_summary": ""
            }
            
            # CORRECTED CONFIG (flat structure)
            config = RunnableConfig(
                configurable={
                    "handler": handler,
                    "thread_id": session_id
                }
            )
            
            # Run with checkpointing
            for step in workflow.stream(initial_state, config):
                if "__end__" in step:
                    final_state = step["__end__"]
                    
                    # Extract final response
                    last_ai_message = next(
                        (msg for msg in reversed(final_state["messages"]) 
                         if isinstance(msg, AIMessage) and not msg.tool_calls),
                        None
                    )
                    
                    if last_ai_message:
                        response = last_ai_message.content
                    else:
                        response = "I couldn't generate a response. Please try again."
                    
                    # Add to UI messages
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.chat_message("assistant").markdown(response)

if __name__ == "__main__":
    main()
