import streamlit as st
from datasets import load_dataset
import pandas as pd
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from typing import Dict
import uuid
import operator

MODEL_NAME = "gpt-3.5-turbo"
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
SUMMARY_MEMORY_LIMIT = 3
memory = MemorySaver()

@st.cache_data
def load_dataset_to_df() -> pd.DataFrame:
    dataset = load_dataset(DATASET_NAME, split="train")
    df = pd.DataFrame(dataset)
    df['category'] = df['category'].str.upper().str.strip()
    df['intent'] = df['intent'].str.lower().str.strip()
    return df[['instruction', 'response', 'category', 'intent']].dropna()

def make_tools(df: pd.DataFrame):
    @tool
    def get_all_categories() -> List[str]:
        """Return all unique categories in the dataset."""
        return sorted(df['category'].unique().tolist())

    @tool
    def get_all_intents() -> List[str]:
        """Return all unique intents in the dataset."""
        return sorted(df['intent'].unique().tolist())

    @tool
    def count_category(category: str) -> int:
        """Counts the number of examples in each category and returns the result."""
        return len(df[df['category'] == category.upper()])

    @tool
    def count_intent(intent: str) -> int:
        """Counts the number of examples in each intent and returns the result."""
        return len(df[df['intent'] == intent.lower()])

    @tool
    def get_intent_distribution() -> Dict[str, int]:
        """Show intent distributions."""
        return df['intent'].value_counts().to_dict()

    @tool
    def get_top_categories(n: int = 5) -> List[str]:
        """Return the top N most frequent categories."""
        return df['category'].value_counts().head(n).index.tolist()

    @tool
    def show_examples(n: int, category: Optional[str] = None, intent: Optional[str] = None) -> str:
        """
        Show up to `n` random conversation examples from the dataset, optionally filtered by category and/or intent.
        Args:
            n: Number of examples to show.
            category: Optional category to filter examples (case-insensitive).
            intent: Optional intent to filter examples (case-insensitive).
        Returns:
            A markdown-formatted string with the selected customer-agent examples.
        """
        filtered = df.copy()
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
    def summarize(topic: str) -> str:
        """
        Provide a brief summary by sampling up to 5 examples from a given category or intent.

        Args:
            topic: A category (uppercase) or intent (lowercase) to summarize.

        Returns:
            A text snippet showing customer-agent interactions for the selected topic.
        """
        filtered = df.copy()
        if topic.upper() in df['category'].unique():
            filtered = filtered[filtered['category'] == topic.upper()]
        elif topic.lower() in df['intent'].unique():
            filtered = filtered[filtered['intent'] == topic.lower()]
        examples = filtered.sample(min(5, len(filtered))).to_dict('records')
        text_examples = "\n\n".join(
            f"Customer: {e['instruction']}\nAgent: {e['response']}"
            for e in examples
        )
        llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7, api_key=OPENAI_API_KEY)
        response = llm.invoke(f"Summarize key patterns from these customer service examples about {topic}:\n\n{text_examples}")
        return response.content

    return [get_all_categories, get_all_intents, get_intent_distribution, count_category, count_intent, get_top_categories, show_examples, summarize]

class GraphState(dict):
    messages: List[BaseMessage]
    session_id: str
    user_summary: str
    last_tool_results: List[str]
    final_response: str

def classify_query(state: GraphState):
    state = state.copy()  # <--- CRITICAL: avoid mutating input directly

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
    last_message = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1]
    system = """
    Classify the user query into one of:
    - structured
    - unstructured
    - out_of_scope
    """
    messages = [("system", system), *[(msg.type, msg.content) for msg in state["messages"]], ("human", "Classify this query: " + last_message.content)]
    response = llm.invoke(messages)
    classification = response.content.lower().strip()

    if "structured" in classification:
        state["query_type"] = "structured"
    elif "unstructured" in classification:
        state["query_type"] = "unstructured"
    else:
        state["query_type"] = "out_of_scope"

    return state  # now safe with return_dict=False



    
def generate_final_response(state: GraphState): 
    """Generate the final assistant response from messages or tool results."""
    if state.get("last_tool_results"):
        result = state["last_tool_results"][-1]
        if isinstance(result, list):
            content = "The most frequent categories are: " + ", ".join(result)
        elif isinstance(result, dict):
            # Format dicts (e.g., intent distribution)
            lines = [f"{k}: {v}" for k, v in result.items()]
            content = "Intent Distribution:\n\n" + "\n".join(lines)
        else:
            content = str(result)
    else:
        last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        content = last_ai.content if last_ai else "No response generated."

    state["final_response"] = content
    state["messages"].append(AIMessage(content=content))
    return state

def structured_agent(state: GraphState, config: RunnableConfig):
    print(">>> Entered structured_agent", flush=True)
    print("State:", state, flush=True)
    tools = config.get("configurable", {}).get("tools", [])
    if not tools:
        state["messages"].append(AIMessage(content="System error: tools not found"))
        return state
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    tool_calls = response.tool_calls
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]
        try:
            tool_func = next(t for t in tools if t.name == tool_name)
            print(f"Invoking tool: {tool_name} with args {args}", flush=True)
            result = tool_func.invoke(args)
            print(f"Tool result: {result}", flush=True)
            results.append(result)
            state["messages"].append(AIMessage(content=f"Tool {tool_name} result: {result}"))
        except StopIteration:
            state["messages"].append(AIMessage(content=f"Unknown tool: {tool_name}"))
    state["last_tool_results"] = results
    return state

def unstructured_agent(state: GraphState, config: RunnableConfig):
    return structured_agent(state, config)

def out_of_scope_handler(state: GraphState):
    state["messages"].append(AIMessage(content="I can only answer questions about customer service data."))
    return state

def update_summary_memory(state: GraphState, config: RunnableConfig):
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
    conversation = "\n".join(f"{msg.type}: {msg.content}" for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage)))
    prompt = f"""
    Extract key facts about the user. Limit to {SUMMARY_MEMORY_LIMIT} items.
    Current Summary: {state.get('user_summary', '')}
    Conversation: {conversation}
    Updated Summary:
    """
    new_summary = llm.invoke(prompt).content
    state["user_summary"] = new_summary
    return state

def build_workflow():
    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("structured_agent", structured_agent)
    workflow.add_node("unstructured_agent", unstructured_agent)
    workflow.add_node("out_of_scope", out_of_scope_handler)
    workflow.add_node("update_memory", update_summary_memory)
    workflow.add_node("generate_final_response", generate_final_response)

    # Set entry point and explicitly add START edge
    workflow.set_entry_point("classify")
    workflow.add_edge(START, "classify") 

    # Define conditional routing function
    def route_from_classify(state: GraphState) -> str:
        return state["query_type"]

    # Add conditional branching
    workflow.add_conditional_edges(
        "classify",
        route_from_classify,
        {
            "structured": "structured_agent",
            "unstructured": "unstructured_agent",
            "out_of_scope": "generate_final_response",
        }
    )

    # Transitions
    workflow.add_edge("structured_agent", "update_memory")
    workflow.add_edge("unstructured_agent", "update_memory")
    workflow.add_edge("update_memory", "generate_final_response")
    workflow.add_edge("generate_final_response", END)

    return workflow.compile(checkpointer=memory)

def main():
    df = load_dataset_to_df()
    tools = make_tools(df)
    workflow = build_workflow()
    st.title("📊 Customer Support Data Analyst")
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
    session_id = st.sidebar.text_input("Session ID", value=st.session_state.session_id)
    if session_id != st.session_state.session_id:
        st.session_state.session_id = session_id
        st.session_state.messages = []
        st.experimental_rerun()
    st.sidebar.write(f"Active Session: `{session_id}`")
    if st.sidebar.button("Show My Memory"):
        config = RunnableConfig(configurable={"thread_id": session_id})
        try:
            state = memory.get(config)
            if state and "user_summary" in state:
                st.sidebar.subheader("Your Memory Summary")
                st.sidebar.markdown(state["user_summary"])
            else:
                st.sidebar.info("No memory stored yet")
        except Exception:
            st.sidebar.warning("Couldn't retrieve memory")
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])
    if prompt := st.chat_input("Ask about the dataset..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        with st.spinner("Analyzing..."):
            initial_state = {
                "messages": [HumanMessage(content=prompt)],
                "session_id": session_id,
                "last_tool_results": [],
                "user_summary": ""
            }
            config = RunnableConfig(configurable={"tools": tools, "thread_id": session_id})
            for step in workflow.stream(initial_state, config):
                print(">>> Step output:", step, flush=True)
                if "generate_final_response" in step:
                    final_state = step["generate_final_response"]
                    response = final_state.get("final_response", "No response generated.")
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    print(f"FINAL RESPONSE TO DISPLAY: {response}")
                    st.chat_message("assistant").markdown(response)
                    

if __name__ == "__main__":
    main()
