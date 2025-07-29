import streamlit as st
from datasets import load_dataset
import pandas as pd
from typing import TypedDict, Dict, Any, Tuple, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from typing import Dict
import uuid
import operator
import re
import os

MODEL_NAME = "gpt-3.5-turbo"
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
SUMMARY_MEMORY_LIMIT = 3
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "CustomerSupportLangGraph"
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
        print(f"🚨 INSIDE TOOL — category={category}, intent={intent}")
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


class GraphState(TypedDict):
    values: Dict[str, Any]
    next: Tuple[str]
    messages: List[BaseMessage]
    session_id: str
    user_summary: str
    last_tool_results: List[str]
    final_response: str
    thread_id: str
    query_type: Optional[str]
    last_category: Optional[str]  # New: track last used category for follow-ups
    last_intent: Optional[str]    # New: track last used intent for follow-ups

def classify_query(state: GraphState):
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
    
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not human_messages:
        state["query_type"] = "out_of_scope"
        return state
    
    last_message = human_messages[-1]
    content = last_message.content.lower()
    
    # Handle follow-up patterns
    if re.search(r"\b(more|another|additional)\b", content):
        if state.get("last_category") or state.get("last_intent"):
            state["query_type"] = "structured"
            return state
    
    if "summarize" in content:
        state["query_type"] = "unstructured"
        return state
        
    # Then check for structured patterns
    structured_keywords = [
        "frequent", "examples", "categories", "distributions", 
        "count", "show", "what", "how many", "list", "intent"
    ]
    if any(keyword in content for keyword in structured_keywords):
        state["query_type"] = "structured"
        return state
    
    # If no patterns match, use LLM for classification
    system = """
    Classify the user query into one of:
    - structured: Questions about counts, distributions, or specific examples
    - unstructured: Requests for summaries
    - out_of_scope: Anything else
    
    Examples:
    - "What are the most frequent categories?" → structured
    - "Show examples of Category X" → structured
    - "Summarize Category X" → unstructured
    - "Who is Magnus Carlson?" → out_of_scope
    """
    
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Classify this query: {last_message.content}")
    ]
    
    response = llm.invoke(messages)
    classification = response.content.lower().strip()
    
    # Set query_type based on classification
    if "structured" in classification:
        state["query_type"] = "structured"
    elif "unstructured" in classification:
        state["query_type"] = "unstructured"
    else:
        state["query_type"] = "out_of_scope"
    
    print(f">>> Classification result: {state['query_type']}")
    return state

def generate_final_response(state: GraphState): 
    # Handle out-of-scope queries
    if state.get("query_type") == "out_of_scope":
        content = "I can only answer questions about customer service data. Please ask about the dataset, categories, intents, or summaries."
    
    # Handle structured/unstructured queries
    elif state.get("last_tool_results"):
        result = state["last_tool_results"][-1]
        if isinstance(result, list):
            content = "The most frequent categories are: " + ", ".join(result)
        elif isinstance(result, dict):
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
    tools = config.get("configurable", {}).get("tools", [])
    if not tools:
        state["messages"].append(AIMessage(content="System error: tools not found"))
        return state

    # Filter tools for structured queries
    structured_tools = [t for t in tools if t.name != "summarize"]
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)

    # Detect if this is a follow-up query
    last_user_msg = state["messages"][-1].content.lower()
    is_follow_up = bool(re.search(r"\b(more|another|additional)\b", last_user_msg))

    # Add system prompt for structured queries
    structured_prompt = SystemMessage(
        content="You are a data analyst for customer support queries. "
                "Answer structured questions about categories, intents, and examples. "
                "Use available tools to get precise data."
    )
    
    # Inject last used context if available and not a fresh query
    context_summary = []
    if is_follow_up: # Only inject context if it's a follow-up
        if state.get("last_category"):
            context_summary.append(f"Category: {state['last_category']}")
        if state.get("last_intent"):
            context_summary.append(f"Intent: {state['last_intent']}")
    
    messages = [structured_prompt]
    if context_summary:
        messages.append(HumanMessage(content="Previous context:\n" + "\n".join(context_summary)))
    messages.extend([msg for msg in state["messages"] if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage)])


    # Bind tools and invoke LLM with tool call
    llm_with_tools = llm.bind_tools(structured_tools)
    response = llm_with_tools.invoke(messages)
    tool_calls = response.tool_calls
    results = []

    # Reset last_category and last_intent before potential new assignments
    # This prevents carrying over context from a previous, unrelated structured query
    if not is_follow_up:
        state["last_category"] = None
        state["last_intent"] = None

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]
        try:
            tool_func = next(t for t in structured_tools if t.name == tool_name)
            print(f"Invoking tool: {tool_name} with args {args}", flush=True)

            # Patch missing follow-up context before execution for show_examples
            if tool_name == "show_examples":
                if (not args.get("category") or args["category"] is None) and state.get("last_category"):
                    args["category"] = state["last_category"]
                    print(f"✅ Using fallback category: {state['last_category']}")
                if (not args.get("intent") or args["intent"] is None) and state.get("last_intent"):
                    args["intent"] = state["last_intent"]
                    print(f"✅ Using fallback intent: {state['last_intent']}")
                print(f"🧪 Final args to show_examples: {args}")

                # Optional: increase count on follow-up
                if is_follow_up:
                    args["n"] = min(args.get("n", 3) + 2, 10)
                    print(f"⬆️ Increased example count to: {args['n']}")
            
            # Store new context *after* potential patching, but before tool execution
            # This ensures that the context being stored is the one actually used by the tool
            if tool_name == "show_examples":
                if "category" in args and args["category"] is not None:
                    state["last_category"] = args["category"]
                if "intent" in args and args["intent"] is not None:
                    state["last_intent"] = args["intent"]
                print(f"✅ Stored context: category={state.get('last_category')}, intent={state.get('last_intent')}", flush=True)


            # Run the tool
            result = tool_func.invoke(args)
            print(f"🛠️ Tool result: {result}", flush=True)

            results.append(result)
            state["messages"].append(AIMessage(content=f"Tool {tool_name} result: {result}"))
        except StopIteration:
            state["messages"].append(AIMessage(content=f"Unknown tool: {tool_name}"))
        except Exception as e:
            state["messages"].append(AIMessage(content=f"Error executing tool {tool_name}: {e}"))
            print(f"Error during tool execution: {e}")

    state["last_tool_results"] = results
    print(f"✅ Finished structured_agent", flush=True)
    return state


def unstructured_agent(state: GraphState, config: RunnableConfig):
    print(">>> Entered unstructured_agent", flush=True)
    tools = config.get("configurable", {}).get("tools", [])
    if not tools:
        state["messages"].append(AIMessage(content="System error: tools not found"))
        return state
    
    # Focus only on summarize tool for unstructured queries
    summarize_tool = next((t for t in tools if t.name == "summarize"), None)
    if not summarize_tool:
        state["messages"].append(AIMessage(content="System error: summarize tool not found"))
        return state
    
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
    
    # Add specialized prompt for summarization
    summarize_prompt = SystemMessage(
        content="You are a summarization specialist. Your task is to generate concise summaries "
                "of customer service interactions. Use the summarize tool to extract key patterns "
                "from the dataset."
    )
    
    # Create new messages with system prompt
    messages = [summarize_prompt] + state["messages"]
    
    llm_with_tool = llm.bind_tools([summarize_tool])
    response = llm_with_tool.invoke(messages)
    tool_calls = response.tool_calls
    results = []
    
    # For unstructured queries, we typically don't carry category/intent context forward.
    # So, explicitly reset them here if this path is taken.
    state["last_category"] = None
    state["last_intent"] = None

    for tool_call in tool_calls:
        if tool_call["name"] == "summarize":
            args = tool_call["args"]
            print(f"Invoking summarize tool with args {args}", flush=True)
            result = summarize_tool.invoke(args)
            print(f"Summarize result: {result}", flush=True)
            results.append(result)
            state["messages"].append(AIMessage(content=f"Summary: {result}"))
        else:
            state["messages"].append(AIMessage(content="I can only generate summaries for unstructured queries"))
    
    state["last_tool_results"] = results
    return state

def out_of_scope_handler(state: GraphState):
    state["messages"].append(AIMessage(content="I can only answer questions about customer service data."))
    state["last_category"] = None # Reset context for out-of-scope
    state["last_intent"] = None
    return state

def update_summary_memory(state: GraphState, config: RunnableConfig):
    # Keep conversation window manageable
    state["messages"] = state["messages"][-10:]
    
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
    conversation = "\n".join(
        f"{msg.type}: {msg.content}" 
        for msg in state["messages"] 
        if isinstance(msg, (HumanMessage, AIMessage))
    )
    prompt = f"""
    Extract key facts about the user. Limit to {SUMMARY_MEMORY_LIMIT} items.
    Current Summary: {state.get('user_summary', '')}
    Conversation: {conversation}
    Updated Summary:
    """
    new_summary = llm.invoke(prompt).content
    state["user_summary"] = new_summary
    return state

def store_context(state: GraphState):
    # This node primarily ensures that the state changes (like last_category/intent)
    # are captured and persisted by the LangGraph checkpoint after structured_agent.
    print(f"✅ Storing context: category={state.get('last_category')}, intent={state.get('last_intent')}")
    return state


def build_workflow():
    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("structured_agent", structured_agent)
    workflow.add_node("unstructured_agent", unstructured_agent)
    workflow.add_node("out_of_scope", out_of_scope_handler)
    workflow.add_node("update_memory", update_summary_memory)
    workflow.add_node("store_context", store_context)
    workflow.add_node("generate_final_response", generate_final_response)

    # Set entry point and explicitly add START edge
    workflow.set_entry_point("classify")
    workflow.add_edge(START, "classify")
    
    def route_from_classify(state: dict) -> str:
        # Access query_type directly from the state dictionary
        if "query_type" not in state:
            print("⚠️ Warning: query_type not found! Defaulting to out_of_scope")
            return "out_of_scope"
        print(f">>> Routing to: {state['query_type']}")
        
        # Route to the appropriate agent or directly to final response for out_of_scope
        if state["query_type"] == "structured":
            return "structured_agent"
        elif state["query_type"] == "unstructured":
            return "unstructured_agent"
        else: # out_of_scope
            return "out_of_scope"

    # Add conditional branching
    workflow.add_conditional_edges(
        "classify",
        route_from_classify,
        {
            "structured": "structured_agent",
            "unstructured": "unstructured_agent",
            "out_of_scope": "generate_final_response", # Directly go to final response for OOS
        }
    )

    # Transitions
    workflow.add_edge("structured_agent", "store_context")
    workflow.add_edge("store_context", "update_memory")
    workflow.add_edge("unstructured_agent", "update_memory")
    workflow.add_edge("out_of_scope", "generate_final_response") # Route from out_of_scope
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
        # Initialize last_category and last_intent in session_state as well
        st.session_state.messages = []
        st.session_state.last_category = None
        st.session_state.last_intent = None
    
    session_id = st.sidebar.text_input("Session ID", value=st.session_state.session_id)
    
    if session_id != st.session_state.session_id:
        st.session_state.session_id = session_id
        st.session_state.messages = []
        st.session_state.last_category = None # Reset when session ID changes
        st.session_state.last_intent = None # Reset when session ID changes
        st.experimental_rerun()
    
    st.sidebar.write(f"Active Session: `{session_id}`")
    
    if st.sidebar.button("Show My Memory"):
        config = RunnableConfig(configurable={
            "thread_id": session_id,
            "checkpoint_ns": "main"
        })
        try:
            state = memory.get(config)
            if state: # Check if state is not None
                st.sidebar.subheader("Your Memory Summary")
                if "user_summary" in state and state["user_summary"]:
                    st.sidebar.markdown(state["user_summary"])
                else:
                    st.sidebar.info("No user summary yet.")

                # Display last_category and last_intent from the retrieved state
                if "last_category" in state and state["last_category"]:
                    st.sidebar.write(f"Last category: {state['last_category']}")
                if "last_intent" in state and state["last_intent"]:
                    st.sidebar.write(f"Last intent: {state['last_intent']}")
                
                if not state.get("user_summary") and not state.get("last_category") and not state.get("last_intent"):
                    st.sidebar.info("No memory stored yet.")
            else:
                st.sidebar.info("No memory stored yet.")
        except Exception as e:
            st.sidebar.warning(f"Couldn't retrieve memory: {e}")
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])
    
    if prompt := st.chat_input("Ask about the dataset..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        
        with st.spinner("Analyzing..."):
            config = RunnableConfig(configurable={
                "tools": tools,
                "thread_id": session_id,
                "checkpoint_ns": "main"
            })
            
            # Retrieve the full state from memory for the current session
            current_state_from_memory = memory.get(config)

            # Initialize current_state for this turn
            # Prioritize state from memory, otherwise use a default structure
            if current_state_from_memory:
                current_state = current_state_from_memory.copy()
            else:
                current_state = {
                    "values": {},
                    "next": ("classify",), # Always start with classify
                    "messages": [], # Messages will be appended below
                    "session_id": session_id,
                    "last_tool_results": [],
                    "user_summary": "",
                    "query_type": None,
                    "thread_id": session_id,
                    "final_response": "",
                    "last_category": None, # Ensure these are initialized as None if no memory
                    "last_intent": None
                }
            
            # Ensure messages from Streamlit session state are in the LangGraph state
            # This is crucial for the LLM to see the full conversation history for classification
            current_state["messages"] = [
                HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
                for msg in st.session_state.messages
            ]
            
            # Add new user message to state (it's already in st.session_state.messages)
            # current_state["messages"].append(HumanMessage(content=prompt)) # This is now handled by the above list comprehension

            # Process workflow
            final_response = None
            final_state = None
            try:
                for output in workflow.stream(current_state, config):
                    for node, state in output.items():
                        final_state = state  # Track the last updated state
                        if node == "generate_final_response":
                            final_response = state.get("final_response")
                
                # After the workflow completes, update Streamlit's session_state
                # with the latest values from the final_state of the graph run.
                if final_state:
                    st.session_state.last_category = final_state.get("last_category")
                    st.session_state.last_intent = final_state.get("last_intent")
                    st.session_state.user_summary = final_state.get("user_summary") # Also update user_summary

            except Exception as e:
                error_msg = f"An error occurred during processing: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.chat_message("assistant").markdown(error_msg)
                print(f"Workflow execution error: {e}")
                final_response = None # Ensure no partial response is displayed

            # Display response
            if final_response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response
                })
                st.chat_message("assistant").markdown(final_response)
            elif not st.session_state.messages or (st.session_state.messages[-1]["role"] != "assistant"):
                # Fallback if no specific final_response but no error
                # This could happen if the graph ends without hitting generate_final_response cleanly
                fallback_msg = "I'm not sure how to respond to that. Can you rephrase?"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": fallback_msg
                })
                st.chat_message("assistant").markdown(fallback_msg)


if __name__ == "__main__":
    main()
