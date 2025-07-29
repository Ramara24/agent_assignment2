# Customer Support Data Analyst (LangGraph + Memory)

An intelligent chatbot that analyzes customer support datasets using LangGraph workflows with persistent memory and advanced query processing.

## 🚀 Features

### 1. **Multi-Agent Query Processing**
- **Structured Agent**: Handles data queries (examples, counts, distributions)
- **Unstructured Agent**: Provides AI-powered summaries of dataset patterns
- **Memory Agent**: Retrieves user interaction history and preferences
- **Out-of-Scope Handler**: Politely redirects non-dataset queries

### 2. **Advanced Dataset Tools**
- **Data Exploration**: Categories, intents, examples with filtering
- **Statistical Analysis**: Counts, distributions, top categories
- **Mathematical Operations**: Calculate sums, totals of intent counts
- **Smart Examples**: Context-aware example generation with category/intent filtering

### 3. **Intelligent Memory System**
- **User Profile Tracking**: Learns your query patterns and preferences
- **Session Persistence**: Maintains context across browser reloads
- **Memory Queries**: Ask "What do you remember about me?" anytime
- **Smart Summarization**: AI-powered memory updates after each interaction

### 4. **Context-Aware Follow-ups**
- **Natural Continuations**: "Show me more examples" works intelligently
- **Category/Intent Persistence**: Remembers your last queried topics
- **Automatic Context Injection**: Follow-up queries use previous context
- **Increased Results**: Follow-up requests show more examples

### 5. **Robust Query Classification**
- **Typo Handling**: Recognizes "remmeber" as "remember"
- **Pattern Matching**: Smart detection of follow-ups, memory queries, summaries
- **LLM Fallback**: Uses GPT for ambiguous query classification
- **Out-of-Scope Detection**: Identifies non-dataset questions

### 6. **Session Management**
- **Persistent Sessions**: Resume conversations after page reload
- **Session ID Control**: Change sessions via sidebar input
- **Memory Visualization**: "Show My Memory" button displays your profile
- **Thread-based Storage**: Each session maintains separate context

## 🛠 Technical Architecture

### State Management
```python
class GraphState(TypedDict):
    messages: List[BaseMessage]          # Conversation history
    user_summary: str                    # User behavior profile
    last_category: str                   # Context for follow-ups
    last_intent: str                     # Intent context
    query_type: str                      # classified query type
    final_response: str                  # Generated response
```

### Workflow Nodes
1. **Classify** → Determines query type (structured/unstructured/memory/out_of_scope)
2. **Agent Processing** → Routes to appropriate specialized agent
3. **Context Storage** → Saves category/intent for follow-ups
4. **Memory Summarization** → Updates user profile intelligently
5. **Response Generation** → Creates final user-facing response

### Memory Intelligence
- **Behavior Analysis**: Tracks what categories you ask about most
- **Preference Learning**: Notes if you prefer examples vs summaries
- **Pattern Recognition**: Identifies follow-up question habits
- **Context Awareness**: Understands your interaction style

## 📊 Dataset

Uses the **Bitext Customer Support Dataset** containing:
- **26,872 customer-agent conversations**
- **27 categories** (ORDER, PAYMENT, ACCOUNT, etc.)
- **60+ intents** (cancel_order, check_invoice, etc.)
- **Real customer service interactions** for authentic responses

## 🔧 Available Tools

### Data Exploration
- `get_all_categories()` - List all available categories
- `get_all_intents()` - List all available intents
- `show_examples(n, category, intent)` - Get filtered examples
- `get_top_categories(n)` - Most frequent categories

### Analytics
- `get_intent_distribution()` - Complete intent frequency distribution
- `count_category(category)` - Count examples in specific category
- `count_intent(intent)` - Count examples for specific intent

### Advanced Calculations
- `calculate_total_of_last_n_intents(n)` - Sum of lowest frequency intents
- `get_last_n_intents(n)` - Get least frequent intents
- `calculate_sum_of_intents(intent_names)` - Sum specific intents
- `perform_calculation(expression)` - Basic math operations

### AI Summarization
- `summarize(topic)` - AI analysis of conversation patterns for categories/intents

## 💬 Example Interactions

### Data Queries
```
User: "Show me ORDER examples"
→ Returns 5 ORDER category examples

User: "Show me more"  
→ Returns 7 more ORDER examples (context preserved)

User: "What's the total count of the last two intents?"
→ Calculates: change_shipping_address (973) + check_cancellation_fee (950) = 1923
```

### Memory System
```
User: "Show ORDER examples" → [Memory: User asks about ORDER category]
User: "Get intent distributions" → [Memory: User interested in distributions]
User: "What do you remember about me?"
→ "You frequently ask about ORDER category, interested in both examples and distributions"
```

### AI Summaries
```
User: "Summarize how agents respond to check_invoice intent"
→ Analyzes 5 real conversations about invoice checking
→ Returns patterns: "Agents typically direct users to billing section, provide step-by-step access instructions..."
```

## 🚀 Setup

### Prerequisites
- Python 3.12
- OpenAI API key
- Streamlit

### Installation
```bash
git clone <repository>
cd customer-support-analyst
pip install -r requirements.txt
```

### Configuration
```bash
# Set up your OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Optional: Configure LangChain tracing
export LANGCHAIN_API_KEY=your_langchain_key
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT=CustomerSupportLangGraph
```

### Run Application
```bash
streamlit run app.py
```

## 🎯 Usage Tips

### Getting Started
1. **Ask for examples**: "Show me ORDER examples"
2. **Follow up naturally**: "Show me more"
3. **Check your memory**: "What do you remember about me?"
4. **Explore data**: "What are the most frequent categories?"

### Advanced Usage
- **Mathematical queries**: "What's the total count of the last 3 intents?"
- **Pattern analysis**: "Summarize how agents handle PAYMENT category"
- **Context switching**: Ask about different categories to build your profile
- **Session management**: Use sidebar to switch between conversation threads

### Memory Building
- Your profile builds automatically as you interact
- Tracks query patterns, preferred categories, interaction style
- Ask memory questions anytime to see what the system has learned
- Each session maintains separate memory profiles

## 🔍 Technical Notes

- **State Persistence**: Uses LangGraph's MemorySaver with proper serialization
- **Error Handling**: Graceful handling of out-of-scope queries and typos
- **Performance**: Efficient dataset querying with pandas operations
- **Scalability**: Thread-based sessions support multiple concurrent users
- **Data Safety**: All calculations convert numpy types for proper serialization
