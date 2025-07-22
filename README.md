# Customer Support Analyst (LangGraph + Memory)

## Features
1. **Conversation Memory**:
   - Track key user facts across sessions
   - Automatic memory summarization
   - "Show My Memory" button to view your profile

2. **Session Management**:
   - Session ID in sidebar
   - Resume conversations after reloads
   - SQLite-backed storage

3. **Follow-up Support**:
   - "Show me more examples" works naturally
   - Context-aware responses
   - Tool result history

## Setup
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_api_key_here
streamlit run langgraph_agent.py
