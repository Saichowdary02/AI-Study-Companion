# Smart Study Buddy

An AI-powered study companion that helps students with note summarization, task planning, and intelligent Q&A using advanced AI agents.

## 🚀 Features

### 📚 Note Summarizer Agent
- Upload PDF lecture notes
- AI-powered summarization with key concepts
- Download summaries as text files
- Support for multi-page PDFs

### 📋 Task Planner Agent
- Create individual study tasks
- AI-powered assignment breakdown
- Automatic subtask generation with deadlines
- Task status tracking and management

### 💬 Q&A Agent (RAG)
- Upload study materials for knowledge base
- Ask questions about your content
- Retrieval-Augmented Generation (RAG) with ChromaDB
- Source attribution and relevance scoring

## 🛠 Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **LangChain** - AI agent orchestration
- **LangGraph** - Agent workflow management
- **ChromaDB** - Vector database for embeddings
- **SQLite** - Relational database for tasks
- **PyMuPDF** - PDF text extraction
- **OpenAI** - LLM and embeddings

### Frontend
- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **React Router DOM** - Navigation
- **Axios** - HTTP client
- **Lucide React** - Icons

## 📁 Project Structure

```
smart-study-buddy/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── requirements.txt        # Python dependencies
│   ├── database.py            # SQLite database management
│   ├── vector_store.py        # ChromaDB integration
│   ├── pdf_processor.py       # PDF text extraction
│   ├── agents.py              # AI agents (LangChain)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── summarize.py       # Note summarization endpoints
│   │   ├── tasks.py           # Task management endpoints
│   │   └── qa.py              # Q&A endpoints
│   ├── tasks.db               # SQLite database (auto-created)
│   ├── chroma_db/             # ChromaDB storage (auto-created)
│   └── env_template.txt       # Environment variables template
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── UploadNotes.jsx    # PDF upload and summarization
│   │   │   ├── TaskManager.jsx    # Task creation and management
│   │   │   └── QnA.jsx            # Q&A interface
│   │   ├── services/
│   │   │   └── api.js             # API client
│   │   ├── App.jsx               # Main app component
│   │   ├── main.jsx              # React entry point
│   │   └── index.css             # Global styles
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── index.html
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- OpenAI API key

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/Saichowdary02/AI-Study-Companion.git
cd AI-Study-Companion

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Copy the template and add your OpenAI API key
copy env_template.txt .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=your_actual_api_key_here

# Run the backend server
python main.py
# Or with uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be available at `http://localhost:8000`

### 3. Frontend Setup

```bash
# Open a new terminal and navigate to frontend
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

### 4. Access the Application

Open your browser and navigate to `http://localhost:5173` to start using Smart Study Buddy!

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `backend` directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### API Endpoints

#### Summarize
- `POST /summarize/` - Upload and summarize PDF
- `GET /summarize/health` - Health check

#### Tasks
- `POST /tasks/` - Create a task
- `GET /tasks/` - Get all tasks
- `POST /tasks/plan` - Create AI task plan
- `PUT /tasks/{id}/status` - Update task status
- `DELETE /tasks/{id}` - Delete task

#### Q&A
- `POST /qa/upload_notes` - Upload PDF for Q&A
- `POST /qa/ask` - Ask a question
- `GET /qa/collection_info` - Get knowledge base info

## 📖 Usage Guide

### 1. Summarize Notes
1. Navigate to "Summarize Notes" page
2. Upload a PDF file containing lecture notes
3. Click "Generate Summary" to get AI-powered summary
4. Download the summary as a text file

### 2. Manage Tasks
1. Go to "Task Manager" page
2. Create individual tasks or use "AI Task Plan" for assignments
3. For AI planning, describe your assignment and set due date
4. Track task progress and mark as completed

### 3. Q&A Assistant
1. Visit "Q&A" page
2. Upload PDF study materials to build knowledge base
3. Ask questions about your uploaded content
4. Get intelligent answers with source attribution

## 🧠 AI Agents

### Note Summarizer Agent
- Uses GPT-3.5-turbo for summarization
- Extracts key concepts and important details
- Organizes information in structured format
- Optimized for academic content

### Task Planner Agent
- Breaks down assignments into subtasks
- Calculates realistic deadlines
- Considers task dependencies
- Generates actionable task descriptions

### Q&A Agent
- Uses Retrieval-Augmented Generation (RAG)
- ChromaDB for vector similarity search
- OpenAI embeddings for semantic search
- Provides source attribution and relevance scores

## 🗄 Data Storage

### SQLite Database (`tasks.db`)
- Stores task information
- Auto-created on first run
- Tables: `tasks` (id, title, description, due_date, status, created_at)

### ChromaDB (`./chroma_db/`)
- Stores document embeddings
- Persistent vector storage
- Collection: `study_notes`
- Auto-reloads on application restart

## 🔍 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure your API key is correctly set in `.env`
   - Check that you have sufficient API credits

2. **PDF Processing Errors**
   - Ensure PDF files are not password-protected
   - Check that PDFs contain extractable text (not just images)

3. **ChromaDB Issues**
   - Delete `./chroma_db` folder to reset vector storage
   - Ensure write permissions in the backend directory

4. **CORS Errors**
   - Ensure backend is running on port 8000
   - Check that frontend is running on port 5173

### Development Tips

- Use `--reload` flag with uvicorn for auto-restart on code changes
- Check browser console for frontend errors
- Monitor backend logs for API issues
- Use FastAPI docs at `http://localhost:8000/docs` for API testing

## 🚀 Local Development Only

This project is configured for local development only:

1. **Backend**: Runs on FastAPI with uvicorn development server
2. **Frontend**: Uses Vite development server
3. **Database**: SQLite for simple local storage
4. **Vector DB**: Local ChromaDB instance
5. **Environment**: Simple .env file for configuration

**Note**: This setup is optimized for local development and testing. For production deployment, additional configuration and security measures would be required.

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For support, please open an issue in the repository or contact the development team.

---

**Happy Studying! 🎓**
