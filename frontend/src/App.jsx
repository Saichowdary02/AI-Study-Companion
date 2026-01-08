import React from 'react'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import UploadNotes from './pages/UploadNotes'
import TaskManager from './pages/TaskManager'
import QnA from './pages/QnA'
import { BookOpen, CheckSquare, MessageCircle, Home } from 'lucide-react'

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        {/* Navigation */}
        <nav className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16">
              <div className="flex items-center">
                <Link to="/" className="flex items-center space-x-2">
                  <BookOpen className="h-8 w-8 text-primary-600" />
                  <span className="text-xl font-bold text-gray-900">Smart Study Buddy</span>
                </Link>
              </div>
              <div className="flex items-center space-x-8">
                <Link 
                  to="/" 
                  className="flex items-center space-x-1 text-gray-600 hover:text-primary-600 transition-colors"
                >
                  <Home className="h-5 w-5" />
                  <span>Home</span>
                </Link>
                <Link 
                  to="/upload" 
                  className="flex items-center space-x-1 text-gray-600 hover:text-primary-600 transition-colors"
                >
                  <BookOpen className="h-5 w-5" />
                  <span>Summarize Notes</span>
                </Link>
                <Link 
                  to="/tasks" 
                  className="flex items-center space-x-1 text-gray-600 hover:text-primary-600 transition-colors"
                >
                  <CheckSquare className="h-5 w-5" />
                  <span>Task Manager</span>
                </Link>
                <Link 
                  to="/qa" 
                  className="flex items-center space-x-1 text-gray-600 hover:text-primary-600 transition-colors"
                >
                  <MessageCircle className="h-5 w-5" />
                  <span>Q&A</span>
                </Link>
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/upload" element={<UploadNotes />} />
            <Route path="/tasks" element={<TaskManager />} />
            <Route path="/qa" element={<QnA />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

function HomePage() {
  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Welcome to Smart Study Buddy
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Your AI-powered study companion that helps you summarize notes, plan tasks, and answer questions using advanced AI agents.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card text-center">
          <BookOpen className="h-12 w-12 text-primary-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2">Summarize Notes</h3>
          <p className="text-gray-600 mb-4">
            Upload PDF lecture notes and get AI-powered summaries with key concepts and important details.
          </p>
          <Link to="/upload" className="btn-primary">
            Get Started
          </Link>
        </div>

        <div className="card text-center">
          <CheckSquare className="h-12 w-12 text-primary-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2">Task Manager</h3>
          <p className="text-gray-600 mb-4">
            Create assignments and let AI break them down into manageable subtasks with deadlines.
          </p>
          <Link to="/tasks" className="btn-primary">
            Manage Tasks
          </Link>
        </div>

        <div className="card text-center">
          <MessageCircle className="h-12 w-12 text-primary-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2">Q&A Assistant</h3>
          <p className="text-gray-600 mb-4">
            Upload study materials and ask questions to get intelligent answers based on your content.
          </p>
          <Link to="/qa" className="btn-primary">
            Ask Questions
          </Link>
        </div>
      </div>

      <div className="card">
        <h2 className="text-2xl font-semibold mb-4">How It Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="bg-primary-100 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-3">
              <span className="text-primary-600 font-bold">1</span>
            </div>
            <h4 className="font-semibold mb-2">Upload Your Materials</h4>
            <p className="text-gray-600 text-sm">Upload PDF lecture notes or create assignment tasks</p>
          </div>
          <div className="text-center">
            <div className="bg-primary-100 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-3">
              <span className="text-primary-600 font-bold">2</span>
            </div>
            <h4 className="font-semibold mb-2">AI Processing</h4>
            <p className="text-gray-600 text-sm">Our AI agents analyze and process your content</p>
          </div>
          <div className="text-center">
            <div className="bg-primary-100 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-3">
              <span className="text-primary-600 font-bold">3</span>
            </div>
            <h4 className="font-semibold mb-2">Get Results</h4>
            <p className="text-gray-600 text-sm">Receive summaries, task plans, and intelligent answers</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
