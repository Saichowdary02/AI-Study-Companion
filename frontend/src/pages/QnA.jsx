import React, { useState, useEffect } from 'react'
import { Upload, MessageCircle, Send, FileText, Loader2, Database, Trash2 } from 'lucide-react'
import { qaAPI } from '../services/api'

function QnA() {
  const [file, setFile] = useState(null)
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [collectionInfo, setCollectionInfo] = useState(null)
  const [qaResult, setQaResult] = useState(null)
  const [uploadSuccess, setUploadSuccess] = useState(false)

  useEffect(() => {
    fetchCollectionInfo()
  }, [])

  const fetchCollectionInfo = async () => {
    try {
      const response = await qaAPI.getCollectionInfo()
      setCollectionInfo(response)
    } catch (err) {
      console.error('Failed to fetch collection info:', err)
    }
  }

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile)
      setError(null)
      setUploadSuccess(false)
    } else {
      setError('Please select a valid PDF file')
      setFile(null)
    }
  }

  const handleUpload = async (e) => {
    e.preventDefault()
    if (!file) {
      setError('Please select a PDF file')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const response = await qaAPI.uploadNotes(file)
      setUploadSuccess(true)
      setFile(null)
      fetchCollectionInfo()
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while uploading the PDF')
    } finally {
      setLoading(false)
    }
  }

  const handleAskQuestion = async (e) => {
    e.preventDefault()
    if (!question.trim()) {
      setError('Please enter a question')
      return
    }

    setLoading(true)
    setError(null)
    setQaResult(null)

    try {
      const response = await qaAPI.askQuestion(question)
      setQaResult(response)
      setQuestion('')
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while processing your question')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Q&A Assistant</h1>
        <p className="text-gray-600">
          Upload study materials and ask questions to get intelligent answers
        </p>
      </div>

      {/* Collection Info */}
      {collectionInfo && (
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Database className="h-5 w-5 text-primary-600" />
              <h3 className="font-semibold">Knowledge Base Status</h3>
            </div>
            {collectionInfo.total_documents > 0 && (
              <button
                onClick={async () => {
                  if (window.confirm('Are you sure you want to clear all documents from the knowledge base? This cannot be undone.')) {
                    try {
                      await qaAPI.clearKnowledgeBase();
                      fetchCollectionInfo();
                    } catch (err) {
                      setError('Failed to clear knowledge base');
                    }
                  }
                }}
                className="text-sm text-red-600 hover:text-red-800 flex items-center"
              >
                <Trash2 className="h-4 w-4 mr-1" />
                Clear All
              </button>
            )}
          </div>
          <p className="text-sm text-gray-600">
            {collectionInfo.total_documents} document(s) available for Q&A
          </p>
        </div>
      )}

      {/* Upload Section */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Upload Study Materials</h2>
        <form onSubmit={handleUpload} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select PDF File
            </label>
            <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-lg hover:border-primary-500 transition-colors">
              <div className="space-y-1 text-center">
                <Upload className="mx-auto h-12 w-12 text-gray-400" />
                <div className="flex text-sm text-gray-600">
                  <label
                    htmlFor="file-upload"
                    className="relative cursor-pointer bg-white rounded-md font-medium text-primary-600 hover:text-primary-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-primary-500"
                  >
                    <span>Upload a file</span>
                    <input
                      id="file-upload"
                      name="file-upload"
                      type="file"
                      className="sr-only"
                      accept=".pdf"
                      onChange={handleFileChange}
                    />
                  </label>
                  <p className="pl-1">or drag and drop</p>
                </div>
                <p className="text-xs text-gray-500">PDF files only</p>
              </div>
            </div>
            {file && (
              <div className="mt-2 flex items-center text-sm text-gray-600">
                <FileText className="h-4 w-4 mr-2" />
                <span>{file.name}</span>
                <span className="ml-2 text-gray-400">
                  ({(file.size / 1024 / 1024).toFixed(2)} MB)
                </span>
              </div>
            )}
          </div>

          <button
            type="submit"
            disabled={!file || loading}
            className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Processing...
              </>
            ) : (
              'Process Document'
            )}
          </button>
        </form>

        {uploadSuccess && (
          <div className="mt-4 bg-green-50 border border-green-200 rounded-lg p-4">
            <p className="text-green-600 text-sm">
              ✅ File uploaded and processed successfully! You can now ask questions about this content.
            </p>
          </div>
        )}
      </div>

      {/* Q&A Section */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Ask Questions</h2>
        <form onSubmit={handleAskQuestion} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Your Question
            </label>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="input-field"
              rows="3"
              placeholder="Ask any question about your uploaded study materials..."
              disabled={!collectionInfo?.total_documents}
            />
            {!collectionInfo?.total_documents && (
              <p className="text-sm text-gray-500 mt-1">
                Please upload some study materials first to enable Q&A
              </p>
            )}
          </div>

          <button
            type="submit"
            disabled={!question.trim() || loading || !collectionInfo?.total_documents}
            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Thinking...
              </>
            ) : (
              <>
                <Send className="h-4 w-4 mr-2" />
                Ask Question
              </>
            )}
          </button>
        </form>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-600 text-sm">{error}</p>
        </div>
      )}

      {/* Q&A Results */}
      {qaResult && (
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <MessageCircle className="h-5 w-5 text-primary-600" />
            <h3 className="text-lg font-semibold">Answer</h3>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4 mb-4">
            <p className="text-gray-700 whitespace-pre-wrap">{qaResult.answer}</p>
          </div>

          {qaResult.sources && qaResult.sources.length > 0 && (
            <div>
              <h4 className="font-medium text-gray-900 mb-2">Sources</h4>
              <div className="space-y-2">
                {qaResult.sources.map((source, index) => (
                  <div key={index} className="bg-white border border-gray-200 rounded-lg p-3">
                    <div className="flex items-start justify-between mb-2">
                      <span className="text-xs text-gray-500">
                        Source {index + 1} • Relevance: {Math.round(source.relevance_score * 100)}%
                      </span>
                    </div>
                    <p className="text-sm text-gray-700">{source.content}</p>
                    {source.metadata.filename && (
                      <p className="text-xs text-gray-500 mt-2">
                        From: {source.metadata.filename}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default QnA
