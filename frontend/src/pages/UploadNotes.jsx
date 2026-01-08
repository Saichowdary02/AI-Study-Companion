import React, { useState } from 'react'
import { Upload, FileText, Download, Loader2 } from 'lucide-react'
import { summarizeAPI } from '../services/api'

function UploadNotes() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile)
      setError(null)
      setResult(null)
    } else {
      setError('Please select a valid PDF file')
      setFile(null)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) {
      setError('Please select a PDF file')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await summarizeAPI.uploadAndSummarize(file)
      setResult(response)
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while processing the PDF')
    } finally {
      setLoading(false)
    }
  }

  const downloadSummary = () => {
    if (!result) return
    
    const element = document.createElement('a')
    const file = new Blob([result.summary], { type: 'text/plain' })
    element.href = URL.createObjectURL(file)
    element.download = `${result.filename.replace('.pdf', '')}_summary.txt`
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Summarize Notes</h1>
        <p className="text-gray-600">
          Upload your PDF lecture notes and get AI-powered summaries with key concepts
        </p>
      </div>

      <div className="card">
        <form onSubmit={handleSubmit} className="space-y-6">
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

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-600 text-sm">{error}</p>
            </div>
          )}

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
              'Generate Summary'
            )}
          </button>
        </form>
      </div>

      {result && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Summary Results</h2>
            <button
              onClick={downloadSummary}
              className="btn-secondary flex items-center"
            >
              <Download className="h-4 w-4 mr-2" />
              Download
            </button>
          </div>
          
          <div className="space-y-4">
            <div className="bg-gray-50 rounded-lg p-4">
              <h3 className="font-medium text-gray-900 mb-2">File Information</h3>
              <div className="text-sm text-gray-600 space-y-1">
                <p><strong>Filename:</strong> {result.filename}</p>
                <p><strong>Pages:</strong> {result.page_count}</p>
                <p><strong>Status:</strong> {result.success ? 'Successfully processed' : 'Processing failed'}</p>
              </div>
            </div>

            <div>
              <h3 className="font-medium text-gray-900 mb-2">AI Summary</h3>
              <div className="bg-white border border-gray-200 rounded-lg p-4 max-h-96 overflow-y-auto">
                <pre className="whitespace-pre-wrap text-sm text-gray-700 font-sans">
                  {result.summary}
                </pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default UploadNotes
