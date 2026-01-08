import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`)
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

// Summarize API
export const summarizeAPI = {
  uploadAndSummarize: async (file) => {
    const formData = new FormData()
    formData.append('file', file)

    const response = await api.post('/api/v1/summarize/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  }
}

// Tasks API
export const tasksAPI = {
  createTask: async (taskData, planId = null) => {
    const response = await api.post('/api/v1/tasks/', taskData, {
      params: planId ? { plan_id: planId } : {}
    })
    return response.data
  },

  getTasks: async () => {
    const response = await api.get('/api/v1/tasks/')
    return response.data
  },

  createTaskPlan: async (assignmentData) => {
    const response = await api.post('/api/v1/tasks/plan', assignmentData)
    return response.data
  },

  updateTaskStatus: async (taskId, status) => {
    const response = await api.put(`/api/v1/tasks/${taskId}/status`, null, {
      params: { status }
    })
    return response.data
  },

  deleteTask: async (taskId) => {
    const response = await api.delete(`/api/v1/tasks/${taskId}`)
    return response.data
  }
}

// Plans API
export const plansAPI = {
  createPlan: async (planData) => {
    const response = await api.post('/api/v1/tasks/plans', planData)
    return response.data
  },

  getPlans: async () => {
    const response = await api.get('/api/v1/tasks/plans')
    return response.data
  },

  getPlan: async (planId) => {
    const response = await api.get(`/api/v1/tasks/plans/${planId}`)
    return response.data
  },

  getPlanTasks: async (planId) => {
    const response = await api.get(`/api/v1/tasks/plans/${planId}/tasks`)
    return response.data
  },

  deletePlan: async (planId) => {
    const response = await api.delete(`/api/v1/tasks/plans/${planId}`)
    return response.data
  }
}

// Q&A API
export const qaAPI = {
  uploadNotes: async (file) => {
    const formData = new FormData()
    formData.append('file', file)

    const response = await api.post('/api/v1/qa/upload_notes', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  askQuestion: async (question) => {
    const response = await api.post('/api/v1/qa/ask', { question })
    return response.data
  },

  getCollectionInfo: async () => {
    const response = await api.get('/api/v1/qa/collection_info')
    return response.data
  },

  clearKnowledgeBase: async () => {
    const response = await api.delete('/api/v1/qa/clear')
    return response.data
  }
}

export default api
