import React, { useState, useEffect } from 'react'
import { Plus, Calendar, Clock, CheckCircle, Circle, Trash2, Loader2, FolderOpen, Target, Users, Eye, EyeOff } from 'lucide-react'
import { tasksAPI, plansAPI } from '../services/api'

function TaskManager() {
  const [tasks, setTasks] = useState([])
  const [plans, setPlans] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showCreateForm, setShowCreateForm] = useState(true)
  const [showPlanForm, setShowPlanForm] = useState(false)
  const [showCreatePlanForm, setShowCreatePlanForm] = useState(false)
  const [expandedPlans, setExpandedPlans] = useState(new Set())
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    due_date: '',
    plan_id: null
  })
  const [planData, setPlanData] = useState({
    assignment_description: '',
    due_date: '',
    steps: 5
  })
  const [createPlanData, setCreatePlanData] = useState({
    name: '',
    description: '',
    assignment_description: '',
    due_date: ''
  })

  useEffect(() => {
    fetchTasks()
    fetchPlans()
  }, [])

  const fetchTasks = async () => {
    try {
      setLoading(true)
      const response = await tasksAPI.getTasks()
      setTasks(response)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch tasks')
    } finally {
      setLoading(false)
    }
  }

  const fetchPlans = async () => {
    try {
      const response = await plansAPI.getPlans()
      setPlans(response)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch plans')
    }
  }

  const handleCreateTask = async (e) => {
    e.preventDefault()
    try {
      setLoading(true)
      await tasksAPI.createTask(formData, formData.plan_id)
      setFormData({ title: '', description: '', due_date: '', plan_id: null })
      setShowCreateForm(true)
      fetchTasks()
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to create task')
    } finally {
      setLoading(false)
    }
  }

  const handleCreatePlan = async (e) => {
    e.preventDefault()
    try {
      setLoading(true)
      await plansAPI.createPlan(createPlanData)
      setCreatePlanData({ name: '', description: '', assignment_description: '', due_date: '' })
      setShowCreatePlanForm(false)
      fetchPlans()
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to create plan')
    } finally {
      setLoading(false)
    }
  }

  const handleCreateTaskPlan = async (e) => {
    e.preventDefault()
    try {
      setLoading(true)
      const steps = typeof planData.steps === 'number' ? planData.steps : parseInt(planData.steps) || 5
      await tasksAPI.createTaskPlan({
        assignment_description: planData.assignment_description,
        due_date: planData.due_date,
        steps
      })
      setPlanData({ assignment_description: '', due_date: '', steps: 5 })
      setShowPlanForm(false)
      fetchTasks()
      fetchPlans()
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to create task plan')
    } finally {
      setLoading(false)
    }
  }

  const handleStatusChange = async (taskId, newStatus) => {
    try {
      await tasksAPI.updateTaskStatus(taskId, newStatus)
      // Always refetch to reflect updated status without removing the task
      fetchTasks()
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to update task status')
    }
  }

  const handleDeleteTask = async (taskId) => {
    if (window.confirm('Are you sure you want to delete this task?')) {
      try {
        await tasksAPI.deleteTask(taskId)
        fetchTasks()
      } catch (err) {
        setError(err.response?.data?.detail || 'Failed to delete task')
      }
    }
  }

  const handleDeletePlan = async (planId) => {
    if (window.confirm('Are you sure you want to delete this plan and all its tasks?')) {
      try {
        await plansAPI.deletePlan(planId)
        fetchPlans()
        fetchTasks()
      } catch (err) {
        setError(err.response?.data?.detail || 'Failed to delete plan')
      }
    }
  }

  const togglePlanExpansion = (planId) => {
    const newExpanded = new Set(expandedPlans)
    if (newExpanded.has(planId)) {
      newExpanded.delete(planId)
    } else {
      newExpanded.add(planId)
    }
    setExpandedPlans(newExpanded)
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    })
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-100'
      case 'in_progress':
        return 'text-blue-600 bg-blue-100'
      case 'cancelled':
        return 'text-red-600 bg-red-100'
      default:
        return 'text-gray-600 bg-gray-100'
    }
  }

  const groupTasksByPlan = () => {
    const plannedTasks = tasks.filter(task => task.plan_id)
    const individualTasks = tasks.filter(task => !task.plan_id)
    
    const planGroups = {}
    plannedTasks.forEach(task => {
      if (!planGroups[task.plan_id]) {
        planGroups[task.plan_id] = {
          plan: plans.find(p => p.id === task.plan_id),
          tasks: []
        }
      }
      planGroups[task.plan_id].tasks.push(task)
    })
    
    return { planGroups, individualTasks }
  }

  // Removed view filter logic

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Smart Task Manager</h1>
        <p className="text-gray-600">
          Create and manage tasks with automatic subtask scheduling and deadline planning
        </p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-600 text-sm">{error}</p>
        </div>
      )}

      {/* Removed view mode toggle */}

      <div className="flex flex-wrap gap-4 justify-center">
        <button
          onClick={() => { setShowCreateForm(true); setShowPlanForm(false); setShowCreatePlanForm(false); }}
          className={`${showCreateForm && !showPlanForm ? 'btn-primary' : 'btn-secondary'} flex items-center`}
        >
          <Plus className="h-4 w-4 mr-2" />
          Create General Task
        </button>
        <button
          onClick={() => { setShowPlanForm(true); setShowCreateForm(false); setShowCreatePlanForm(false); }}
          className={`${showPlanForm ? 'btn-primary' : 'btn-secondary'} flex items-center`}
        >
          <Target className="h-4 w-4 mr-2" />
          Create Task with AI Plan
        </button>
      </div>

      {/* Create Task Form moved to replace Plans section */}

      {/* Create Plan Form */}
      {showCreatePlanForm && (
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Create New Plan</h2>
          <form onSubmit={handleCreatePlan} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Plan Name
              </label>
              <input
                type="text"
                value={createPlanData.name}
                onChange={(e) => setCreatePlanData({ ...createPlanData, name: e.target.value })}
                className="input-field"
                placeholder="e.g., Final Project Plan"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description
              </label>
              <textarea
                value={createPlanData.description}
                onChange={(e) => setCreatePlanData({ ...createPlanData, description: e.target.value })}
                className="input-field"
                rows="3"
                placeholder="Brief description of this plan..."
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Assignment Description
              </label>
              <textarea
                value={createPlanData.assignment_description}
                onChange={(e) => setCreatePlanData({ ...createPlanData, assignment_description: e.target.value })}
                className="input-field"
                rows="3"
                placeholder="Detailed description of the assignment..."
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Due Date
              </label>
              <input
                type="date"
                value={createPlanData.due_date}
                onChange={(e) => setCreatePlanData({ ...createPlanData, due_date: e.target.value })}
                className="input-field"
                required
              />
            </div>
            <div className="flex gap-2">
              <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Create Plan'}
              </button>
              <button
                type="button"
                onClick={() => setShowCreatePlanForm(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* AI Task Plan Form */}
      {showPlanForm && (
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">AI Task Planning</h2>
          <form onSubmit={handleCreateTaskPlan} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Assignment Description
              </label>
              <textarea
                value={planData.assignment_description}
                onChange={(e) => setPlanData({ ...planData, assignment_description: e.target.value })}
                className="input-field"
                rows="4"
                placeholder="Describe your assignment in detail..."
                required
              />
            </div>
            <div className="bg-blue-50 p-3 rounded-lg border border-blue-100">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Due Date
              </label>
              <input
                type="date"
                value={planData.due_date}
                onChange={(e) => setPlanData({ ...planData, due_date: e.target.value })}
                className="input-field bg-white"
                required
              />
              {planData.due_date && (
                <div className="mt-2">
                  <p className="text-xs font-medium text-gray-700 mb-1">Estimated Subtask Deadlines:</p>
                  <div className="space-y-1">
                    {Array.from({length: planData.steps}).map((_, i) => {
                      const today = new Date();
                      const dueDate = new Date(planData.due_date);
                      const daysDiff = Math.floor((dueDate - today) / (1000 * 60 * 60 * 24));
                      const stepDate = new Date(today);
                      stepDate.setDate(today.getDate() + Math.floor(daysDiff * (i + 1) / planData.steps));
                      return (
                        <div key={i} className="flex items-center text-xs text-gray-600">
                          <span className="w-20">Step {i + 1}:</span>
                          <span>{stepDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Number of Steps (Optional, default: 5)
              </label>
              <input
                type="number"
                min="1"
                max="10"
                value={planData.steps}
                onChange={(e) => {
                  const raw = e.target.value
                  // Allow empty while typing; clamp on submit
                  if (raw === '') {
                    setPlanData({ ...planData, steps: '' })
                  } else {
                    const num = Math.max(1, Math.min(10, parseInt(raw))) || ''
                    setPlanData({ ...planData, steps: num })
                  }
                }}
                className="input-field"
                placeholder="5"
              />
            </div>
            <div className="flex gap-2">
              <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Generate Plan'}
              </button>
              <button
                type="button"
                onClick={() => setShowPlanForm(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Tasks and Plans List */}
      {!showPlanForm && (
      <div className="space-y-6">
        {showCreateForm ? (
          <div className="card">
            <h2 className="text-xl font-semibold mb-4">Create New Task</h2>
            <form onSubmit={handleCreateTask} className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Task Title
                    </label>
                    <input
                      type="text"
                      value={formData.title}
                      onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                      className="input-field"
                      placeholder="What needs to be done?"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Description
                    </label>
                    <textarea
                      value={formData.description}
                      onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                      className="input-field"
                      rows="3"
                      placeholder="Add details about the task..."
                      required
                    />
                  </div>
                  <div className="bg-blue-50 p-3 rounded-lg border border-blue-100">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Due Date
                    </label>
                    <input
                      type="date"
                      value={formData.due_date}
                      onChange={(e) => setFormData({ ...formData, due_date: e.target.value })}
                      className="input-field bg-white"
                      required
                    />
                  </div>
                  <div className="flex gap-2">
                    <button type="submit" className="btn-primary" disabled={loading}>
                      {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Create Task'}
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowCreateForm(false)}
                      className="btn-secondary"
                    >
                      Cancel
                    </button>
                  </div>
                </form>
              </div>
        ) : loading && !tasks.length ? (
          <div className="card">
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin mr-2" />
              <span>Loading tasks...</span>
            </div>
          </div>
        ) : tasks.length === 0 ? (
          <div className="card">
            <div className="text-center py-8 text-gray-500">
              <Clock className="h-12 w-12 mx-auto mb-4 text-gray-300" />
              <p>No tasks yet. Create your first task to get started!</p>
            </div>
          </div>
        ) : (
          <>
            {/* Plans Section OR Create New Task (when creating general task) */}
            <div className="card">
                <h2 className="text-xl font-semibold mb-4 flex items-center">
                  <FolderOpen className="h-5 w-5 mr-2 text-blue-600" />
                  Plans ({Object.keys(groupTasksByPlan().planGroups).length})
                </h2>
                {Object.keys(groupTasksByPlan().planGroups).length === 0 ? (
                  <div className="text-center py-6 text-gray-500">
                    <Target className="h-8 w-8 mx-auto mb-2 text-gray-300" />
                    <p>No plans yet. Create a plan to organize your tasks!</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {Object.entries(groupTasksByPlan().planGroups).map(([planId, planGroup]) => {
                      const plan = planGroup.plan
                      const planTasks = planGroup.tasks
                      const isExpanded = expandedPlans.has(parseInt(planId))
                      const completedTasks = planTasks.filter(task => task.status === 'completed').length
                      const progressPercentage = planTasks.length > 0 ? (completedTasks / planTasks.length) * 100 : 0
                      
                      return (
                        <div key={planId} className="border border-blue-200 rounded-lg bg-blue-50/30">
                          {/* Plan Header */}
                          <div 
                            className="p-4 cursor-pointer hover:bg-blue-50/50 transition-colors"
                            onClick={() => togglePlanExpansion(parseInt(planId))}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-3">
                                <button className="text-blue-600 hover:text-blue-800 transition-colors">
                                  {isExpanded ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                                </button>
                                <div>
                                  <h3 className="font-semibold text-gray-900">{plan?.name || 'Unknown Plan'}</h3>
                                  <p className="text-sm text-gray-600">{plan?.description}</p>
                                </div>
                              </div>
                              <div className="flex items-center gap-4">
                                <div className="text-right">
                                  <div className="text-sm text-gray-600">
                                    {completedTasks}/{planTasks.length} completed
                                  </div>
                                  <div className="w-24 bg-gray-200 rounded-full h-2 mt-1">
                                    <div 
                                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                      style={{ width: `${progressPercentage}%` }}
                                    ></div>
                                  </div>
                                </div>
                                <div className="text-right">
                                  <div className="text-xs text-gray-500">
                                    Due: {plan?.due_date ? formatDate(plan.due_date) : 'N/A'}
                                  </div>
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation()
                                      handleDeletePlan(parseInt(planId))
                                    }}
                                    className="text-red-400 hover:text-red-600 transition-colors mt-1"
                                  >
                                    <Trash2 className="h-4 w-4" />
                                  </button>
                                </div>
                              </div>
                            </div>
                          </div>
                          
                          {/* Plan Tasks */}
                          {isExpanded && (
                            <div className="border-t border-blue-200 bg-white rounded-b-lg">
                              <div className="p-4 space-y-3">
                                {planTasks.map((task) => (
                                  <div key={task.id} className="border border-gray-200 rounded-lg p-3 bg-white hover:shadow-sm transition-shadow">
                                    <div className="flex items-start justify-between">
                                      <div className="flex-1">
                                        <div className="flex items-center gap-3 mb-1">
                                          <button
                                            onClick={() => handleStatusChange(
                                              task.id, 
                                              task.status === 'completed' ? 'pending' : 'completed'
                                            )}
                                            className="text-gray-400 hover:text-gray-600 transition-colors"
                                          >
                                            {task.status === 'completed' ? (
                                              <CheckCircle className="h-4 w-4 text-green-500" />
                                            ) : (
                                              <Circle className="h-4 w-4" />
                                            )}
                                          </button>
                                          <h4 className={`text-sm font-medium ${task.status === 'completed' ? 'line-through text-gray-500' : 'text-gray-900'}`}>
                                            {task.title}
                                          </h4>
                                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(task.status)}`}>
                                            {task.status.replace('_', ' ')}
                                          </span>
                                        </div>
                                        <p className="text-gray-600 text-xs ml-7">{task.description}</p>
                                  <div className="flex items-center text-xs text-gray-500 ml-7 mt-1">
                                    <Calendar className="h-3 w-3 mr-1" />
                                    <span>Due: {formatDate(task.due_date)}</span>
                                    {task.plan_id && (
                                      <span className="ml-2 px-1.5 py-0.5 rounded text-xs bg-blue-100 text-blue-800">
                                        Subtask
                                      </span>
                                    )}
                                  </div>
                                      </div>
                                      <button
                                        onClick={() => handleDeleteTask(task.id)}
                                        className="text-red-400 hover:text-red-600 transition-colors ml-2"
                                      >
                                        <Trash2 className="h-3 w-3" />
                                      </button>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>

            
          </>
        )}
        {/* Individual Tasks Section - always visible */}
        <div className="card">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <Users className="h-5 w-5 mr-2 text-gray-600" />
              Individual Tasks ({groupTasksByPlan().individualTasks.length})
            </h2>
            {groupTasksByPlan().individualTasks.length === 0 ? (
              <div className="text-center py-6 text-gray-500">
                <Users className="h-8 w-8 mx-auto mb-2 text-gray-300" />
                <p>No individual tasks. Create a task or add tasks to a plan!</p>
              </div>
            ) : (
              <div className="space-y-3">
                {groupTasksByPlan().individualTasks.map((task) => (
                  <div key={task.id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <button
                            onClick={() => handleStatusChange(
                              task.id, 
                              task.status === 'completed' ? 'pending' : 'completed'
                            )}
                            className="text-gray-400 hover:text-gray-600 transition-colors"
                          >
                            {task.status === 'completed' ? (
                              <CheckCircle className="h-5 w-5 text-green-500" />
                            ) : (
                              <Circle className="h-5 w-5" />
                            )}
                          </button>
                          <h3 className={`font-medium ${task.status === 'completed' ? 'line-through text-gray-500' : 'text-gray-900'}`}>
                            {task.title}
                          </h3>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(task.status)}`}>
                            {task.status.replace('_', ' ')}
                          </span>
                        </div>
                        <p className="text-gray-600 text-sm mb-2">{task.description}</p>
                        <div className="flex items-center text-xs text-gray-500">
                          <Calendar className="h-3 w-3 mr-1" />
                          <span>Due: {formatDate(task.due_date)}</span>
                        </div>
                      </div>
                      <button
                        onClick={() => handleDeleteTask(task.id)}
                        className="text-red-400 hover:text-red-600 transition-colors ml-4"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
      </div>
      )}
    </div>
  )
}

export default TaskManager
