const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api/v1'

let authToken = localStorage.getItem('gp_token') || ''

export function setToken(token) {
  authToken = token || ''
}

async function request(path, options = {}) {
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  }
  if (authToken) headers.Authorization = `Bearer ${authToken}`

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    const raw = data.detail || data.error || data.message || data.username || res.statusText
    const msg = Array.isArray(raw) ? raw[0] : raw
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
  }
  return data
}

function rows(data) {
  if (Array.isArray(data)) return data
  if (data && Array.isArray(data.results)) return data.results
  return []
}

export const api = {
  health: () => request('/health/'),
  predict: (body) =>
    request('/predictions/run/', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  history: () => request('/predictions/history/').then(rows),
  activeModel: () => request('/predictions/models/active/'),
  me: () => request('/auth/me/'),
  register: (body) =>
    request('/auth/register/', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  login: (username, password) =>
    request('/auth/token/', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }),
}

export default api
