const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api/v1'

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    const msg = data.detail || data.error || data.message || res.statusText
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
  }
  return data
}

export const api = {
  health: () => request('/health/'),
  predict: (body) =>
    request('/predictions/run/', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  activeModel: () => request('/predictions/models/active/'),
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
