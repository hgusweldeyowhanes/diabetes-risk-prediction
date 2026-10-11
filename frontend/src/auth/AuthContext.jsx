import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { api, setToken } from '../services/api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    const token = localStorage.getItem('gp_token')
    if (!token) {
      setReady(true)
      return
    }
    setToken(token)
    api
      .me()
      .then(setUser)
      .catch(() => {
        setToken('')
        localStorage.removeItem('gp_token')
        localStorage.removeItem('gp_refresh')
      })
      .finally(() => setReady(true))
  }, [])

  const value = useMemo(
    () => ({
      user,
      ready,
      async login(username, password) {
        const tokens = await api.login(username, password)
        localStorage.setItem('gp_token', tokens.access)
        localStorage.setItem('gp_refresh', tokens.refresh || '')
        setToken(tokens.access)
        const me = await api.me()
        setUser(me)
        return me
      },
      async register(body) {
        await api.register(body)
        const tokens = await api.login(body.username, body.password)
        localStorage.setItem('gp_token', tokens.access)
        localStorage.setItem('gp_refresh', tokens.refresh || '')
        setToken(tokens.access)
        const me = await api.me()
        setUser(me)
        return me
      },
      logout() {
        setToken('')
        localStorage.removeItem('gp_token')
        localStorage.removeItem('gp_refresh')
        setUser(null)
      },
    }),
    [user, ready]
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth outside provider')
  return ctx
}
