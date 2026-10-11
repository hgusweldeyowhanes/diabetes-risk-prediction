import { Navigate, Route, Routes } from 'react-router-dom'
import Navigation from './components/Navigation'
import Home from './pages/Home'
import Predict from './pages/Predict'
import Results from './pages/Results'
import Analytics from './pages/Analytics'
import Login from './pages/Login'
import Register from './pages/Register'
import History from './pages/History'

export default function App() {
  return (
    <div className="app-shell">
      <Navigation />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/results" element={<Results />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/history" element={<History />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  )
}
