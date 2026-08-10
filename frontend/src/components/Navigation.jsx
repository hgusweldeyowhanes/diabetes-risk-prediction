import { NavLink } from 'react-router-dom'

export default function Navigation() {
  return (
    <header className="nav">
      <NavLink to="/" className="nav-brand">
        GlycoPath
      </NavLink>
      <nav className="nav-links">
        <NavLink to="/">Home</NavLink>
        <NavLink to="/predict">Assess</NavLink>
        <NavLink to="/results">Results</NavLink>
        <NavLink to="/analytics">Insights</NavLink>
      </nav>
    </header>
  )
}
