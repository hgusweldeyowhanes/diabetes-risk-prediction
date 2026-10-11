import { NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import { useI18n } from '../i18n/I18nContext'
import LangSwitch from './LangSwitch'

export default function Navigation() {
  const { t } = useI18n()
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  return (
    <header className="nav no-print">
      <NavLink to="/" className="nav-brand">
        {t('brand')}
      </NavLink>
      <nav className="nav-links">
        <NavLink to="/">{t('navHome')}</NavLink>
        <NavLink to="/predict">{t('navAssess')}</NavLink>
        <NavLink to="/results">{t('navResults')}</NavLink>
        {user && <NavLink to="/history">{t('navHistory')}</NavLink>}
        <NavLink to="/analytics">{t('navInsights')}</NavLink>
        {user ? (
          <button
            type="button"
            className="linkish"
            onClick={() => {
              logout()
              navigate('/')
            }}
          >
            {t('navLogout')}
          </button>
        ) : (
          <>
            <NavLink to="/login">{t('navLogin')}</NavLink>
            <NavLink to="/register">{t('navRegister')}</NavLink>
          </>
        )}
        <LangSwitch />
      </nav>
    </header>
  )
}
