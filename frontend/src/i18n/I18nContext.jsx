import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { DICTS } from './translations'

const KEY = 'gp_lang'
const I18nContext = createContext(null)

export function I18nProvider({ children }) {
  const [lang, setLangState] = useState(() => {
    const saved = localStorage.getItem(KEY)
    return DICTS[saved] ? saved : 'en'
  })

  useEffect(() => {
    document.documentElement.lang = lang
  }, [lang])

  const value = useMemo(() => {
    const dict = DICTS[lang] || DICTS.en
    return {
      lang,
      t: (key) => dict[key] ?? DICTS.en[key] ?? key,
      setLang: (next) => {
        const id = DICTS[next] ? next : 'en'
        localStorage.setItem(KEY, id)
        setLangState(id)
        document.documentElement.lang = id
      },
    }
  }, [lang])

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n() {
  const ctx = useContext(I18nContext)
  if (!ctx) throw new Error('useI18n outside provider')
  return ctx
}
