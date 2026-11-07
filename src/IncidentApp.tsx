import { useState, useEffect } from 'react'
import {
  seedStore,
  rankHotspots,
  residentId,
  addResidentReport,
  reviewReport,
  updateIncident,
  addVideoObservations,
} from './incidents'
import type { Store } from './incidents'
import ReportForm from './ReportForm'
import { MyReports, ReviewQueue, Incidents, VideoMonitor } from './IncidentPanels'
import './workflow.css'

const STORAGE_KEY = 'bantai_session_store_v2'

function loadSavedStore(): Store | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw) as Store
      if (Array.isArray(parsed?.observations) && Array.isArray(parsed?.incidents)) {
        return parsed
      }
    }
  } catch {
    // Fall back to seed store
  }
  return null
}

export default function IncidentApp() {
  const [mode, setMode] = useState<'user' | 'admin'>('user')
  const [store, setStore] = useState<Store>(() => loadSavedStore() ?? seedStore())
  const [page, setPage] = useState<'submit' | 'mine' | 'review' | 'incidents' | 'video'>('submit')
  const [selectedStreet, setSelectedStreet] = useState<string>('')
  const [notice, setNotice] = useState<string>('')

  // Persist store to sessionStorage across accidental page refreshes
  useEffect(() => {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(store))
    } catch {
      // Storage quota or privacy restriction
    }
  }, [store])

  const pendingCount = store.observations.filter(o => o.review === 'Pending review').length
  const myReportsCount = store.observations.filter(o => o.reporter === residentId).length

  function handleResetDemo() {
    sessionStorage.removeItem(STORAGE_KEY)
    setStore(seedStore())
    setSelectedStreet('')
    setNotice('Demo data reset to fresh starting state.')
  }

  function switchRole(next: 'user' | 'admin') {
    setMode(next)
    setNotice('')
    if (next === 'user') {
      setPage('submit')
    } else {
      setPage('review')
    }
  }

  // Strict Privacy Navigation
  const navigation =
    mode === 'user'
      ? [
          { key: 'submit', label: '➕ Report a Problem' },
          { key: 'mine', label: `📋 My Sent Reports (${myReportsCount})` },
        ]
      : [
          { key: 'review', label: `📍 Priority Hotspots & Review (${pendingCount})` },
          { key: 'incidents', label: '📋 All Street Records' },
          { key: 'video', label: '🎥 Patrol Scanner' },
        ]

  return (
    <div className="app-shell workflow">
      {/* Header with Role Switcher & Reset Button */}
      <header className="site-header">
        <div className="page-width header-top">
          <div>
            <div className="brand-name">
              BANTAI <span>STREET MONITOR</span>
            </div>
            <div className="brand-subtitle">BARANGAY ROAD INFRASTRUCTURE & CLEANLINESS</div>
          </div>

          <div className="header-right" style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <button
              type="button"
              className={`button ${mode === 'user' ? 'button-primary' : 'button-outline'}`}
              onClick={() => switchRole('user')}
              title="Resident Mode: Submit reports and track their status"
            >
              👤 Resident (Grandma Mode)
            </button>

            <button
              type="button"
              className={`button ${mode === 'admin' ? 'button-primary' : 'button-outline'}`}
              onClick={() => switchRole('admin')}
              title="Admin Mode: Review queue, location hotspots, and all street records"
            >
              🛡️ Barangay Admin
            </button>

            <button
              type="button"
              className="button button-outline"
              style={{ fontSize: '11px', padding: '4px 8px', height: '30px', color: '#666', borderStyle: 'dashed' }}
              onClick={handleResetDemo}
              title="Reset data back to seed state"
            >
              🔄 Reset Demo
            </button>
          </div>
        </div>

        {/* Navigation tabs */}
        <nav className="nav-bar" aria-label="Main navigation">
          <div className="page-width nav-inner">
            {navigation.map(item => (
              <button
                key={item.key}
                type="button"
                className={`nav-item ${page === item.key ? 'active' : ''}`}
                aria-current={page === item.key ? 'page' : undefined}
                onClick={() => {
                  setPage(item.key as typeof page)
                  setNotice('')
                }}
              >
                {item.label}
              </button>
            ))}
          </div>
        </nav>
      </header>

      {/* Main Content Area */}
      <main className="page-width main-content">
        <div className="eyebrow">
          {mode === 'admin' ? '🛡️ BARANGAY OPERATIONS PORTAL' : '👤 RESIDENT COMMUNITY PORTAL'}
        </div>

        <h1>
          {page === 'submit' && 'Report a Street Problem'}
          {page === 'mine' && 'My Sent Reports & Status'}
          {page === 'review' && 'Priority Hotspots & Review Queue'}
          {page === 'incidents' && 'All Street Records & Cleanup Progress'}
          {page === 'video' && 'Patrol Camera Video Scanner'}
        </h1>

        <p>
          {mode === 'user'
            ? 'Report road potholes and trash piles in your neighborhood. Track each report until it is cleaned and repaired.'
            : 'Review incoming community reports, inspect scanned bounding boxes, and manage barangay cleanup priorities.'}
        </p>

        {notice && (
          <div className="notice" role="status" style={{ marginBottom: '16px' }}>
            {notice}
          </div>
        )}

        {/* --- RESIDENT SCREENS (STRICT PRIVACY) --- */}
        {mode === 'user' && page === 'submit' && (
          <ReportForm
            onSubmit={report => {
              const id = `OBS-${crypto.randomUUID().slice(0, 6).toUpperCase()}`
              setStore(s => addResidentReport(s, report, id, new Date().toISOString()))
              setPage('mine')
              setNotice(`✅ Report #${id} submitted successfully! You can track its progress below, or switch to Barangay Admin at the top to review it.`)
            }}
          />
        )}

        {mode === 'user' && page === 'mine' && <MyReports store={store} />}

        {/* --- ADMIN SCREENS --- */}
        {mode === 'admin' && page === 'review' && (
          <ReviewQueue
            store={store}
            selectedStreet={selectedStreet}
            onSelectStreet={setSelectedStreet}
            onReview={(id, decision, target, note) => {
              setStore(s =>
                reviewReport(
                  s,
                  id,
                  decision,
                  target,
                  note,
                  new Date().toISOString(),
                  `INC-${crypto.randomUUID().slice(0, 6).toUpperCase()}`
                )
              )
              setNotice(
                decision === 'reject'
                  ? 'Report declined with note saved.'
                  : 'Report approved! Street Hotspot priorities and incident records have been updated.'
              )
            }}
          />
        )}

        {mode === 'admin' && page === 'incidents' && (
          <Incidents
            store={store}
            onUpdate={(id, status, note) => {
              setStore(s => updateIncident(s, id, status, note, new Date().toISOString()))
              setNotice(`Updated ${id} to "${status}".`)
            }}
          />
        )}

        {mode === 'admin' && page === 'video' && (
          <VideoMonitor
            onGenerate={(location, category) => {
              setStore(s =>
                addVideoObservations(
                  s,
                  location,
                  category,
                  new Date().toISOString(),
                  crypto.randomUUID().slice(0, 6)
                )
              )
              setNotice('Patrol video clip simulated and added to the review queue.')
            }}
            onAddRealObservations={newObs => {
              setStore(s => ({ ...s, observations: [...newObs, ...s.observations] }))
              setNotice(`Scanned video added to the review queue as a single entry with frame-by-frame stepper.`)
              setPage('review')
            }}
            onReview={() => setPage('review')}
          />
        )}
      </main>
    </div>
  )
}
