import { useState, useEffect } from 'react'
import {
  seedStore,
  residentId,
  addResidentReport,
  reviewReport,
  updateIncident,
} from './incidents'
import type { Store } from './incidents'
import ReportForm from './ReportForm'
import { MyReports, ReviewQueue, Incidents, VideoMonitor } from './IncidentPanels'
import './workflow.css'

const STORAGE_KEY = 'bantai_municipal_session_v3'

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

  // Persist session store across page refreshes
  useEffect(() => {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(store))
    } catch {
      // Storage quota or privacy restriction
    }
  }, [store])

  const pendingCount = store.observations.filter(o => o.review === 'Pending review').length
  const myReportsCount = store.observations.filter(
    o => o.reporter.includes('Verified Resident') || o.reporter === residentId
  ).length

  function switchRole(next: 'user' | 'admin') {
    setMode(next)
    setNotice('')
    if (next === 'user') {
      setPage('submit')
    } else {
      setPage('review')
    }
  }

  const navigation =
    mode === 'user'
      ? [
          { key: 'submit', label: '➕ Submit Report' },
          { key: 'mine', label: `📋 My Reports (${myReportsCount})` },
        ]
      : [
          { key: 'review', label: `📍 Priority Hotspots & Review (${pendingCount})` },
          { key: 'incidents', label: '📋 Incident Records' },
          { key: 'video', label: '🎥 Patrol Scanner' },
        ]

  return (
    <div className="app-shell workflow">
      {/* Site Header */}
      <header className="site-header">
        <div className="page-width header-top">
          <div>
            <div className="brand-name">
              BANTAI <span>STREET & SANITATION INTELLIGENCE</span>
            </div>
            <div className="brand-subtitle">MUNICIPAL INCIDENT MONITORING · BARANGAY SANTA ELENA</div>
          </div>

          <div className="header-right" style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <button
              type="button"
              className={`button ${mode === 'user' ? 'button-primary' : 'button-outline'}`}
              onClick={() => switchRole('user')}
            >
              Resident Portal
            </button>

            <button
              type="button"
              className={`button ${mode === 'admin' ? 'button-primary' : 'button-outline'}`}
              onClick={() => switchRole('admin')}
            >
              Barangay Operations {pendingCount > 0 && <span className="count-pill" style={{ marginLeft: '6px' }}>{pendingCount}</span>}
            </button>
          </div>
        </div>

        {/* Primary Navigation */}
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
          {mode === 'admin' ? 'BARANGAY OPERATIONS MANAGEMENT' : 'SANTA ELENA CITIZEN PORTAL'}
        </div>

        <h1>
          {page === 'submit' && 'Submit an Incident Report'}
          {page === 'mine' && 'My Submitted Reports'}
          {page === 'review' && 'Priority Hotspots & Review Queue'}
          {page === 'incidents' && 'Street Incident Records'}
          {page === 'video' && 'Patrol Scanner & Media Analysis'}
        </h1>

        <p>
          {mode === 'user'
            ? 'Report potholes, surface damage, and waste accumulation in your area. Track status updates as maintenance crews inspect and resolve issues.'
            : 'Review verified citizen reports, examine automated bounding boxes, and manage barangay maintenance priorities.'}
        </p>

        {notice && (
          <div className="notice" role="status" style={{ marginBottom: '16px' }}>
            {notice}
          </div>
        )}

        {/* Resident Portal Views */}
        {mode === 'user' && page === 'submit' && (
          <ReportForm
            onSubmit={report => {
              const id = `OBS-2026-${crypto.randomUUID().slice(0, 5).toUpperCase()}`
              setStore(s => addResidentReport(s, report, id, new Date().toISOString()))
              setPage('mine')
              setNotice(`Report #${id} submitted successfully. You can monitor its status under My Reports.`)
            }}
          />
        )}

        {mode === 'user' && page === 'mine' && <MyReports store={store} />}

        {/* Barangay Operations Views */}
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
                  `INC-MAR-${crypto.randomUUID().slice(0, 4).toUpperCase()}`
                )
              )
              setNotice(
                decision === 'reject'
                  ? 'Report declined and marked with reason.'
                  : 'Report approved. Street Hotspot priority and incident records have been updated.'
              )
            }}
          />
        )}

        {mode === 'admin' && page === 'incidents' && (
          <Incidents
            store={store}
            onUpdate={(id, status, note) => {
              setStore(s => updateIncident(s, id, status, note, new Date().toISOString()))
              setNotice(`Updated ${id} status to "${status}".`)
            }}
          />
        )}

        {mode === 'admin' && page === 'video' && (
          <VideoMonitor
            onAddRealObservations={newObs => {
              setStore(s => ({ ...s, observations: [...newObs, ...s.observations] }))
              setNotice(`Patrol video clip processed and registered into the review queue.`)
              setPage('review')
            }}
            onReview={() => setPage('review')}
          />
        )}
      </main>

      {/* Production Municipal Footer */}
      <footer className="site-footer" style={{ borderTop: '1px solid #dce5df', background: '#fff', marginTop: 'auto', padding: '16px 0' }}>
        <div className="page-width footer-inner" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', color: '#66746e' }}>
          <span>BANTAI Municipal Street & Sanitation Intelligence Platform</span>
          <span>Pilot Deployment: Barangay Santa Elena, Marikina City</span>
        </div>
      </footer>
    </div>
  )
}
