import { useEffect, useState } from 'react'
import {
  loadPersistedStore,
  savePersistedStore,
  resetPersistedStore,
  rankIncidents,
  categories,
  residentId,
  addResidentReport,
  reviewReport,
  updateIncident,
  addVideoObservations,
} from './incidents'
import type { Store } from './incidents'
import ReportForm from './ReportForm'
import {
  MyReports,
  ReviewQueue,
  Incidents,
  VideoMonitor,
  HotspotLeaderboard,
  Status,
  formatDate,
} from './IncidentPanels'
import './workflow.css'

function CivicIcon({ name, size = 18 }: { name: 'shield' | 'trash' | 'pothole' | 'check'; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {name === 'shield' && <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />}
      {name === 'trash' && (
        <>
          <path d="M3 6h18" />
          <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
        </>
      )}
      {name === 'pothole' && (
        <>
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
          <line x1="12" y1="9" x2="12" y2="13" />
          <line x1="12" y1="17" x2="12.01" y2="17" />
        </>
      )}
      {name === 'check' && <path d="M20 6 9 17l-5-5" />}
    </svg>
  )
}

export default function IncidentApp() {
  const [mode, setMode] = useState<'user' | 'admin'>('user')
  const [store, setStore] = useState<Store>(() => loadPersistedStore())
  const [page, setPage] = useState('overview')
  const [notice, setNotice] = useState('')
  const [reportLocationPrefill, setReportLocationPrefill] = useState<string | undefined>(undefined)

  useEffect(() => {
    savePersistedStore(store)
  }, [store])

  const pending = store.observations.filter(o => o.review === 'Pending review').length
  const navigation =
    mode === 'user'
      ? [
          ['overview', 'Overview & Hotspots'],
          ['submit', 'Submit Report'],
          ['mine', 'My Reports'],
        ]
      : [
          ['overview', 'Operations Dashboard'],
          ['review', `Review Queue (${pending})`],
          ['incidents', 'Incident Records'],
          ['video', 'CCTV / Dashcam Ingest'],
        ]

  function switchMode(next: 'user' | 'admin') {
    setMode(next)
    setPage(next === 'admin' ? 'review' : 'mine')
    setNotice('')
  }

  const ranked = rankIncidents(store)

  return (
    <div className="app-shell workflow">
      <header className="site-header">
        <div className="page-width header-top">
          <div className="brand">
            <div className="brand-mark civic-shield-mark" aria-hidden="true">
              <CivicIcon name="shield" size={22} />
            </div>
            <div>
              <div className="brand-name">
                BANTAI — <span>Barangay Automated Network for Trash and Anomaly Identification</span>
              </div>
              <div className="brand-subtitle">BARANGAY SANTA ELENA · CITY OF MARIKINA</div>
            </div>
          </div>
          <div className="header-right">
            <span className="pilot-pill">
              <span className="pilot-dot" /> MUNICIPAL AI ACTIVE
            </span>
            <button
              aria-pressed={mode === 'user'}
              className={`button ${mode === 'user' ? 'button-primary' : 'button-outline'}`}
              onClick={() => switchMode('user')}
            >
              Resident Portal
            </button>
            <button
              aria-pressed={mode === 'admin'}
              className={`button ${mode === 'admin' ? 'button-primary' : 'button-outline'}`}
              onClick={() => switchMode('admin')}
            >
              Barangay Operations
            </button>
          </div>
        </div>
        <nav className="nav-bar" aria-label="Main navigation">
          <div className="page-width nav-inner">
            {navigation.map(([key, label]) => (
              <button
                key={key}
                className={`nav-item ${page === key ? 'active' : ''}`}
                aria-current={page === key ? 'page' : undefined}
                onClick={() => {
                  if (key !== 'submit') setReportLocationPrefill(undefined)
                  setPage(key)
                  setNotice('')
                }}
              >
                {label}
              </button>
            ))}
            <span className="nav-context">
              DUAL-MODEL ROUTING <span className="context-divider">·</span> TRASH (MODEL A) &amp; POTHOLE (MODEL B)
            </span>
          </div>
        </nav>
      </header>

      <main className="page-width main-content">
        <div className="eyebrow">
          <span className="eyebrow-line" />
          {mode === 'admin'
            ? 'BARANGAY OPERATIONS & DISPATCH'
            : 'CITIZEN REPORTING & TRANSPARENCY PORTAL'}
        </div>
        <h1>
          {page === 'overview'
            ? mode === 'admin'
              ? 'Municipal Incident & Hotspot Operations'
              : 'A cleaner, safer Barangay Santa Elena starts with your report.'
            : page === 'mine'
              ? 'My Submitted Reports'
              : page === 'review'
                ? 'Barangay Verification Queue'
                : page === 'incidents'
                  ? 'Consolidated Incident Records'
                  : page === 'video'
                    ? 'CCTV & Mobile Dashcam Ingestion'
                    : 'Submit a Street-Level Report'}
        </h1>
        <p className="lead-copy">
          Report roadside trash or road potholes with photo or video evidence. Track every submission from AI analysis to barangay verification and field resolution.
        </p>

        {notice && (
          <div className="notice" role="status">
            {notice}
          </div>
        )}

        {page === 'submit' && mode === 'user' && (
          <ReportForm
            initialLocation={reportLocationPrefill}
            onSubmit={report => {
              const now = new Date().toISOString()
              const baseId = `OBS-${crypto.randomUUID().slice(0, 8).toUpperCase()}`
              const frames = report.dualModelResult?.frames
              const resolvedVideoUrl =
                report.videoUrl || report.dualModelResult?.source_media?.media_url
              const firstFrame = frames && frames.length > 0 ? frames[0] : undefined

              setStore(s =>
                addResidentReport(
                  s,
                  {
                    ...report,
                    photo: firstFrame?.frame_url || report.photo,
                    videoUrl: resolvedVideoUrl,
                    detections: firstFrame?.trash_findings?.detections || report.detections,
                    regions: firstFrame?.trash_findings?.region_list || report.regions,
                    defects: firstFrame?.road_defect_findings?.defects || report.defects,
                    imageWidth: firstFrame?.width || report.imageWidth,
                    imageHeight: firstFrame?.height || report.imageHeight,
                  },
                  baseId,
                  now
                )
              )
              setReportLocationPrefill(undefined)
              setPage('mine')
              setNotice(
                frames && frames.length > 1
                  ? `${baseId} submitted with ${frames.length} extracted video frames queued for Barangay verification.`
                  : `${baseId} submitted and queued for Barangay verification.`
              )
            }}
          />
        )}

        {page === 'mine' && mode === 'user' && <MyReports store={store} />}

        {page === 'review' && mode === 'admin' && (
          <ReviewQueue
            store={store}
            onReview={(id, decision, target, note) => {
              setStore(s =>
                reviewReport(
                  s,
                  id,
                  decision,
                  target,
                  note,
                  new Date().toISOString(),
                  `INC-${crypto.randomUUID().slice(0, 8).toUpperCase()}`
                )
              )
              setNotice(
                decision === 'reject'
                  ? 'Submission marked as rejected. The resident can view your verification note.'
                  : 'Submission verified. Incident frequency and Barangay Hotspot Leaderboard updated.'
              )
            }}
          />
        )}

        {page === 'incidents' && mode === 'admin' && (
          <Incidents
            store={store}
            onUpdate={(id, status, note) => {
              setStore(s => updateIncident(s, id, status, note, new Date().toISOString()))
              setNotice(`${id} status updated to ${status}.`)
            }}
          />
        )}

        {page === 'video' && mode === 'admin' && (
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
              setNotice('Three patrol baseline observations added to the verification queue.')
            }}
            onAddRealObservations={newObs => {
              setStore(s => ({ ...s, observations: [...newObs, ...s.observations] }))
              setNotice(
                `${newObs.length} extracted video frame observation(s) added to the verification queue.`
              )
              setPage('review')
            }}
            onReview={() => setPage('review')}
          />
        )}

        {page === 'overview' && (
          <>
            <div className="overview-action">
              <button
                className="button button-primary"
                onClick={() => setPage(mode === 'admin' ? 'review' : 'submit')}
              >
                {mode === 'admin'
                  ? `Review ${pending} Pending Submission${pending === 1 ? '' : 's'}`
                  : '+ Submit a Report'}
              </button>
              {mode === 'admin' && (
                <button
                  type="button"
                  className="button button-outline"
                  onClick={() => {
                    const fresh = resetPersistedStore()
                    setStore(fresh)
                    setNotice('Local municipal store reset to baseline seed records.')
                  }}
                >
                  Reset Baseline Store
                </button>
              )}
            </div>

            <div className="summary-grid">
              {categories.map(category => (
                <div className="summary-card" key={category}>
                  <div className="summary-top">
                    <span>{category.toUpperCase()} INCIDENTS</span>
                    <span className="summary-icon">
                      <CivicIcon name={category === 'Trash' ? 'trash' : 'pothole'} size={17} />
                    </span>
                  </div>
                  <div className="summary-value">
                    {ranked.filter(i => i.category === category).length}
                  </div>
                  <div className="summary-note">
                    {category === 'Trash'
                      ? 'Model A · Grounding DINO-T'
                      : 'Model B · RF-DETR Nano Cell 8A'}
                  </div>
                </div>
              ))}
              <div className="summary-card">
                <div className="summary-top">
                  <span>{mode === 'admin' ? 'AWAITING VERIFICATION' : 'MY SUBMISSIONS'}</span>
                  <span className="summary-icon">
                    <CivicIcon name="check" size={17} />
                  </span>
                </div>
                <div className="summary-value">
                  {
                    store.observations.filter(o =>
                      mode === 'admin' ? o.review === 'Pending review' : o.reporter === residentId
                    ).length
                  }
                </div>
                <div className="summary-note">
                  {mode === 'admin' ? 'Queued for barangay review' : 'Tracked in Resident Portal'}
                </div>
              </div>
              <div className="summary-card summary-emphasis">
                <div className="summary-top">
                  <span>VALIDATED OBSERVATIONS</span>
                  <span className="summary-icon">
                    <CivicIcon name="shield" size={17} />
                  </span>
                </div>
                <div className="summary-value">
                  {store.observations.filter(o => o.review === 'Validated').length}
                </div>
                <div className="summary-note">Active &amp; historical verified reports</div>
              </div>
            </div>

            {/* Section 2.2: Unified Barangay Hotspot Leaderboard */}
            <HotspotLeaderboard
              store={store}
              mode={mode}
              onReportHere={loc => {
                setReportLocationPrefill(loc)
                setMode('user')
                setPage('submit')
              }}
              onInspectIncidents={() => {
                setMode('admin')
                setPage('incidents')
              }}
            />

            <div className="workflow-grid">
              <section className="panel">
                <div className="panel-heading">
                  <div>
                    <div className="section-index">02 / FREQUENCY-BASED PRIORITIZATION</div>
                    <h2>Prioritized Active Incidents</h2>
                    <p>Ranked by validated observation frequency; recency breaks ties.</p>
                  </div>
                </div>
                {ranked.map((incident, index) => (
                  <div className="priority-item" key={incident.id}>
                    <span className="rank-box">{index + 1}</span>
                    <div>
                      <strong>{incident.category}</strong>
                      <p>{incident.location}</p>
                      <small>Latest observation: {formatDate(incident.latest)}</small>
                    </div>
                    <div className="frequency">
                      <strong>{incident.frequency}</strong>
                      <small>observations</small>
                    </div>
                    <Status value={incident.status} />
                  </div>
                ))}
                {!ranked.length && (
                  <div className="empty-state">No active validated incidents.</div>
                )}
              </section>

              <section className="panel">
                <div className="panel-heading">
                  <div>
                    <div className="section-index">03 / DESCRIPTIVE TRENDS</div>
                    <h2>Validated Observations by Category</h2>
                    <p>Includes historical observations from resolved incidents.</p>
                  </div>
                </div>
                <div className="trend-bars trend-bars-vertical">
                  {categories.map(category => {
                    const count = store.observations.filter(
                      o => o.review === 'Validated' && o.category === category
                    ).length
                    const total = store.observations.filter(
                      o => o.review === 'Validated'
                    ).length
                    return (
                      <div key={category}>
                        <p>
                          <strong>{category}</strong> · {count} verified observation
                          {count === 1 ? '' : 's'}
                        </p>
                        <div
                          className="progress-track"
                          role="img"
                          aria-label={`${category}: ${count} validated observations`}
                        >
                          <div style={{ width: `${total ? (count / total) * 100 : 0}%` }} />
                        </div>
                      </div>
                    )
                  })}
                </div>
                <p className="prototype-note">
                  AI detection scores assist barangay reviewers during triage. Resolved incidents remain in historical records and exit the active dispatch queue.
                </p>
              </section>
            </div>
          </>
        )}
      </main>

      <footer className="site-footer">
        <div className="page-width footer-inner">
          <span>BANTAI — BARANGAY AUTOMATED NETWORK FOR TRASH AND ANOMALY IDENTIFICATION</span>
          <span>BARANGAY SANTA ELENA · CITY OF MARIKINA</span>
        </div>
      </footer>
    </div>
  )
}
