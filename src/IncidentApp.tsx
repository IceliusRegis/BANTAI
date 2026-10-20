import { useState } from 'react'
import { seedStore, rankIncidents, rankHotspots, categories, residentId, addResidentReport, reviewReport, updateIncident, addVideoObservations } from './incidents'
import type { Store } from './incidents'
import ReportForm from './ReportForm'
import { MyReports, ReviewQueue, Incidents, VideoMonitor, Status, formatDate } from './IncidentPanels'
import './workflow.css'

export default function IncidentApp() {
  const [mode, setMode] = useState<'user' | 'admin'>('user')
  const [store, setStore] = useState<Store>(() => seedStore())
  const [page, setPage] = useState('overview')
  const [notice, setNotice] = useState('')
  const pending = store.observations.filter(o => o.review === 'Pending review').length
  const navigation = mode === 'user' ? [['overview', 'Overview'], ['submit', 'Submit report'], ['mine', 'My reports']] : [['overview', 'Dashboard'], ['review', `Review queue (${pending})`], ['incidents', 'Incident records'], ['video', 'Video simulation']]
  function switchMode(next: 'user' | 'admin') { setMode(next); setPage(next === 'admin' ? 'review' : 'mine'); setNotice('') }
  const ranked = rankIncidents(store)
  const hotspots = rankHotspots(store)
  return <div className="app-shell workflow">
    <header className="site-header"><div className="page-width header-top">
      <div><div className="brand-name">BARANGAY <span>INCIDENT MONITOR</span></div><div className="brand-subtitle">ENVIRONMENT & ROAD INFRASTRUCTURE</div></div>
      <div className="header-right"><button aria-pressed={mode === 'user'} className={`button ${mode === 'user' ? 'button-primary' : 'button-outline'}`} onClick={() => switchMode('user')}>Regular user</button><button aria-pressed={mode === 'admin'} className={`button ${mode === 'admin' ? 'button-primary' : 'button-outline'}`} onClick={() => switchMode('admin')}>Admin</button></div>
    </div><nav className="nav-bar" aria-label="Main navigation"><div className="page-width nav-inner">{navigation.map(([key, label]) => <button key={key} className={`nav-item ${page === key ? 'active' : ''}`} aria-current={page === key ? 'page' : undefined} onClick={() => { setPage(key); setNotice('') }}>{label}</button>)}</div></nav></header>
    <main className="page-width main-content">
      <div className="eyebrow">{mode === 'admin' ? 'BARANGAY OPERATIONS' : 'RESIDENT PORTAL'} / INTERACTIVE WIREFRAME</div>
      <h1>{page === 'overview' ? (mode === 'admin' ? 'Incident management' : 'A better barangay starts with a report.') : page === 'mine' ? 'My reports' : page === 'review' ? 'Review submissions' : page === 'incidents' ? 'Incident records' : page === 'video' ? 'Video monitoring simulation' : 'Report an issue'}</h1>
      <p>Report garbage piles, potholes, and road surface damage. Follow each report from review to resolution.</p>
      <div className="dashboard-note">Demo mode switch · Both roles share this session. Detection is simulated. Refreshing resets the demo.</div>
      {notice && <div className="notice" role="status">{notice}</div>}
      {page === 'submit' && mode === 'user' && <ReportForm onSubmit={report => { const id = `OBS-${crypto.randomUUID().slice(0, 8).toUpperCase()}`; setStore(s => addResidentReport(s, report, id, new Date().toISOString())); setPage('mine'); setNotice(`${id} submitted. Switch to Admin to review it.`) }} />}
      {page === 'mine' && mode === 'user' && <MyReports store={store} />}
      {page === 'review' && mode === 'admin' && <ReviewQueue store={store} onReview={(id, decision, target, note) => { setStore(s => reviewReport(s, id, decision, target, note, new Date().toISOString(), `INC-${crypto.randomUUID().slice(0, 8).toUpperCase()}`)); setNotice(decision === 'reject' ? 'Report rejected. The reporter can see your explanation.' : 'Report validated. Incident frequency and hotspot rankings updated.') }} />}
      {page === 'incidents' && mode === 'admin' && <Incidents store={store} onUpdate={(id, status, note) => { setStore(s => updateIncident(s, id, status, note, new Date().toISOString())); setNotice(`${id} updated to ${status}. Switch to Regular user to see the update.`) }} />}
      {page === 'video' && mode === 'admin' && <VideoMonitor onGenerate={(location, category) => { setStore(s => addVideoObservations(s, location, category, new Date().toISOString(), crypto.randomUUID().slice(0, 6))); setNotice('Three video observations added to the review queue.') }} onReview={() => setPage('review')} />}
      {page === 'overview' && <>
      <div className="overview-action"><button className="button button-primary" onClick={() => setPage(mode === 'admin' ? 'review' : 'submit')}>{mode === 'admin' ? `Review ${pending} submissions` : '+ Submit a report'}</button></div>
      <div className="summary-grid">
        {categories.map(category => <div className="summary-card" key={category}><div className="summary-top">{category}</div><div className="summary-value">{ranked.filter(i => i.category === category).length}</div><div className="summary-note">Active incidents</div></div>)}
        <div className="summary-card"><div className="summary-top">{mode === 'admin' ? 'Awaiting review' : 'My reports'}</div><div className="summary-value">{store.observations.filter(o => mode === 'admin' ? o.review === 'Pending review' : o.reporter === residentId).length}</div></div>
      </div>
      <div className="workflow-grid"><section className="panel"><div className="panel-heading"><div><div className="section-index">01 / FREQUENCY-BASED PRIORITIZATION</div><h2>Prioritized incidents</h2><p>Validated observations first. Recency breaks ties.</p></div></div>{ranked.map((incident, index) => <div className="priority-item" key={incident.id}><span className="rank-box">{index + 1}</span><div><strong>{incident.category}</strong><p>{incident.location}</p><small>Latest: {formatDate(incident.latest)}</small></div><div className="frequency"><strong>{incident.frequency}</strong><small>observations</small></div><Status value={incident.status} /></div>)}{!ranked.length && <div className="empty-state">No active validated incidents.</div>}</section>
      <section className="panel"><div className="panel-heading"><div><div className="section-index">02 / LOCATION PRIORITIES</div><h2>Priority hotspots</h2><p>Validated observations across active incidents.</p></div></div>{hotspots.map((spot, index) => <div className="hotspot-row" key={spot.location}><span className="rank-box">{index + 1}</span><div><strong>{spot.location}</strong><p>{spot.frequency} observations · {spot.incidents} active incidents</p></div></div>)}<p className="prototype-note">Detection confidence is for review only. Resolved incidents stay in records and leave the active priority list.</p></section></div>
      <section className="panel trends-panel"><div className="panel-heading"><div><div className="section-index">03 / DESCRIPTIVE TRENDS</div><h2>Validated observations by category</h2><p>Includes historical observations from resolved incidents.</p></div></div><div className="trend-bars">{categories.map(category => { const count = store.observations.filter(o => o.review === 'Validated' && o.category === category).length; const total = store.observations.filter(o => o.review === 'Validated').length; return <div key={category}><p><strong>{category}</strong> · {count} observations</p><div className="progress-track" role="img" aria-label={`${category}: ${count} validated observations`}><div style={{ width: `${total ? count / total * 100 : 0}%` }} /></div></div> })}</div></section>
      {mode === 'admin' && <p className="prototype-note">Evaluation pending: per-category detection performance, consolidation accuracy, ranking consistency, processing time, usability, and basic accessibility require labeled data and user testing.</p>}
      </>}
    </main>
  </div>
}
