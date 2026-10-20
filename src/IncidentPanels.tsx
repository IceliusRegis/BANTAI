import { useState } from 'react'
import { categories, locations, duplicateCandidates, incidentStatuses } from './incidents'
import type { Category, IncidentStatus, Observation, Store } from './incidents'

export function formatDate(value: string) { return new Date(value).toLocaleString('en-PH', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) }
export function Status({ value }: { value: string }) { return <span className={`incident-status status-${value.toLowerCase().replace(/ /g, '-')}`}>{value}</span> }
export function Evidence({ report }: { report: Observation }) {
  return report.photo ? <img className="evidence-image" src={report.photo} alt={`Submitted ${report.category.toLowerCase()} at ${report.location}`} /> : <div className="evidence-placeholder"><span aria-hidden="true">▧</span><strong>{report.source === 'Video observation' ? `Video observation · ${report.clipTime}` : 'Demo photo placeholder'}</strong><small>{report.category}</small></div>
}

export function MyReports({ store }: { store: Store }) {
  const reports = store.observations.filter(o => o.reporter === 'Demo resident').sort((a, b) => b.submittedAt.localeCompare(a.submittedAt))
  return <div className="report-cards">{reports.map(report => {
    const incident = store.incidents.find(i => i.id === report.incidentId)
    return <article className="panel report-card" key={report.id}><Evidence report={report} /><div className="report-card-body"><div className="report-meta"><span>{report.id}</span><Status value={incident?.status || report.review} /></div><h2>{report.category}</h2><p className="location-copy">{report.location}</p><p>{report.description}</p><small>Observed {formatDate(report.observedAt)} · {report.source}</small><div className="report-update"><strong>{incident ? `Admin update · ${incident.id}` : 'Review update'}</strong><p>{incident?.note || report.reviewNote || 'Your report is waiting for admin review. It does not affect priority rankings yet.'}</p>{incident && <small>Updated {formatDate(incident.updatedAt)} · {report.reviewNote}</small>}</div><div className="report-meta"><Status value={report.review} /><small>Detection: {report.category} · {report.confidence}% (simulated)</small></div></div></article>
  })}{!reports.length && <div className="panel empty-state">No reports yet. Submit your first observation to get started.</div>}</div>
}

function ReviewCard({ store, report, onReview }: { store: Store; report: Observation; onReview: (id: string, decision: 'validate' | 'reject', target: string | undefined, note: string) => void }) {
  const candidates = duplicateCandidates(store, report)
  const [target, setTarget] = useState(candidates[0]?.id || '')
  const [note, setNote] = useState('')
  const [error, setError] = useState('')
  // An incident may be resolved while this observation is open in the review queue.
  const activeTarget = candidates.some(i => i.id === target) ? target : ''
  return <article className="panel review-card"><Evidence report={report} /><div className="review-body"><div className="report-meta"><span>{report.id} · {report.reporter}</span><Status value={report.review} /></div><h2>{report.category}</h2><p className="location-copy">{report.location}</p><p>{report.description}</p><small>Observed {formatDate(report.observedAt)} · Submitted {formatDate(report.submittedAt)} · {report.source}</small>
    <div className="detection-line"><strong>Simulated detection</strong><span>{report.category} · {report.confidence}% confidence</span><small>Confidence is displayed for review; it does not affect priority.</small></div>
    {candidates.length > 0 && <p className="match-note">Possible duplicate: {candidates.length} active incident{candidates.length === 1 ? '' : 's'} with the same category and location. Confirm the same physical issue before linking.</p>}
    <label className="form-field">Consolidate this observation<select value={activeTarget} onChange={e => setTarget(e.target.value)}><option value="">Create a separate incident</option>{candidates.map(i => <option key={i.id} value={i.id}>Link to {i.id} · {i.category} · {i.status}</option>)}</select></label>
    <label className="form-field">Review note <span className="muted">(required when rejecting)</span><textarea maxLength={600} rows={2} value={note} onChange={e => setNote(e.target.value)} placeholder="Explain the decision to the reporter" /></label>
    {error && <p className="form-error" role="alert">{error}</p>}
    <div className="review-actions"><button className="button button-outline" onClick={() => { if (!note.trim()) { setError('Add a reason before rejecting this report.'); return } onReview(report.id, 'reject', undefined, note) }}>Reject report</button><button className="button button-primary" onClick={() => onReview(report.id, 'validate', activeTarget || undefined, note)}>{activeTarget ? 'Validate & link observation' : 'Validate & create incident'}</button></div>
  </div></article>
}

export function ReviewQueue({ store, onReview }: { store: Store; onReview: (id: string, decision: 'validate' | 'reject', target: string | undefined, note: string) => void }) {
  const [filter, setFilter] = useState('All categories')
  const pending = store.observations.filter(o => o.review === 'Pending review' && (filter === 'All categories' || o.category === filter))
  return <><div className="list-toolbar"><strong>{pending.length} observations awaiting review</strong><label className="form-field">Category<select value={filter} onChange={e => setFilter(e.target.value)}><option>All categories</option>{categories.map(c => <option key={c}>{c}</option>)}</select></label></div><div className="report-cards">{pending.map(report => <ReviewCard key={`${report.id}-${duplicateCandidates(store, report).map(i => i.id).join('-')}`} store={store} report={report} onReview={onReview} />)}{!pending.length && <div className="panel empty-state"><h2>All caught up</h2><p>No pending observations match this category. Switch to Regular user and submit a report to try the full workflow.</p></div>}</div></>
}

function IncidentEditor({ store, id, onUpdate }: { store: Store; id: string; onUpdate: (id: string, status: IncidentStatus, note: string) => void }) {
  const incident = store.incidents.find(i => i.id === id)!
  const [status, setStatus] = useState(incident.status)
  const [note, setNote] = useState(incident.note)
  const evidence = store.observations.filter(o => o.incidentId === id && o.review === 'Validated')
  return <details className="panel incident-editor"><summary><span><strong>{incident.category}</strong><small>{id} · {incident.location}</small></span><span className="incident-summary-end"><span>{evidence.length} observations</span><Status value={incident.status} /><span aria-hidden="true">⌄</span></span></summary><div className="incident-editor-body"><form onSubmit={e => { e.preventDefault(); if (note.trim()) onUpdate(id, status, note) }}><div className="form-grid"><label className="form-field">Incident status<select value={status} onChange={e => setStatus(e.target.value as IncidentStatus)}>{incidentStatuses.map(s => <option key={s}>{s}</option>)}</select></label><label className="form-field">Update for the reporter<textarea required maxLength={600} rows={2} value={note} onChange={e => setNote(e.target.value)} /></label></div><button className="button button-primary" disabled={!note.trim()}>Save status update</button></form><h3>Consolidated evidence</h3><div className="evidence-list">{evidence.map(o => <div className="evidence-entry" key={o.id}><Evidence report={o} /><div><strong>{o.id}</strong><p>{o.description}</p><small>{o.source} · {formatDate(o.observedAt)} · {o.confidence}% simulated confidence</small></div></div>)}</div></div></details>
}

export function Incidents({ store, onUpdate }: { store: Store; onUpdate: (id: string, status: IncidentStatus, note: string) => void }) {
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState('All statuses')
  const incidents = store.incidents.filter(i => (filter === 'All statuses' || i.status === filter) && `${i.id} ${i.category} ${i.location}`.toLowerCase().includes(query.toLowerCase()))
  return <><div className="list-toolbar"><label className="form-field">Search incidents<input value={query} onChange={e => setQuery(e.target.value)} placeholder="Location, category, or incident ID" /></label><label className="form-field">Status<select value={filter} onChange={e => setFilter(e.target.value)}><option>All statuses</option>{incidentStatuses.map(s => <option key={s}>{s}</option>)}</select></label></div><div className="report-cards">{incidents.map(i => <IncidentEditor key={`${i.id}-${i.updatedAt}`} store={store} id={i.id} onUpdate={onUpdate} />)}{!incidents.length && <div className="panel empty-state">No incidents match your search.</div>}</div></>
}

export function VideoMonitor({ onGenerate, onReview }: { onGenerate: (location: string, category: Category) => void; onReview: () => void }) {
  const [location, setLocation] = useState(locations[0])
  const [category, setCategory] = useState<Category>('Garbage pile')
  const [generated, setGenerated] = useState(false)
  return <div className="submit-layout"><section className="panel"><div className="panel-heading"><div><div className="section-index">PRERECORDED CCTV WORKFLOW</div><h2>Video observation simulator</h2><p>A storyboard demonstrates three observation moments from one clip.</p></div></div><div className="form-body"><div className="video-storyboard">{['00:04', '00:12', '00:23'].map((time, i) => <div key={time}><span className="camera-label">DEMO CAMERA 01</span><div className="video-road"><span className={`anomaly-box anomaly-${i}`}>{category}</span></div><strong>{time}</strong><small>Illustrative frame</small></div>)}</div><div className="form-grid"><label className="form-field">Camera location<select disabled={generated} value={location} onChange={e => setLocation(e.target.value)}>{locations.map(l => <option key={l}>{l}</option>)}</select></label><label className="form-field">Simulated detected category<select disabled={generated} value={category} onChange={e => setCategory(e.target.value as Category)}>{categories.map(c => <option key={c}>{c}</option>)}</select></label></div><div className="form-actions"><span>{generated ? '3 observations sent to the admin review queue.' : 'Three frames → three pending observations'}</span>{generated ? <button className="button button-primary" onClick={onReview}>Review video observations</button> : <button className="button button-primary" onClick={() => { onGenerate(location, category); setGenerated(true) }}>Simulate clip processing</button>}</div>{generated && <button className="text-button" onClick={() => setGenerated(false)}>Set up another clip</button>}</div></section><aside className="panel guidance-panel"><div className="section-index">OBSERVE → VALIDATE → CONSOLIDATE</div><h2>Multiple views, one incident</h2><p>Review the generated observations and link matching frames to the same incident. Each validated observation increases its frequency once.</p><p>Separate physical issues can be kept as separate incidents, even when they share a category and location.</p><p className="prototype-note">This is a storyboard simulation. No CCTV feed, video decoding, or pretrained model is connected. Processing time and detection performance have not been measured.</p></aside></div>
}
