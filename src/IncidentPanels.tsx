import { useState, useRef } from 'react'
import { categories, locations, duplicateCandidates, incidentStatuses } from './incidents'
import type { Category, IncidentStatus, Observation, Store, TrashRegion, RoadDefect, UnifiedIncidentResponse } from './incidents'

export function formatDate(value: string) { return new Date(value).toLocaleString('en-PH', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) }
export function Status({ value }: { value: string }) { return <span className={`incident-status status-${value.toLowerCase().replace(/ /g, '-')}`}>{value}</span> }

export function Evidence({ report }: { report: Observation }) {
  return report.photo ? (
    <img className="evidence-image" src={report.photo} alt={`Submitted ${report.category.toLowerCase()} at ${report.location}`} />
  ) : (
    <div className="evidence-placeholder">
      <span aria-hidden="true">▧</span>
      <strong>{report.source === 'Video observation' ? `Video observation · ${report.clipTime || 'Frame'}` : 'Demo photo placeholder'}</strong>
      <small>{report.category}</small>
    </div>
  )
}

function RegionBadge({ region }: { region: TrashRegion }) {
  const isIndividual = region.category === 'individual_litter'
  return (
    <div className="region-chip" style={{ margin: '4px 0', padding: '6px 10px', background: 'rgba(235, 94, 40, 0.08)', borderRadius: '6px', borderLeft: '3px solid #eb5e28' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <strong>{region.display_label}</strong>
        {isIndividual && region.model_score !== null ? (
          <span style={{ fontSize: '12px', fontWeight: 600 }}>{(region.model_score * 100).toFixed(0)}% conf</span>
        ) : (
          <span style={{ fontSize: '12px', fontWeight: 600, color: '#eb5e28' }}>{region.item_count} items</span>
        )}
      </div>
      {region.category_breakdown && Object.keys(region.category_breakdown).length > 0 && (
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '4px' }}>
          {Object.entries(region.category_breakdown).map(([cat, count]) => (
            <span key={cat} style={{ fontSize: '11px', background: '#fff', border: '1px solid #ddd', padding: '1px 6px', borderRadius: '4px' }}>
              {cat}: {count}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function DefectBadge({ defect }: { defect: RoadDefect }) {
  const sevColor = defect.severity === 'high' ? '#d90429' : defect.severity === 'moderate' ? '#f77f00' : '#2b9348'
  return (
    <div className="defect-chip" style={{ margin: '4px 0', padding: '6px 10px', background: 'rgba(43, 147, 72, 0.08)', borderRadius: '6px', borderLeft: `3px solid ${sevColor}` }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <strong>Pothole Defect</strong>
        <span style={{ fontSize: '12px', fontWeight: 600, color: sevColor, textTransform: 'uppercase' }}>{defect.severity}</span>
      </div>
      <div style={{ fontSize: '12px', marginTop: '2px', display: 'flex', gap: '8px' }}>
        <span>Est. Diameter: ~{defect.estimated_diameter_cm} cm</span>
        <span>·</span>
        <span>Confidence: {(defect.confidence * 100).toFixed(0)}%</span>
      </div>
    </div>
  )
}

export function MyReports({ store }: { store: Store }) {
  const reports = store.observations.filter(o => o.reporter === 'Demo resident').sort((a, b) => b.submittedAt.localeCompare(a.submittedAt))
  return <div className="report-cards">{reports.map(report => {
    const incident = store.incidents.find(i => i.id === report.incidentId)
    return <article className="panel report-card" key={report.id}>
      <Evidence report={report} />
      <div className="report-card-body">
        <div className="report-meta">
          <span>{report.id}</span>
          <Status value={incident?.status || report.review} />
        </div>
        <h2>{report.category}</h2>
        <p className="location-copy">{report.location}</p>
        <p>{report.description}</p>
        <small>Observed {formatDate(report.observedAt)} · {report.source}</small>
        
        {/* Model A & Model B Findings */}
        {report.regions && report.regions.length > 0 && (
          <div style={{ marginTop: '8px' }}>
            <small style={{ fontWeight: 600, textTransform: 'uppercase', color: '#666' }}>Model A Trash Regions (D-112)</small>
            {report.regions.map(r => <RegionBadge key={r.region_id} region={r} />)}
          </div>
        )}
        {report.defects && report.defects.length > 0 && (
          <div style={{ marginTop: '8px' }}>
            <small style={{ fontWeight: 600, textTransform: 'uppercase', color: '#666' }}>Model B Pothole Defects</small>
            {report.defects.map(d => <DefectBadge key={d.defect_id} defect={d} />)}
          </div>
        )}
        {report.hotspotScore !== undefined && (
          <div style={{ marginTop: '6px', fontSize: '12px', fontWeight: 600 }}>
            <span>Municipal Hotspot Priority: </span>
            <span style={{ color: report.priorityLevel === 'CRITICAL' ? '#d90429' : report.priorityLevel === 'ELEVATED' ? '#f77f00' : '#2b9348' }}>
              {report.hotspotScore}/100 [{report.priorityLevel}]
            </span>
          </div>
        )}

        <div className="report-update">
          <strong>{incident ? `Admin update · ${incident.id}` : 'Review update'}</strong>
          <p>{incident?.note || report.reviewNote || 'Your report is waiting for admin review. It does not affect priority rankings yet.'}</p>
          {incident && <small>Updated {formatDate(incident.updatedAt)} · {report.reviewNote}</small>}
        </div>
        <div className="report-meta">
          <Status value={report.review} />
          {!report.regions && !report.defects && (
            <small>Detection: {report.category} · {report.confidence}% (simulated)</small>
          )}
        </div>
      </div>
    </article>
  })}{!reports.length && <div className="panel empty-state">No reports yet. Submit your first observation to get started.</div>}</div>
}

function ReviewCard({ store, report, onReview }: { store: Store; report: Observation; onReview: (id: string, decision: 'validate' | 'reject', target: string | undefined, note: string) => void }) {
  const candidates = duplicateCandidates(store, report)
  const [target, setTarget] = useState(candidates[0]?.id || '')
  const [note, setNote] = useState('')
  const [error, setError] = useState('')
  const activeTarget = candidates.some(i => i.id === target) ? target : ''

  return <article className="panel review-card">
    <Evidence report={report} />
    <div className="review-body">
      <div className="report-meta">
        <span>{report.id} · {report.reporter}</span>
        <Status value={report.review} />
      </div>
      <h2>{report.category}</h2>
      <p className="location-copy">{report.location}</p>
      <p>{report.description}</p>
      <small>Observed {formatDate(report.observedAt)} · Submitted {formatDate(report.submittedAt)} · {report.source}</small>
      
      {/* AI Model Findings */}
      {report.regions && report.regions.length > 0 ? (
        <div style={{ margin: '8px 0', padding: '8px 12px', background: '#f8f9fa', borderRadius: '6px' }}>
          <strong style={{ display: 'block', fontSize: '13px', marginBottom: '4px' }}>Model A: D-112 Spatial Grouping Findings</strong>
          {report.regions.map(r => <RegionBadge key={r.region_id} region={r} />)}
        </div>
      ) : null}

      {report.defects && report.defects.length > 0 ? (
        <div style={{ margin: '8px 0', padding: '8px 12px', background: '#f8f9fa', borderRadius: '6px' }}>
          <strong style={{ display: 'block', fontSize: '13px', marginBottom: '4px' }}>Model B: Road Defect Findings</strong>
          {report.defects.map(d => <DefectBadge key={d.defect_id} defect={d} />)}
        </div>
      ) : null}

      {report.hotspotScore !== undefined ? (
        <div style={{ margin: '6px 0', fontSize: '13px' }}>
          <strong>Section 6.3 Hotspot Priority Score: </strong>
          <span style={{ fontWeight: 700, color: report.priorityLevel === 'CRITICAL' ? '#d90429' : report.priorityLevel === 'ELEVATED' ? '#f77f00' : '#2b9348' }}>
            {report.hotspotScore}/100 [{report.priorityLevel}]
          </span>
        </div>
      ) : (
        <div className="detection-line">
          <strong>Simulated detection</strong>
          <span>{report.category} · {report.confidence}% confidence</span>
          <small>Confidence is displayed for review; it does not affect priority.</small>
        </div>
      )}

      {candidates.length > 0 && <p className="match-note">Possible duplicate: {candidates.length} active incident{candidates.length === 1 ? '' : 's'} with the same category and location. Confirm the same physical issue before linking.</p>}
      <label className="form-field">Consolidate this observation<select value={activeTarget} onChange={e => setTarget(e.target.value)}><option value="">Create a separate incident</option>{candidates.map(i => <option key={i.id} value={i.id}>Link to {i.id} · {i.category} · {i.status}</option>)}</select></label>
      <label className="form-field">Review note <span className="muted">(required when rejecting)</span><textarea maxLength={600} rows={2} value={note} onChange={e => setNote(e.target.value)} placeholder="Explain the decision to the reporter" /></label>
      {error && <p className="form-error" role="alert">{error}</p>}
      <div className="review-actions"><button className="button button-outline" onClick={() => { if (!note.trim()) { setError('Add a reason before rejecting this report.'); return } onReview(report.id, 'reject', undefined, note) }}>Reject report</button><button className="button button-primary" onClick={() => onReview(report.id, 'validate', activeTarget || undefined, note)}>{activeTarget ? 'Validate & link observation' : 'Validate & create incident'}</button></div>
    </div>
  </article>
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

export function VideoMonitor({
  onGenerate,
  onReview,
  onAddRealObservations,
}: {
  onGenerate: (location: string, category: Category) => void
  onReview: () => void
  onAddRealObservations?: (observations: Observation[]) => void
}) {
  const [location, setLocation] = useState(locations[0])
  const [category, setCategory] = useState<Category>('Garbage pile')
  const [generated, setGenerated] = useState(false)
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [processingVideo, setProcessingVideo] = useState(false)
  const [videoError, setVideoError] = useState('')
  const videoInputRef = useRef<HTMLInputElement>(null)

  async function handleRealVideoProcess() {
    if (!videoFile) return
    setProcessingVideo(true)
    setVideoError('')

    try {
      const formData = new FormData()
      formData.append('file', videoFile)
      formData.append('location', location)
      formData.append('sample_fps', '1.0')

      const res = await fetch('/api/v1/analyze', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`)
      }

      const result: UnifiedIncidentResponse = await res.json()

      if (onAddRealObservations && result.frames && result.frames.length > 0) {
        const now = new Date().toISOString()
        const newObs: Observation[] = result.frames.map((frame, idx) => {
          const timestamp = frame.timestamp_sec.toFixed(1)
          const regions = frame.trash_findings?.region_list || []
          const defects = frame.road_defect_findings?.defects || []
          const hasPile = regions.some(r => r.category === 'garbage_pile')
          const hasPothole = defects.length > 0

          let obsCat: Category = 'Garbage pile'
          if (hasPile) obsCat = 'Garbage pile'
          else if (hasPothole) obsCat = 'Pothole'
          else if (regions.length > 0) obsCat = 'Garbage pile'

          return {
            id: `VID-${result.incident_id.slice(-6)}-${idx + 1}`,
            source: 'Video observation',
            reporter: 'Patrol Camera 01',
            location,
            category: obsCat,
            observedAt: now,
            submittedAt: now,
            description: `1.0 FPS extraction frame ${frame.frame_index} (t=${timestamp}s). Single-pass dual-model detection.`,
            photo: frame.frame_url,
            clipTime: `00:${String(Math.floor(frame.timestamp_sec)).padStart(2, '0')}`,
            confidence: 90,
            review: 'Pending review',
            reviewNote: '',
            dualModelResult: result,
            regions,
            defects,
            hotspotScore: result.hotspot_score,
            priorityLevel: result.priority_level,
          }
        })

        onAddRealObservations(newObs)
        setGenerated(true)
      }
    } catch (err) {
      setVideoError(`Could not analyze video with backend server: ${err instanceof Error ? err.message : String(err)}. You can still run simulated clip processing below.`)
    } finally {
      setProcessingVideo(false)
    }
  }

  return (
    <div className="submit-layout">
      <section className="panel">
        <div className="panel-heading">
          <div>
            <div className="section-index">PATROL VIDEO INGESTION</div>
            <h2>Video observation & frame extractor</h2>
            <p>Upload a street patrol video for single-pass 1.0 FPS frame extraction or run the storyboard simulation.</p>
          </div>
        </div>
        <div className="form-body">
          {/* Real Video Upload Section */}
          <div style={{ marginBottom: '20px', padding: '16px', background: '#f8f9fa', borderRadius: '8px', border: '1px solid #e9ecef' }}>
            <strong style={{ display: 'block', marginBottom: '6px' }}>Upload Real Video (.mp4, .mov, .webm)</strong>
            <input
              ref={videoInputRef}
              type="file"
              accept="video/mp4,video/webm,video/quicktime"
              style={{ display: 'none' }}
              onChange={e => setVideoFile(e.target.files?.[0] || null)}
            />
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
              <button type="button" className="button button-outline" onClick={() => videoInputRef.current?.click()}>
                {videoFile ? videoFile.name : 'Choose Video File'}
              </button>
              {videoFile && (
                <button
                  type="button"
                  className="button button-primary"
                  disabled={processingVideo}
                  onClick={handleRealVideoProcess}
                >
                  {processingVideo ? 'Extracting Frames & Analyzing…' : 'Process Video (1.0 FPS Single-Pass)'}
                </button>
              )}
            </div>
            {videoError && <p className="form-error" style={{ marginTop: '8px' }}>{videoError}</p>}
          </div>

          <div className="video-storyboard">
            {['00:04', '00:12', '00:23'].map((time, i) => (
              <div key={time}>
                <span className="camera-label">DEMO CAMERA 01</span>
                <div className="video-road"><span className={`anomaly-box anomaly-${i}`}>{category}</span></div>
                <strong>{time}</strong>
                <small>1.0 FPS Extracted Frame</small>
              </div>
            ))}
          </div>

          <div className="form-grid">
            <label className="form-field">
              Camera location
              <select disabled={generated} value={location} onChange={e => setLocation(e.target.value)}>
                {locations.map(l => <option key={l}>{l}</option>)}
              </select>
            </label>
            <label className="form-field">
              Simulated detected category
              <select disabled={generated} value={category} onChange={e => setCategory(e.target.value as Category)}>
                {categories.map(c => <option key={c}>{c}</option>)}
              </select>
            </label>
          </div>

          <div className="form-actions">
            <span>{generated ? 'Observations sent to the admin review queue.' : 'Frames → pending observations'}</span>
            {generated ? (
              <button className="button button-primary" onClick={onReview}>Review video observations</button>
            ) : (
              <button
                className="button button-primary"
                onClick={() => {
                  onGenerate(location, category)
                  setGenerated(true)
                }}
              >
                Simulate clip processing
              </button>
            )}
          </div>
          {generated && <button className="text-button" onClick={() => setGenerated(false)}>Set up another clip</button>}
        </div>
      </section>

      <aside className="panel guidance-panel">
        <div className="section-index">SINGLE-PASS INGESTION</div>
        <h2>1.0 FPS Video Extraction</h2>
        <p>Uploaded video files are sampled ONCE at 1.0 FPS RGB JPEGs into a shared session directory. Both Model A (Trash Detector) and Model B (Road Defects) run on the exact same frames.</p>
        <p>Review the generated observations and link matching frames to the same incident. Each validated observation updates municipal priority frequency.</p>
        <p className="prototype-note">Model A 3-Model Consensus & Model B RF-DETR Nano run sequentially to preserve GPU VRAM.</p>
      </aside>
    </div>
  )
}
