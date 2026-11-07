import { useState, useRef } from 'react'
import { categories, locations, duplicateCandidates, incidentStatuses } from './incidents'
import type { Category, IncidentStatus, Observation, Store, TrashRegion, TrashDetection, RoadDefect, UnifiedIncidentResponse } from './incidents'

export function formatDate(value: string) { return new Date(value).toLocaleString('en-PH', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) }
export function Status({ value }: { value: string }) { return <span className={`incident-status status-${value.toLowerCase().replace(/ /g, '-')}`}>{value}</span> }

export function Evidence({ report, showBoxes = false }: { report: Observation; showBoxes?: boolean }) {
  if (!report.photo) {
    return (
      <div className="evidence-placeholder">
        <span aria-hidden="true">▧</span>
        <strong>{report.source === 'Video observation' ? `Video observation · ${report.clipTime || 'Frame'}` : 'Demo photo placeholder'}</strong>
        <small>{report.category}</small>
      </div>
    )
  }

  // If showBoxes is false or no detections/regions exist, render clean image only (Regular User view)
  const hasBoxes = showBoxes && ((report.detections && report.detections.length > 0) || (report.regions && report.regions.length > 0))
  if (!hasBoxes) {
    return <img className="evidence-image" src={report.photo} alt={`Submitted ${report.category.toLowerCase()} at ${report.location}`} />
  }

  // Admin View: Responsive SVG Bounding Box and Region Overlay
  const imgW = report.imageWidth || 1280
  const imgH = report.imageHeight || 720

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: '200px', maxHeight: '360px', background: '#eaf0ec', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
      <div style={{ position: 'relative', maxWidth: '100%', maxHeight: '100%', aspectRatio: `${imgW} / ${imgH}`, display: 'flex' }}>
        <img
          src={report.photo}
          alt={`Submitted ${report.category.toLowerCase()} at ${report.location}`}
          style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
        />
        <svg
          viewBox={`0 0 ${imgW} ${imgH}`}
          preserveAspectRatio="xMidYMid meet"
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
          }}
        >
          {/* 1. Multi-item Region Boxes (k >= 2) */}
          {report.regions?.filter(r => r.item_count >= 2).map((reg, idx) => {
            const [bx, by, bw, bh] = reg.bbox
            const isPile = reg.category === 'garbage_pile'
            const strokeColor = isPile ? '#ff8c00' : '#00b4d8'
            const bannerBg = isPile ? 'rgba(255, 140, 0, 0.95)' : 'rgba(0, 180, 216, 0.95)'
            const labelText = isPile ? `Garbage Pile (${reg.item_count} items)` : `Group of Individual Litter (${reg.item_count} items)`
            const fontSize = Math.max(16, Math.round(imgW * 0.016))
            const bannerPad = Math.round(fontSize * 0.4)
            const bannerH = fontSize + bannerPad * 2
            const textW = labelText.length * (fontSize * 0.58)

            return (
              <g key={`reg-${reg.region_id || idx}`}>
                <rect
                  x={bx - 4}
                  y={by - 4}
                  width={bw + 8}
                  height={bh + 8}
                  fill={isPile ? 'rgba(255, 140, 0, 0.08)' : 'rgba(0, 180, 216, 0.08)'}
                  stroke={strokeColor}
                  strokeWidth={Math.max(3, imgW * 0.003)}
                  strokeDasharray="8 6"
                  rx="4"
                />
                <rect
                  x={bx - 4}
                  y={Math.max(0, by - 4 - bannerH)}
                  width={textW + 16}
                  height={bannerH}
                  fill={bannerBg}
                  rx="3"
                />
                <text
                  x={bx - 4 + 8}
                  y={Math.max(0, by - 4 - bannerH) + fontSize + (bannerPad * 0.3)}
                  fill="#ffffff"
                  fontSize={fontSize}
                  fontWeight="bold"
                  fontFamily="system-ui, -apple-system, sans-serif"
                >
                  {labelText}
                </text>
              </g>
            )
          })}

          {/* 2. Individual Detection Boxes */}
          {report.detections?.map((det, idx) => {
            const [bx, by, bw, bh] = det.bbox
            let color = '#3cb44b' // Plastic: Green
            if (det.bantai_category === 'Potentially Hazardous') color = '#e6194b' // Crimson
            else if (det.bantai_category === 'Paper') color = '#d97706' // Amber/Gold
            else if (det.bantai_category === 'Metal') color = '#911eb4' // Purple
            else if (det.bantai_category === 'Organic') color = '#2b9348'

            const labelText = `${det.bantai_category} ${det.model_score.toFixed(2)}`
            const fontSize = Math.max(13, Math.round(imgW * 0.013))
            const padY = Math.round(fontSize * 0.3)
            const badgeH = fontSize + padY * 2
            const badgeW = labelText.length * (fontSize * 0.56) + 12
            const badgeY = Math.max(0, by - badgeH)

            return (
              <g key={`det-${det.detection_id || idx}`}>
                <rect
                  x={bx}
                  y={by}
                  width={bw}
                  height={bh}
                  fill="none"
                  stroke={color}
                  strokeWidth={Math.max(2, imgW * 0.002)}
                  rx="2"
                />
                <rect
                  x={bx}
                  y={badgeY}
                  width={badgeW}
                  height={badgeH}
                  fill={color}
                  rx="2"
                />
                <text
                  x={bx + 6}
                  y={badgeY + fontSize + (padY * 0.2)}
                  fill="#ffffff"
                  fontSize={fontSize}
                  fontWeight="600"
                  fontFamily="system-ui, -apple-system, sans-serif"
                >
                  {labelText}
                </text>
              </g>
            )
          })}
        </svg>
      </div>
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
      <Evidence report={report} showBoxes={false} />
      <div className="report-card-body">
        <div className="report-meta">
          <span>{report.id}</span>
          <Status value={incident?.status || report.review} />
        </div>
        <h2>{report.category}</h2>
        <p className="location-copy">{report.location}</p>
        <p>{report.description}</p>
        <small>Observed {formatDate(report.observedAt)} · {report.source}</small>
        
        {/* Model A Trash Findings */}
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
    <Evidence report={report} showBoxes={true} />
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
  return <details className="panel incident-editor"><summary><span><strong>{incident.category}</strong><small>{id} · {incident.location}</small></span><span className="incident-summary-end"><span>{evidence.length} observations</span><Status value={incident.status} /><span aria-hidden="true">⌄</span></span></summary><div className="incident-editor-body"><form onSubmit={e => { e.preventDefault(); if (note.trim()) onUpdate(id, status, note) }}><div className="form-grid"><label className="form-field">Incident status<select value={status} onChange={e => setStatus(e.target.value as IncidentStatus)}>{incidentStatuses.map(s => <option key={s}>{s}</option>)}</select></label><label className="form-field">Update for the reporter<textarea required maxLength={600} rows={2} value={note} onChange={e => setNote(e.target.value)} /></label></div><button className="button button-primary" disabled={!note.trim()}>Save status update</button></form><h3>Consolidated evidence</h3><div className="evidence-list">{evidence.map(o => <div className="evidence-entry" key={o.id}><Evidence report={o} showBoxes={true} /><div><strong>{o.id}</strong><p>{o.description}</p><small>{o.source} · {formatDate(o.observedAt)} · {o.confidence}% simulated confidence</small></div></div>)}</div></div></details>
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

          let obsCat: Category = 'Garbage pile'
          if (hasPile) obsCat = 'Garbage pile'
          else if (regions.length > 0) obsCat = 'Garbage pile'

          return {
            id: `VID-${result.incident_id.slice(-6)}-${idx + 1}`,
            source: 'Video observation',
            reporter: 'Patrol Camera 01',
            location,
            category: obsCat,
            observedAt: now,
            submittedAt: now,
            description: `1.0 FPS extraction frame ${frame.frame_index} (t=${timestamp}s). Single-pass Model A trash detection.`,
            photo: frame.frame_url,
            clipTime: `00:${String(Math.floor(frame.timestamp_sec)).padStart(2, '0')}`,
            confidence: 90,
            review: 'Pending review',
            reviewNote: '',
            dualModelResult: result,
            detections: frame.trash_findings?.detections || [],
            regions,
            imageWidth: frame.width || 1080,
            imageHeight: frame.height || 1920,
            defects,
            hotspotScore: (frame as any).frame_hotspot_score ?? result.hotspot_score,
            priorityLevel: (frame as any).frame_priority_level ?? result.priority_level,
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

  return <section className="panel"><div className="panel-heading"><div><div className="section-index">CAMERA MONITORING</div><h2>Video simulation & media detection</h2><p>Process a real video (.mp4) through Model A or simulate automated patrol captures.</p></div></div>
    <div className="form-body">
      {/* Real Video Upload Section */}
      <div style={{ padding: '16px', background: 'rgba(43, 147, 72, 0.05)', borderRadius: '8px', border: '1px solid #b7e4c7', marginBottom: '20px' }}>
        <strong style={{ display: 'block', fontSize: '15px', marginBottom: '4px', color: '#1b4332' }}>Run Live Video Detection (Model A)</strong>
        <p style={{ fontSize: '13px', color: '#40916c', marginBottom: '12px' }}>Upload a street patrol video (e.g., Motion_Extract_Testing_2.mp4). The backend extracts 1.0 FPS frames, runs 3-model consensus + D-112 spatial grouping, and adds confirmed hazard frames to the Review queue.</p>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
          <input
            ref={videoInputRef}
            type="file"
            accept="video/mp4,video/quicktime,video/webm"
            style={{ display: 'none' }}
            onChange={e => setVideoFile(e.target.files?.[0] || null)}
          />
          <button type="button" className="button button-outline" onClick={() => videoInputRef.current?.click()}>
            {videoFile ? `Selected: ${videoFile.name}` : 'Choose video file (.mp4)'}
          </button>
          <button
            type="button"
            className="button button-primary"
            disabled={!videoFile || processingVideo}
            onClick={handleRealVideoProcess}
          >
            {processingVideo ? 'Extracting & Running Model A…' : 'Process Video with AI Model'}
          </button>
        </div>
        {videoError && <p className="form-error" style={{ marginTop: '8px' }}>{videoError}</p>}
      </div>

      <div className="field-label">Simulated Camera Location</div>
      <label className="form-field"><select value={location} onChange={e => setLocation(e.target.value)}>{locations.map(l => <option key={l}>{l}</option>)}</select></label>
      <div className="field-label">Issue Category</div>
      <label className="form-field"><select value={category} onChange={e => setCategory(e.target.value as Category)}>{categories.map(c => <option key={c}>{c}</option>)}</select></label>
      <div className="video-storyboard"><div><span>00:04</span><strong>Frame capture</strong><small>Wide angle scan</small></div><div><span>00:12</span><strong>Anomaly flagged</strong><small>Model threshold passed</small></div><div><span>00:23</span><strong>Scene verified</strong><small>Added to review queue</small></div></div>
      <div className="review-actions"><button className="button button-primary" onClick={() => { onGenerate(location, category); setGenerated(true) }}>Generate simulated batch</button>{generated && <button className="button button-outline" onClick={onReview}>Go to review queue</button>}</div>
    </div>
  </section>
}
