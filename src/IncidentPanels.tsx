import { useState, useRef } from 'react'
import { categories, locations, duplicateCandidates, incidentStatuses, rankHotspots } from './incidents'
import type { Category, IncidentStatus, Observation, Store, UiDetectionBox, UiVideoFrame, UnifiedIncidentResponse } from './incidents'

export function formatDate(value: string) {
  return new Date(value).toLocaleString('en-PH', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
}

export function Status({ value }: { value: string }) {
  return <PlainStatus value={value} />
}

export function PlainStatus({ value }: { value: string }) {
  const map: Record<string, { label: string; colorClass: string }> = {
    'Pending review': { label: '🟡 Waiting for Check', colorClass: 'status-pending-review' },
    'Open': { label: '🔵 Approved — For Inspection', colorClass: 'status-open' },
    'For inspection': { label: '🔵 Approved — For Inspection', colorClass: 'status-for-inspection' },
    'In progress': { label: '🟣 Crew Working on It', colorClass: 'status-in-progress' },
    'Resolved': { label: '🟢 Fixed & Cleaned!', colorClass: 'status-resolved' },
    'Rejected': { label: '⚪ Declined', colorClass: 'status-rejected' },
  }
  const item = map[value] || { label: value, colorClass: 'status-open' }
  return <span className={`incident-status ${item.colorClass}`}>{item.label}</span>
}

export function Evidence({ report }: { report: Observation }) {
  return (
    <MediaDetectionViewer
      photo={report.photo}
      mediaType={report.mediaType}
      boxes={report.boxes}
      frames={report.frames}
      potholeCount={report.potholeCount}
      trashCount={report.trashCount}
      previewVideoUrl={report.previewVideoUrl}
    />
  )
}

/**
 * Universal Bounding Box & Frame-by-Frame Inspector.
 * Renders on Pictures, Videos (with stepper and preview), and Fallback Placeholders.
 */
export function MediaDetectionViewer({
  photo,
  mediaType = 'image',
  boxes = [],
  frames = [],
  potholeCount,
  trashCount,
  previewVideoUrl,
}: {
  photo?: string
  mediaType?: 'image' | 'video'
  boxes?: UiDetectionBox[]
  frames?: UiVideoFrame[]
  potholeCount?: number
  trashCount?: number
  previewVideoUrl?: string
}) {
  const [showBoxes, setShowBoxes] = useState(true)
  const [selectedFrameIdx, setSelectedFrameIdx] = useState(0)
  const [isPlayingPreview, setIsPlayingPreview] = useState(false)

  const isVideo = mediaType === 'video' || (frames && frames.length > 0)
  const currentFrame = isVideo && frames && frames.length > 0 ? frames[selectedFrameIdx] : null
  const currentPhoto = currentFrame ? currentFrame.frameUrl : photo
  const rawBoxes = currentFrame ? currentFrame.boxes : boxes
  const currentBoxes = (rawBoxes || []).filter(b => b.kind === 'pothole')

  const currentPotholes = currentFrame ? currentFrame.potholeCount : (potholeCount ?? currentBoxes.length)

  return (
    <div className="media-detection-viewer">
      {/* 1. Main Viewing Stage */}
      <div className="bbox-stage">
        {isPlayingPreview && previewVideoUrl ? (
          <video
            src={previewVideoUrl}
            controls
            autoPlay
            loop
            style={{ width: '100%', maxHeight: '460px', background: '#000', display: 'block' }}
          />
        ) : currentPhoto ? (
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', width: '100%', background: '#0b1512' }}>
            <div style={{ position: 'relative', display: 'inline-block', lineHeight: 0, maxWidth: '100%' }}>
              <img
                src={currentPhoto}
                alt="Scanned street evidence"
                style={{ maxWidth: '100%', maxHeight: '460px', width: 'auto', height: 'auto', display: 'block' }}
              />
              {showBoxes && currentBoxes.map(box => {
                const bPct = box.boxPct || (box as any).box_pct || { x: 20, y: 20, w: 30, h: 20 }
                const cPct = box.confidencePct ?? (box as any).confidence_pct ?? 90
                return (
                  <div
                    key={box.id}
                    className="bbox-rect bbox-pothole"
                    style={{
                      left: `${bPct.x}%`,
                      top: `${bPct.y}%`,
                      width: `${bPct.w}%`,
                      height: `${bPct.h}%`,
                    }}
                    title={`${box.label} (${cPct}% certainty)`}
                  >
                    <span className="bbox-tag tag-pothole">
                      🕳️ {box.label} · {cPct}%
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        ) : (
          <div className="evidence-placeholder" style={{ minHeight: '180px' }}>
            <span aria-hidden="true">📷</span>
            <strong>Media Preview</strong>
            <small>No image or video file attached</small>
          </div>
        )}
      </div>

      {/* 2. Non-Tech Grandma Amount Summary Bar */}
      <div className="amount-bar">
        <div className="amount-pills">
          {currentPotholes > 0 ? (
            <span className="amount-pill amount-pill-pothole">
              🕳️ <strong>Potholes Found: {currentPotholes}</strong>
            </span>
          ) : (
            <span className="amount-pill amount-pill-clean">
              ✅ <strong>Road Surface Clear (0 Potholes)</strong>
            </span>
          )}
        </div>

        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {currentBoxes && currentBoxes.length > 0 && !isPlayingPreview && (
            <button
              type="button"
              className="button button-outline"
              style={{ padding: '4px 10px', fontSize: '11px', height: '30px' }}
              onClick={() => setShowBoxes(!showBoxes)}
            >
              {showBoxes ? 'Hide Colored Boxes' : 'Show Colored Boxes'}
            </button>
          )}

          {previewVideoUrl && (
            <button
              type="button"
              className={`button ${isPlayingPreview ? 'button-outline' : 'button-primary'}`}
              style={{ padding: '4px 12px', fontSize: '11px', height: '30px', display: 'flex', alignItems: 'center', gap: '4px' }}
              onClick={() => setIsPlayingPreview(!isPlayingPreview)}
            >
              {isPlayingPreview ? '⏹ Stop Video Preview' : '🎬 Watch Video Preview'}
            </button>
          )}
        </div>
      </div>

      {/* 3. Single-Entry Video Frame-by-Frame Stepper */}
      {isVideo && frames && frames.length > 1 && !isPlayingPreview && (
        <div className="frame-stepper">
          <div className="frame-stepper-left">
            <button
              type="button"
              className="button button-outline"
              style={{ padding: '4px 8px', fontSize: '11px' }}
              disabled={selectedFrameIdx <= 0}
              onClick={() => setSelectedFrameIdx(selectedFrameIdx - 1)}
            >
              ◀ Prev Frame
            </button>
            <span style={{ fontSize: '12px', fontWeight: 700, color: '#333' }}>
              Frame {selectedFrameIdx + 1} of {frames.length} ({frames[selectedFrameIdx].clipTime})
            </span>
            <button
              type="button"
              className="button button-outline"
              style={{ padding: '4px 8px', fontSize: '11px' }}
              disabled={selectedFrameIdx >= frames.length - 1}
              onClick={() => setSelectedFrameIdx(selectedFrameIdx + 1)}
            >
              Next Frame ▶
            </button>
          </div>

          {/* Filmstrip thumbnails */}
          <div className="filmstrip-row">
            {frames.map((f, i) => (
              <button
                type="button"
                key={f.frameIndex}
                className={`filmstrip-thumb ${i === selectedFrameIdx ? 'active' : ''}`}
                onClick={() => setSelectedFrameIdx(i)}
                title={`Jump to frame ${i + 1}`}
              >
                {f.frameUrl ? <img src={f.frameUrl} alt={`Frame ${i + 1}`} /> : <span>{i + 1}</span>}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * Location Hotspots Board for Admin.
 * Ranks streets automatically based on validated potholes and trash piles.
 */
export function LocationHotspotsBoard({
  store,
  selectedStreet,
  onSelectStreet,
}: {
  store: Store
  selectedStreet: string
  onSelectStreet: (street: string) => void
}) {
  const hotspots = rankHotspots(store)
  return (
    <div style={{ marginBottom: '22px' }}>
      <div className="panel-heading" style={{ padding: '0 0 10px 0' }}>
        <div>
          <div className="section-index">📍 PRIORITY STREETS</div>
          <h2 style={{ fontSize: '18px', margin: '4px 0' }}>Location Hotspots</h2>
          <p style={{ fontSize: '13px', color: '#555' }}>
            Streets automatically ranked by where reported potholes and trash piles need municipal attention. Click any street card to filter the reports below.
          </p>
        </div>
      </div>
      <div className="hotspots-board">
        {hotspots.map((spot, idx) => {
          const isSelected = selectedStreet === spot.location
          const prioClass = spot.priorityLevel === 'Urgent' ? 'priority-urgent' : (spot.priorityLevel === 'High' ? 'priority-high' : 'priority-moderate')
          return (
            <div
              key={spot.location}
              className={`hotspot-card ${prioClass} ${isSelected ? 'selected' : ''}`}
              onClick={() => onSelectStreet(isSelected ? '' : spot.location)}
            >
              <div className="hotspot-card-top">
                <span className="hotspot-rank">RANK #{idx + 1}</span>
                <span className={`incident-status status-${spot.priorityLevel.toLowerCase()}`} style={{ fontSize: '10px', padding: '2px 6px' }}>
                  {spot.priorityLevel === 'Urgent' ? '🔴 Urgent' : spot.priorityLevel === 'High' ? '🟠 High Priority' : '🟢 Moderate'}
                </span>
              </div>
              <strong className="hotspot-street-name">{spot.location}</strong>
              <div className="hotspot-stats">
                <span>🕳️ {spot.potholesCount} Potholes</span>
                <span>·</span>
                <span>{spot.frequency} Reports</span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/**
 * Resident Portal: "My Sent Reports".
 * Shows only Grandma's own reports with status and detection details.
 */
export function MyReports({ store }: { store: Store }) {
  const reports = store.observations.filter(o => o.reporter === 'Demo resident').sort((a, b) => b.submittedAt.localeCompare(a.submittedAt))

  return (
    <div className="report-cards">
      {reports.map(report => {
        const incident = store.incidents.find(i => i.id === report.incidentId)
        return (
          <article className="panel report-card" key={report.id} style={{ gridTemplateColumns: '1fr', padding: '22px', gap: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '8px' }}>
              <div>
                <span style={{ fontSize: '12px', fontWeight: 700, color: '#087a72' }}>Report #{report.id}</span>
                <h2 style={{ fontSize: '20px', margin: '4px 0' }}>{report.category}</h2>
                <p style={{ margin: '2px 0', fontWeight: 600, color: '#222' }}>📍 {report.location}</p>
                <small style={{ color: '#666' }}>Submitted {formatDate(report.submittedAt)} · {report.source}</small>
              </div>
              <PlainStatus value={incident?.status || report.review} />
            </div>

            <Evidence report={report} />

            <div style={{ background: '#f8faf9', padding: '14px', borderRadius: '6px', border: '1px solid #dae5df' }}>
              <p style={{ margin: '0 0 8px 0', fontSize: '13px' }}><strong>Your Note:</strong> {report.description}</p>
              <div style={{ borderTop: '1px solid #e2ece6', paddingTop: '10px', marginTop: '10px' }}>
                <strong style={{ fontSize: '12px', color: '#087a72' }}>Barangay Update:</strong>
                <p style={{ margin: '4px 0 0 0', fontSize: '13px', color: '#333' }}>
                  {incident?.note || report.reviewNote || 'Your report is waiting for check by the barangay team.'}
                </p>
              </div>
            </div>
          </article>
        )
      })}
      {!reports.length && (
        <div className="panel empty-state">
          <h2>No reports sent yet</h2>
          <p>Click "Report a Problem" above to send your first photo or video!</p>
        </div>
      )}
    </div>
  )
}

function ReviewCard({
  store,
  report,
  onReview,
}: {
  store: Store
  report: Observation
  onReview: (id: string, decision: 'validate' | 'reject', target: string | undefined, note: string) => void
}) {
  const candidates = duplicateCandidates(store, report)
  const [target, setTarget] = useState(candidates[0]?.id || '')
  const [note, setNote] = useState('')
  const [error, setError] = useState('')
  const activeTarget = candidates.some(i => i.id === target) ? target : ''

  return (
    <article className="panel review-card" style={{ gridTemplateColumns: '1fr', gap: '16px', padding: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <span style={{ fontSize: '12px', fontWeight: 700, color: '#087a72' }}>{report.id} · {report.reporter}</span>
          <h2 style={{ fontSize: '20px', margin: '4px 0' }}>{report.category}</h2>
          <p style={{ margin: '2px 0', fontWeight: 600, color: '#333' }}>📍 {report.location}</p>
          <small style={{ color: '#666' }}>Observed {formatDate(report.observedAt)} · Submitted {formatDate(report.submittedAt)}</small>
        </div>
        <PlainStatus value={report.review} />
      </div>

      <MediaDetectionViewer
        photo={report.photo}
        mediaType={report.mediaType}
        boxes={report.boxes}
        frames={report.frames}
        potholeCount={report.potholeCount}
        trashCount={report.trashCount}
        previewVideoUrl={report.previewVideoUrl}
      />

      <div style={{ background: '#f8faf9', padding: '14px', borderRadius: '6px', border: '1px solid #dae5df' }}>
        <p style={{ margin: '0 0 10px 0', fontSize: '13px' }}><strong>Resident Note:</strong> {report.description}</p>

        {candidates.length > 0 && (
          <p className="match-note" style={{ margin: '8px 0' }}>
            ℹ️ Possible duplicate: {candidates.length} active incident{candidates.length === 1 ? '' : 's'} on this same street.
          </p>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '12px', marginTop: '10px' }}>
          <label className="form-field" style={{ margin: 0 }}>
            Street Incident Link
            <select value={activeTarget} onChange={e => setTarget(e.target.value)}>
              <option value="">Create new incident on this street</option>
              {candidates.map(i => (
                <option key={i.id} value={i.id}>
                  Link to {i.id} · {i.category} ({i.status})
                </option>
              ))}
            </select>
          </label>

          <label className="form-field" style={{ margin: 0 }}>
            Note for Resident (optional when approving)
            <input
              type="text"
              value={note}
              onChange={e => setNote(e.target.value)}
              placeholder="e.g. Approved. Maintenance crew scheduled."
            />
          </label>
        </div>

        {error && <p className="form-error" role="alert" style={{ marginTop: '8px' }}>{error}</p>}

        <div className="review-actions" style={{ marginTop: '14px', display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
          <button
            type="button"
            className="button button-outline"
            onClick={() => {
              if (!note.trim()) {
                setError('Please add a reason for the resident before declining.')
                return
              }
              onReview(report.id, 'reject', undefined, note)
            }}
          >
            ❌ Decline Report
          </button>

          <button
            type="button"
            className="button button-primary"
            style={{ fontWeight: 700 }}
            onClick={() => onReview(report.id, 'validate', activeTarget || undefined, note || 'Approved and added to barangay priority list.')}
          >
            ✅ {activeTarget ? 'Approve & Link to Incident' : 'Approve & Add to Street Hotspot'}
          </button>
        </div>
      </div>
    </article>
  )
}

export function ReviewQueue({
  store,
  onReview,
  selectedStreet = '',
  onSelectStreet,
}: {
  store: Store
  onReview: (id: string, decision: 'validate' | 'reject', target: string | undefined, note: string) => void
  selectedStreet?: string
  onSelectStreet?: (street: string) => void
}) {
  const [filter, setFilter] = useState('All categories')
  const pending = store.observations.filter(
    o => o.review === 'Pending review' &&
      (filter === 'All categories' || o.category === filter) &&
      (!selectedStreet || o.location === selectedStreet)
  )

  return (
    <>
      {onSelectStreet && (
        <LocationHotspotsBoard
          store={store}
          selectedStreet={selectedStreet}
          onSelectStreet={onSelectStreet}
        />
      )}

      <div className="list-toolbar">
        <div>
          <strong>{pending.length} incoming reports waiting for review</strong>
          {selectedStreet && (
            <span style={{ marginLeft: '10px', fontSize: '12px', color: '#087a72', fontWeight: 600 }}>
              Filtered to: {selectedStreet} ({' '}
              <button
                type="button"
                className="text-button"
                style={{ textDecoration: 'underline' }}
                onClick={() => onSelectStreet?.('')}
              >
                Clear filter
              </button>{' '}
              )
            </span>
          )}
        </div>
        <label className="form-field">
          Category
          <select value={filter} onChange={e => setFilter(e.target.value)}>
            <option>All categories</option>
            {categories.map(c => <option key={c}>{c}</option>)}
          </select>
        </label>
      </div>

      <div className="report-cards">
        {pending.map(report => (
          <ReviewCard
            key={report.id}
            store={store}
            report={report}
            onReview={onReview}
          />
        ))}
        {!pending.length && (
          <div className="panel empty-state">
            <h2>All caught up!</h2>
            <p>No reports currently waiting for review{selectedStreet ? ' on this street' : ''}.</p>
          </div>
        )}
      </div>
    </>
  )
}

function IncidentEditor({
  store,
  id,
  onUpdate,
}: {
  store: Store
  id: string
  onUpdate: (id: string, status: IncidentStatus, note: string) => void
}) {
  const incident = store.incidents.find(i => i.id === id)!
  const [status, setStatus] = useState(incident.status)
  const [note, setNote] = useState(incident.note)
  const evidence = store.observations.filter(o => o.incidentId === id && o.review === 'Validated')

  return (
    <details className="panel incident-editor">
      <summary>
        <span>
          <strong>{incident.category}</strong>
          <small>{id} · 📍 {incident.location}</small>
        </span>
        <span className="incident-summary-end">
          <span>{evidence.length} validated report{evidence.length === 1 ? '' : 's'}</span>
          <PlainStatus value={incident.status} />
          <span aria-hidden="true">⌄</span>
        </span>
      </summary>
      <div className="incident-editor-body">
        <form onSubmit={e => { e.preventDefault(); if (note.trim()) onUpdate(id, status, note) }}>
          <div className="form-grid">
            <label className="form-field">
              Change Street Status
              <select value={status} onChange={e => setStatus(e.target.value as IncidentStatus)}>
                {incidentStatuses.map(s => <option key={s}>{s}</option>)}
              </select>
            </label>
            <label className="form-field">
              Update Note for Residents
              <textarea required maxLength={600} rows={2} value={note} onChange={e => setNote(e.target.value)} />
            </label>
          </div>
          <button type="submit" className="button button-primary" disabled={!note.trim()}>
            Save Status Update
          </button>
        </form>

        <h3 style={{ marginTop: '20px' }}>Attached Evidence Photos & Scans</h3>
        <div className="evidence-list">
          {evidence.map(o => (
            <div className="evidence-entry" key={o.id} style={{ gridTemplateColumns: '140px 1fr' }}>
              <Evidence report={o} />
              <div>
                <strong>{o.id}</strong>
                <p>{o.description}</p>
                <small>{o.source} · {formatDate(o.observedAt)}</small>
              </div>
            </div>
          ))}
        </div>
      </div>
    </details>
  )
}

export function Incidents({
  store,
  onUpdate,
}: {
  store: Store
  onUpdate: (id: string, status: IncidentStatus, note: string) => void
}) {
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState('All statuses')
  const incidents = store.incidents.filter(
    i => (filter === 'All statuses' || i.status === filter) &&
      `${i.id} ${i.category} ${i.location}`.toLowerCase().includes(query.toLowerCase())
  )

  return (
    <>
      <div className="list-toolbar">
        <label className="form-field">
          Search Streets or Incidents
          <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Type a street name or incident ID..." />
        </label>
        <label className="form-field">
          Status
          <select value={filter} onChange={e => setFilter(e.target.value)}>
            <option>All statuses</option>
            {incidentStatuses.map(s => <option key={s}>{s}</option>)}
          </select>
        </label>
      </div>
      <div className="report-cards">
        {incidents.map(i => (
          <IncidentEditor key={`${i.id}-${i.updatedAt}`} store={store} id={i.id} onUpdate={onUpdate} />
        ))}
        {!incidents.length && (
          <div className="panel empty-state">No street records match your search.</div>
        )}
      </div>
    </>
  )
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
  const [category, setCategory] = useState<Category>('Pothole')
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
        const isPothole = category === 'Pothole'

        const singleObservation: Observation = {
          id: `VID-${result.incident_id.slice(-6)}-1`,
          source: 'Video observation',
          reporter: 'Street Patrol Camera 01',
          location,
          category,
          observedAt: now,
          submittedAt: now,
          description: `Street patrol video footage (${result.source_media.filename}). ${result.frames.length} key frames scanned for road hazards.`,
          photo: result.frames[0]?.frame_url,
          clipTime: '00:01',
          confidence: 93,
          review: 'Pending review',
          reviewNote: '',
          mediaType: 'video',
          frames: result.frames.map((frame, idx) => ({
            frameIndex: idx + 1,
            timestampSec: frame.timestamp_sec,
            clipTime: `00:${String(Math.floor(frame.timestamp_sec)).padStart(2, '0')}`,
            frameUrl: frame.frame_url,
            boxes: frame.boxes || [],
            potholeCount: frame.pothole_count || (frame.road_defect_findings?.defects?.length || 0),
            trashCount: frame.trash_count || (frame.trash_findings?.total_items || 0),
          })),
          boxes: result.frames[0]?.boxes || [],
          potholeCount: result.summary?.total_potholes || 0,
          trashCount: result.summary?.total_trash_items || 0,
          previewVideoUrl: result.preview_video_url || undefined,
          dualModelResult: result,
          hotspotScore: result.hotspot_score,
          priorityLevel: result.priority_level,
        }

        onAddRealObservations([singleObservation])
        setGenerated(true)
      }
    } catch (err) {
      setVideoError(`Could not scan video: ${err instanceof Error ? err.message : String(err)}. You can still run the simulated clip below.`)
    } finally {
      setProcessingVideo(false)
    }
  }

  return (
    <div className="submit-layout">
      <section className="panel">
        <div className="panel-heading">
          <div>
            <div className="section-index">STREET PATROL SCANNER</div>
            <h2>Video observation & frame extractor</h2>
            <p>Upload a patrol video for single-pass 1.0 FPS scanning or run the storyboard simulation.</p>
          </div>
        </div>
        <div className="form-body">
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
                  {processingVideo ? 'Scanning Frames & Generating Preview…' : 'Scan Video (1 Entry with Frame Stepper)'}
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
                <small>1.0 FPS Sample</small>
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
            <span>{generated ? 'Observation added to the review queue.' : 'Frames scanned → single review entry'}</span>
            {generated ? (
              <button type="button" className="button button-primary" onClick={onReview}>Review video observations</button>
            ) : (
              <button
                type="button"
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
          {generated && <button type="button" className="text-button" onClick={() => setGenerated(false)}>Set up another clip</button>}
        </div>
      </section>

      <aside className="panel guidance-panel">
        <div className="section-index">SINGLE-ENTRY SCANNER</div>
        <h2>1.0 FPS Video Extraction</h2>
        <p>Uploaded video files are sampled at 1.0 FPS and analyzed by the BANTAI Pothole Model (Bundle v5 + Cell 8A Vision Engine).</p>
        <p>The entire video is kept in <strong>ONE single review card</strong> with a frame stepper and an annotated video preview generator.</p>
      </aside>
    </div>
  )
}
