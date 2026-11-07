import { useState, useRef, useEffect, useMemo } from 'react'
import {
  categories,
  locations,
  duplicateCandidates,
  incidentStatuses,
  rankHotspots,
} from './incidents'
import type {
  Category,
  IncidentStatus,
  Observation,
  Store,
  UiDetectionBox,
  UiVideoFrame,
  UnifiedIncidentResponse,
} from './incidents'

export function formatDate(value: string) {
  return new Date(value).toLocaleString('en-PH', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

export function Status({ value }: { value: string }) {
  return <PlainStatus value={value} />
}

export function PlainStatus({ value }: { value: string }) {
  const map: Record<string, { label: string; colorClass: string }> = {
    'Pending review': { label: 'Waiting for Review', colorClass: 'status-pending-review' },
    'Open': { label: 'Approved — For Inspection', colorClass: 'status-open' },
    'For inspection': { label: 'Scheduled for Inspection', colorClass: 'status-for-inspection' },
    'In progress': { label: 'Maintenance Crew Dispatched', colorClass: 'status-in-progress' },
    'Resolved': { label: 'Completed & Resolved', colorClass: 'status-resolved' },
    'Rejected': { label: 'Declined', colorClass: 'status-rejected' },
  }
  const item = map[value] || { label: value, colorClass: 'status-open' }
  return <span className={`incident-status ${item.colorClass}`}>{item.label}</span>
}

export function Evidence({ report, showAdminOverlay = true }: { report: Observation; showAdminOverlay?: boolean }) {
  return (
    <MediaDetectionViewer
      photo={report.photo}
      mediaType={report.mediaType}
      boxes={report.boxes}
      frames={report.frames}
      potholeCount={report.potholeCount}
      trashCount={report.trashCount}
      previewVideoUrl={report.previewVideoUrl}
      regions={report.regions}
      detections={report.detections}
      imageWidth={report.imageWidth}
      imageHeight={report.imageHeight}
      showAdminOverlay={showAdminOverlay}
    />
  )
}

/**
 * Universal Dual-Model Bounding Box & Frame-by-Frame Inspector.
 * Integrates precision SVG overlays, responsive box percentages, and animated video playback.
 */
export function MediaDetectionViewer({
  photo,
  mediaType = 'image',
  boxes = [],
  frames = [],
  potholeCount,
  trashCount,
  previewVideoUrl,
  regions = [],
  detections = [],
  imageWidth,
  imageHeight,
  showAdminOverlay = true,
}: {
  photo?: string
  mediaType?: 'image' | 'video'
  boxes?: UiDetectionBox[]
  frames?: UiVideoFrame[]
  potholeCount?: number
  trashCount?: number
  previewVideoUrl?: string
  regions?: any[]
  detections?: any[]
  imageWidth?: number
  imageHeight?: number
  showAdminOverlay?: boolean
}) {
  const [showBoxes, setShowBoxes] = useState(true)
  const initialFrameIdx = useMemo(() => {
    if (!frames || frames.length === 0) return 0
    let bestIdx = 0
    let maxCount = -1
    frames.forEach((f, idx) => {
      const c = (f.trashCount || 0) + (f.potholeCount || 0)
      if (c > maxCount) {
        maxCount = c
        bestIdx = idx
      }
    })
    return bestIdx >= 0 ? bestIdx : 0
  }, [frames])

  const [selectedFrameIdx, setSelectedFrameIdx] = useState(initialFrameIdx)

  useEffect(() => {
    setSelectedFrameIdx(initialFrameIdx)
  }, [initialFrameIdx])

  const [isPlayingPreview, setIsPlayingPreview] = useState(false)

  const isVideo = mediaType === 'video' || (frames && frames.length > 0)
  const currentFrame = isVideo && frames && frames.length > 0 ? frames[selectedFrameIdx] : null
  const currentPhoto = currentFrame ? currentFrame.frameUrl : photo
  const currentBoxes = currentFrame ? (currentFrame.boxes || []) : (boxes || [])

  const currentPotholes = currentFrame
    ? (currentFrame.potholeCount ?? currentBoxes.filter(b => b.kind === 'pothole').length)
    : (potholeCount ?? currentBoxes.filter(b => b.kind === 'pothole').length)

  const currentTrash = currentFrame
    ? (currentFrame.trashCount ?? currentBoxes.filter(b => b.kind === 'trash').reduce((sum, b) => sum + (b.count || 1), 0))
    : (trashCount ?? currentBoxes.filter(b => b.kind === 'trash').reduce((sum, b) => sum + (b.count || 1), 0))

  const currentRegions = currentFrame?.regions || regions || []
  const currentDetections = currentFrame?.detections || detections || []
  const imgW = currentFrame?.width || imageWidth || 1280
  const imgH = currentFrame?.height || imageHeight || 720

  const hasSvgAnnotations =
    showAdminOverlay &&
    (currentRegions.length > 0 || currentDetections.length > 0)

  return (
    <div className="media-detection-viewer">
      {/* 1. Main Media Stage */}
      <div className="bbox-stage">
        {isPlayingPreview && previewVideoUrl ? (
          <video
            src={previewVideoUrl}
            controls
            autoPlay
            loop
            style={{ width: '100%', maxHeight: '460px', background: '#0b1512', display: 'block' }}
          />
        ) : currentPhoto ? (
          <div
            style={{
              position: 'relative',
              width: '100%',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              background: '#0b1512',
            }}
          >
            <div
              style={{
                position: 'relative',
                display: 'inline-block',
                lineHeight: 0,
                maxWidth: '100%',
                maxHeight: '460px',
              }}
            >
              <img
                src={currentPhoto}
                alt="Municipal street incident evidence"
                style={{
                  maxWidth: '100%',
                  maxHeight: '460px',
                  width: 'auto',
                  height: 'auto',
                  display: 'block',
                }}
              />

              {/* SVG Overlays for Model A Trash Regions & Detections (when available) */}
              {showBoxes && hasSvgAnnotations && (
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
                  {/* Multi-item Regions (k >= 2) */}
                  {currentRegions
                    .filter((r: any) => (r.item_count || 1) >= 2)
                    .map((reg: any, idx: number) => {
                      const [bx, by, bw, bh] = reg.bbox || [0, 0, 0, 0]
                      const isPile = reg.category === 'garbage_pile'
                      const strokeColor = isPile ? '#ff8c00' : '#00b4d8'
                      const bannerBg = isPile ? 'rgba(255, 140, 0, 0.95)' : 'rgba(0, 180, 216, 0.95)'
                      const labelText = isPile
                        ? `Garbage Pile (${reg.item_count} items)`
                        : `Group of Individual Litter (${reg.item_count} items)`
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
                            y={Math.max(0, by - 4 - bannerH) + fontSize + bannerPad * 0.3}
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

                  {/* Individual Item Detections */}
                  {currentDetections.map((det: any, idx: number) => {
                    const [bx, by, bw, bh] = det.bbox || [0, 0, 0, 0]
                    let color = '#3cb44b'
                    if (det.bantai_category === 'Potentially Hazardous') color = '#e6194b'
                    else if (det.bantai_category === 'Paper') color = '#d97706'
                    else if (det.bantai_category === 'Metal') color = '#911eb4'
                    else if (det.bantai_category === 'Organic') color = '#2b9348'

                    const scoreVal = typeof det.model_score === 'number' ? det.model_score.toFixed(2) : ''
                    const labelText = `${det.bantai_category || det.label} ${scoreVal}`.trim()
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
                          y={badgeY + fontSize + padY * 0.2}
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
              )}

              {/* HTML Box Overlay for Potholes & Standard Boxes */}
              {showBoxes &&
                currentBoxes.map(box => {
                  // If already rendered via SVG regions, avoid duplicate rendering of trash items
                  if (hasSvgAnnotations && box.kind === 'trash') return null

                  const bPct = box.boxPct || (box as any).box_pct || { x: 20, y: 20, w: 30, h: 20 }
                  const cPct = box.confidencePct ?? (box as any).confidence_pct ?? 90
                  const isPothole = box.kind === 'pothole'
                  return (
                    <div
                      key={box.id}
                      className={`bbox-rect ${isPothole ? 'bbox-pothole' : 'bbox-trash'}`}
                      style={{
                        left: `${bPct.x}%`,
                        top: `${bPct.y}%`,
                        width: `${bPct.w}%`,
                        height: `${bPct.h}%`,
                      }}
                      title={`${box.label} (${cPct}% confidence)`}
                    >
                      <span className={`bbox-tag ${isPothole ? 'tag-pothole' : 'tag-trash'}`}>
                        {isPothole ? '🕳️' : '🗑️'} {box.label} · {cPct}%
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
            <small>No photo or video file attached</small>
          </div>
        )}
      </div>

      {/* 2. Detection Telemetry Bar */}
      <div className="amount-bar">
        <div className="amount-pills">
          {currentPotholes > 0 ? (
            <span className="amount-pill amount-pill-pothole">
              🕳️ <strong>Potholes Detected: {currentPotholes}</strong>
            </span>
          ) : (
            <span className="amount-pill amount-pill-clean">
              ✅ <strong>Road Surface: Clear</strong>
            </span>
          )}

          {currentTrash > 0 ? (
            <span className="amount-pill amount-pill-trash">
              🗑️ <strong>Waste Items: {currentTrash}</strong>
            </span>
          ) : (
            <span className="amount-pill amount-pill-clean">
              ✅ <strong>Sanitation: Clear</strong>
            </span>
          )}
        </div>

        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {(currentBoxes.length > 0 || hasSvgAnnotations) && !isPlayingPreview && (
            <button
              type="button"
              className="button button-outline"
              style={{ padding: '4px 10px', fontSize: '11px', height: '30px' }}
              onClick={() => setShowBoxes(!showBoxes)}
            >
              {showBoxes ? 'Hide Overlays' : 'Show Overlays'}
            </button>
          )}

          {previewVideoUrl && (
            <button
              type="button"
              className={`button ${isPlayingPreview ? 'button-outline' : 'button-primary'}`}
              style={{
                padding: '4px 12px',
                fontSize: '11px',
                height: '30px',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
              }}
              onClick={() => setIsPlayingPreview(!isPlayingPreview)}
            >
              {isPlayingPreview ? '⏹ Stop Video' : '🎬 Annotated Video Preview'}
            </button>
          )}
        </div>
      </div>

      {/* 3. Video Frame-by-Frame Stepper */}
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
              ◀ Previous Frame
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
 * Location Hotspots Board for Barangay Operations.
 * Ranks streets automatically based on validated reports and detected hazard density.
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
          <div className="section-index">PRIORITY CORRIDORS</div>
          <h2 style={{ fontSize: '18px', margin: '4px 0' }}>Location Hotspots</h2>
          <p style={{ fontSize: '13px', color: '#555' }}>
            Streets ranked by reported pothole and sanitation hazard recurrence. Click any street card to filter the queue below.
          </p>
        </div>
      </div>
      <div className="hotspots-board">
        {hotspots.map((spot, idx) => {
          const isSelected = selectedStreet === spot.location
          const prioClass =
            spot.priorityLevel === 'Urgent'
              ? 'priority-urgent'
              : spot.priorityLevel === 'High'
              ? 'priority-high'
              : 'priority-moderate'
          return (
            <div
              key={spot.location}
              className={`hotspot-card ${prioClass} ${isSelected ? 'selected' : ''}`}
              onClick={() => onSelectStreet(isSelected ? '' : spot.location)}
            >
              <div className="hotspot-card-top">
                <span className="hotspot-rank">RANK #{idx + 1}</span>
                <span
                  className={`incident-status status-${spot.priorityLevel.toLowerCase()}`}
                  style={{ fontSize: '10px', padding: '2px 6px' }}
                >
                  {spot.priorityLevel === 'Urgent'
                    ? '🔴 Urgent Action'
                    : spot.priorityLevel === 'High'
                    ? '🟠 High Priority'
                    : '🟢 Scheduled'}
                </span>
              </div>
              <strong className="hotspot-street-name">{spot.location}</strong>
              <div className="hotspot-stats">
                <span>🕳️ {spot.potholesCount} Potholes</span>
                <span>·</span>
                <span>🗑️ {spot.trashCount} Waste</span>
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
 * Resident Portal: "My Submitted Reports".
 * Shows only the resident's verified submissions with current municipal status.
 */
export function MyReports({ store }: { store: Store }) {
  const reports = store.observations
    .filter(o => o.reporter.includes('Verified Resident') || o.reporter === 'Maria Santos (Verified Resident)')
    .sort((a, b) => b.submittedAt.localeCompare(a.submittedAt))

  return (
    <div className="report-cards">
      {reports.map(report => {
        const incident = store.incidents.find(i => i.id === report.incidentId)
        return (
          <article
            className="panel report-card"
            key={report.id}
            style={{ gridTemplateColumns: '1fr', padding: '22px', gap: '16px' }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start',
                flexWrap: 'wrap',
                gap: '8px',
              }}
            >
              <div>
                <span style={{ fontSize: '12px', fontWeight: 700, color: '#087a72' }}>
                  Report #{report.id}
                </span>
                <h2 style={{ fontSize: '20px', margin: '4px 0' }}>{report.category}</h2>
                <p style={{ margin: '2px 0', fontWeight: 600, color: '#222' }}>
                  📍 {report.location}
                </p>
                <small style={{ color: '#666' }}>
                  Submitted {formatDate(report.submittedAt)} · {report.source}
                </small>
              </div>
              <PlainStatus value={incident?.status || report.review} />
            </div>

            <Evidence report={report} showAdminOverlay={true} />

            <div
              style={{
                background: '#f8faf9',
                padding: '14px',
                borderRadius: '6px',
                border: '1px solid #dae5df',
              }}
            >
              <p style={{ margin: '0 0 8px 0', fontSize: '13px' }}>
                <strong>Your Notes:</strong> {report.description}
              </p>
              <div
                style={{
                  borderTop: '1px solid #e2ece6',
                  paddingTop: '10px',
                  marginTop: '10px',
                }}
              >
                <strong style={{ fontSize: '12px', color: '#087a72' }}>Barangay Action Update:</strong>
                <p style={{ margin: '4px 0 0 0', fontSize: '13px', color: '#333' }}>
                  {incident?.note ||
                    report.reviewNote ||
                    'Your report has been logged and is scheduled for review by the barangay team.'}
                </p>
              </div>
            </div>
          </article>
        )
      })}
      {!reports.length && (
        <div className="panel empty-state">
          <h2>No reports submitted yet</h2>
          <p>Click "Submit an Incident Report" above to file your first photo or video.</p>
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
    <article
      className="panel review-card"
      style={{ gridTemplateColumns: '1fr', gap: '16px', padding: '20px' }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          flexWrap: 'wrap',
          gap: '8px',
        }}
      >
        <div>
          <span style={{ fontSize: '12px', fontWeight: 700, color: '#087a72' }}>
            {report.id} · {report.reporter}
          </span>
          <h2 style={{ fontSize: '20px', margin: '4px 0' }}>{report.category}</h2>
          <p style={{ margin: '2px 0', fontWeight: 600, color: '#333' }}>📍 {report.location}</p>
          <small style={{ color: '#666' }}>
            Observed {formatDate(report.observedAt)} · Submitted {formatDate(report.submittedAt)}
          </small>
        </div>
        <PlainStatus value={report.review} />
      </div>

      <Evidence report={report} showAdminOverlay={true} />

      <div
        style={{
          background: '#f8faf9',
          padding: '14px',
          borderRadius: '6px',
          border: '1px solid #dae5df',
        }}
      >
        <p style={{ margin: '0 0 10px 0', fontSize: '13px' }}>
          <strong>Citizen Description:</strong> {report.description}
        </p>

        {candidates.length > 0 && (
          <p className="match-note" style={{ margin: '8px 0' }}>
            ℹ️ Correlated street record: {candidates.length} active incident{candidates.length === 1 ? '' : 's'} on this street corridor.
          </p>
        )}

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '12px',
            marginTop: '10px',
          }}
        >
          <label className="form-field" style={{ margin: 0 }}>
            Incident Correlation
            <select value={activeTarget} onChange={e => setTarget(e.target.value)}>
              <option value="">Register as new street incident</option>
              {candidates.map(i => (
                <option key={i.id} value={i.id}>
                  Link to {i.id} · {i.category} ({i.status})
                </option>
              ))}
            </select>
          </label>

          <label className="form-field" style={{ margin: 0 }}>
            Dispatch / Review Notes
            <input
              type="text"
              value={note}
              onChange={e => setNote(e.target.value)}
              placeholder="e.g. Approved. Maintenance work order generated."
            />
          </label>
        </div>

        {error && (
          <p className="form-error" role="alert" style={{ marginTop: '8px' }}>
            {error}
          </p>
        )}

        <div
          className="review-actions"
          style={{ marginTop: '14px', display: 'flex', gap: '10px', justifyContent: 'flex-end' }}
        >
          <button
            type="button"
            className="button button-outline"
            onClick={() => {
              if (!note.trim()) {
                setError('Please provide a reason before declining.')
                return
              }
              onReview(report.id, 'reject', undefined, note)
            }}
          >
            Decline Report
          </button>

          <button
            type="button"
            className="button button-primary"
            style={{ fontWeight: 750 }}
            onClick={() =>
              onReview(
                report.id,
                'validate',
                activeTarget || undefined,
                note || 'Validated report registered. Assigned for municipal action.'
              )
            }
          >
            {activeTarget ? 'Validate & Link Incident' : 'Validate & Create Incident'}
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
    o =>
      o.review === 'Pending review' &&
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
          <strong>{pending.length} reports awaiting review</strong>
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
            {categories.map(c => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="report-cards">
        {pending.map(report => (
          <ReviewCard key={report.id} store={store} report={report} onReview={onReview} />
        ))}
        {!pending.length && (
          <div className="panel empty-state">
            <h2>Queue clear</h2>
            <p>No reports currently awaiting review{selectedStreet ? ' on this street corridor' : ''}.</p>
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
          <small>
            {id} · 📍 {incident.location}
          </small>
        </span>
        <span className="incident-summary-end">
          <span>{evidence.length} validated report{evidence.length === 1 ? '' : 's'}</span>
          <PlainStatus value={incident.status} />
          <span aria-hidden="true">⌄</span>
        </span>
      </summary>
      <div className="incident-editor-body">
        <form
          onSubmit={e => {
            e.preventDefault()
            if (note.trim()) onUpdate(id, status, note)
          }}
        >
          <div className="form-grid">
            <label className="form-field">
              Status Designation
              <select value={status} onChange={e => setStatus(e.target.value as IncidentStatus)}>
                {incidentStatuses.map(s => (
                  <option key={s}>{s}</option>
                ))}
              </select>
            </label>
            <label className="form-field">
              Official Progress Update
              <textarea
                required
                maxLength={600}
                rows={2}
                value={note}
                onChange={e => setNote(e.target.value)}
              />
            </label>
          </div>
          <button type="submit" className="button button-primary" disabled={!note.trim()}>
            Save Status Update
          </button>
        </form>

        <h3 style={{ marginTop: '20px' }}>Attached Photographic & Video Evidence</h3>
        <div className="evidence-list">
          {evidence.map(o => (
            <div className="evidence-entry" key={o.id} style={{ gridTemplateColumns: '140px 1fr' }}>
              <Evidence report={o} showAdminOverlay={true} />
              <div>
                <strong>{o.id}</strong>
                <p>{o.description}</p>
                <small>
                  {o.source} · {formatDate(o.observedAt)}
                </small>
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
    i =>
      (filter === 'All statuses' || i.status === filter) &&
      `${i.id} ${i.category} ${i.location}`.toLowerCase().includes(query.toLowerCase())
  )

  return (
    <>
      <div className="list-toolbar">
        <label className="form-field">
          Search Municipal Records
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Type street name or incident ID..."
          />
        </label>
        <label className="form-field">
          Status
          <select value={filter} onChange={e => setFilter(e.target.value)}>
            <option>All statuses</option>
            {incidentStatuses.map(s => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
      </div>
      <div className="report-cards">
        {incidents.map(i => (
          <IncidentEditor
            key={`${i.id}-${i.updatedAt}`}
            store={store}
            id={i.id}
            onUpdate={onUpdate}
          />
        ))}
        {!incidents.length && (
          <div className="panel empty-state">No incident records match the current filter.</div>
        )}
      </div>
    </>
  )
}

export function VideoMonitor({
  onAddRealObservations,
  onReview,
}: {
  onAddRealObservations?: (observations: Observation[]) => void
  onReview: () => void
}) {
  const [location, setLocation] = useState(locations[0])
  const [category, setCategory] = useState<Category>('Pothole')
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

      const resp = await fetch('/api/v1/analyze', {
        method: 'POST',
        body: formData,
      })

      if (!resp.ok) {
        const errText = await resp.text()
        throw new Error(`Server returned HTTP ${resp.status}: ${errText}`)
      }

      const result: UnifiedIncidentResponse = await resp.json()

      if (onAddRealObservations && result.frames && result.frames.length > 0) {
        const now = new Date().toISOString()
        const primaryFrame = result.frames[0]

        const singleObservation: Observation = {
          id: `VID-${result.incident_id.slice(-6)}-1`,
          source: 'Patrol dashcam',
          reporter: 'Patrol Surveillance Unit 01',
          location,
          category,
          observedAt: now,
          submittedAt: now,
          description: `Patrol footage (${result.source_media.filename}). Scanned ${result.frames.length} key frames for road hazards.`,
          photo: primaryFrame?.frame_url,
          clipTime: '00:01',
          confidence: 94,
          review: 'Pending review',
          reviewNote: '',
          mediaType: 'video',
          frames: result.frames.map((frame, idx) => ({
            frameIndex: idx + 1,
            timestampSec: frame.timestamp_sec,
            clipTime: `00:${String(Math.floor(frame.timestamp_sec)).padStart(2, '0')}`,
            frameUrl: frame.frame_url,
            boxes: frame.boxes || [],
            potholeCount: frame.pothole_count ?? (frame.road_defect_findings?.defects?.length || 0),
            trashCount: frame.trash_count ?? (frame.trash_findings?.total_items || 0),
            regions: frame.trash_findings?.region_list || [],
            detections: frame.trash_findings?.detections || [],
            defects: frame.road_defect_findings?.defects || [],
            width: frame.width,
            height: frame.height,
            frameHotspotScore: frame.frame_hotspot_score,
            framePriorityLevel: frame.frame_priority_level,
          })),
          boxes: primaryFrame?.boxes || [],
          potholeCount: result.summary?.total_potholes || 0,
          trashCount: result.summary?.total_trash_items || 0,
          regions: primaryFrame?.trash_findings?.region_list || [],
          detections: primaryFrame?.trash_findings?.detections || [],
          defects: primaryFrame?.road_defect_findings?.defects || [],
          previewVideoUrl: result.preview_video_url || undefined,
          dualModelResult: result,
          hotspotScore: result.hotspot_score,
          priorityLevel: result.priority_level,
          imageWidth: primaryFrame?.width,
          imageHeight: primaryFrame?.height,
        }

        onAddRealObservations([singleObservation])
      }
    } catch (err) {
      setVideoError(
        `Unable to analyze video: ${err instanceof Error ? err.message : String(err)}`
      )
    } finally {
      setProcessingVideo(false)
    }
  }

  return (
    <div className="submit-layout">
      <section className="panel">
        <div className="panel-heading">
          <div>
            <div className="section-index">PATROL SCANNER</div>
            <h2>Corridor Video Inspection</h2>
            <p>Upload a street patrol video clip for 1.0 FPS frame extraction, dual-model scanning, and annotated preview rendering.</p>
          </div>
        </div>
        <div className="form-body">
          <div
            style={{
              padding: '16px',
              background: '#f8faf9',
              borderRadius: '8px',
              border: '1px solid #d5e2da',
              marginBottom: '20px',
            }}
          >
            <strong style={{ display: 'block', fontSize: '14px', marginBottom: '4px', color: '#134e48' }}>
              Upload Patrol Video (.mp4, .mov, .webm)
            </strong>
            <p style={{ fontSize: '13px', color: '#4b5563', marginBottom: '12px' }}>
              The ingestion service samples the video once at 1.0 FPS, applies smart hazard filtering, generates an annotated MP4 preview, and packages the findings into a consolidated card with a frame stepper.
            </p>
            <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
              <input
                ref={videoInputRef}
                type="file"
                accept="video/mp4,video/quicktime,video/webm"
                style={{ display: 'none' }}
                onChange={e => setVideoFile(e.target.files?.[0] || null)}
              />
              <button
                type="button"
                className="button button-outline"
                onClick={() => videoInputRef.current?.click()}
              >
                {videoFile ? `Selected: ${videoFile.name}` : 'Select Video File'}
              </button>
              {videoFile && (
                <button
                  type="button"
                  className="button button-primary"
                  disabled={processingVideo}
                  onClick={handleRealVideoProcess}
                >
                  {processingVideo ? 'Scanning Frames & Processing Preview…' : 'Process Video'}
                </button>
              )}
            </div>
            {videoError && (
              <p className="form-error" style={{ marginTop: '10px' }}>
                {videoError}
              </p>
            )}
          </div>

          <div className="form-grid">
            <label className="form-field">
              Patrol Street Corridor
              <select value={location} onChange={e => setLocation(e.target.value)}>
                {locations.map(l => (
                  <option key={l}>{l}</option>
                ))}
              </select>
            </label>
            <label className="form-field">
              Primary Surveillance Focus
              <select value={category} onChange={e => setCategory(e.target.value as Category)}>
                {categories.map(c => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </label>
          </div>
        </div>
      </section>

      <aside className="panel guidance-panel">
        <div className="section-index">CORRIDOR PROCESSING</div>
        <h2>Automated Hazard Detection</h2>
        <div className="guide-item">
          <span>01</span>
          <p>
            <strong>Dual-Model Engine:</strong> Model A identifies waste accumulations while Model B analyzes road surface integrity.
          </p>
        </div>
        <div className="guide-item">
          <span>02</span>
          <p>
            <strong>Smart Frame Filtering:</strong> Filters out non-hazard empty frames, presenting only actionable frames with confirmed defects.
          </p>
        </div>
        <div className="guide-item">
          <span>03</span>
          <p>
            <strong>Consolidated Review:</strong> All retained frames remain grouped under a single incident card with interactive stepping.
          </p>
        </div>
      </aside>
    </div>
  )
}
