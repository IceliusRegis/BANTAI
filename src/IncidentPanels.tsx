import { useState, useRef, useEffect } from 'react'
import {
  categories,
  locations,
  duplicateCandidates,
  incidentStatuses,
  residentId,
  rankHotspots,
} from './incidents'
import type {
  Category,
  IncidentStatus,
  Observation,
  Store,
  TrashRegion,
  TrashDetection,
  RoadDefect,
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
  return (
    <span className={`incident-status status-${value.toLowerCase().replace(/ /g, '-')}`}>
      {value}
    </span>
  )
}

const SEARCH_HISTORY_KEY = 'bantai_search_history_v1'

function getSearchHistory(): string[] {
  if (typeof window === 'undefined' || !window.sessionStorage) return []
  try {
    const raw = window.sessionStorage.getItem(SEARCH_HISTORY_KEY)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed.slice(0, 5) : []
  } catch {
    return []
  }
}

function pushSearchHistory(term: string): string[] {
  const clean = term.trim()
  if (!clean) return getSearchHistory()
  const next = [clean, ...getSearchHistory().filter(item => item.toLowerCase() !== clean.toLowerCase())].slice(0, 5)
  if (typeof window !== 'undefined' && window.sessionStorage) {
    try {
      window.sessionStorage.setItem(SEARCH_HISTORY_KEY, JSON.stringify(next))
    } catch {
      // Ignore storage error
    }
  }
  return next
}

function removeSearchHistoryItem(term: string): string[] {
  const next = getSearchHistory().filter(item => item !== term)
  if (typeof window !== 'undefined' && window.sessionStorage) {
    try {
      window.sessionStorage.setItem(SEARCH_HISTORY_KEY, JSON.stringify(next))
    } catch {
      // Ignore storage error
    }
  }
  return next
}

function withinDateRange(isoDate: string, range: string): boolean {
  if (range === 'All time') return true
  const timestamp = new Date(isoDate).getTime()
  if (!Number.isFinite(timestamp)) return true
  const diffHours = (Date.now() - timestamp) / 3600000
  if (range === 'Last 24 hours') return diffHours <= 24
  if (range === 'Past 7 days') return diffHours <= 24 * 7
  if (range === 'Past 30 days') return diffHours <= 24 * 30
  return true
}

function SvgBoundingBoxes({
  imgW,
  imgH,
  regions,
  detections,
  defects,
}: {
  imgW: number
  imgH: number
  regions?: TrashRegion[]
  detections?: TrashDetection[]
  defects?: RoadDefect[]
}) {
  return (
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
      {regions
        ?.filter(r => r.item_count >= 2)
        .map((reg, idx) => {
          const [bx, by, bw, bh] = reg.bbox
          const isPile = reg.category === 'garbage_pile'
          const strokeColor = isPile ? '#ff8c00' : '#00b4d8'
          const bannerBg = isPile ? 'rgba(255, 140, 0, 0.95)' : 'rgba(0, 180, 216, 0.95)'
          const labelText = isPile
            ? `Garbage Pile (${reg.item_count} items)`
            : `Cluster of Litter (${reg.item_count} items)`
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

      {/* 2. Individual Detection Boxes */}
      {detections?.map((det, idx) => {
        const [bx, by, bw, bh] = det.bbox
        let color = '#3cb44b' // Plastic: Green
        if (det.bantai_category === 'Potentially Hazardous') color = '#e6194b'
        else if (det.bantai_category === 'Paper') color = '#d97706'
        else if (det.bantai_category === 'Metal') color = '#911eb4'
        else if (det.bantai_category === 'Organic') color = '#2b9348'

        const labelText = `${det.bantai_category} ${Math.round(det.model_score * 100)}%`
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
            <rect x={bx} y={badgeY} width={badgeW} height={badgeH} fill={color} rx="2" />
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

      {/* 3. Model B Pothole Defect Boxes */}
      {defects?.map((defect, idx) => {
        const [bx, by, bw, bh] = defect.bbox
        const isHigh = defect.severity === 'high'
        const isMod = defect.severity === 'moderate'
        const strokeColor = isHigh ? '#dc2626' : isMod ? '#ea580c' : '#16a34a'
        const fillTint = isHigh
          ? 'rgba(220, 38, 38, 0.12)'
          : isMod
            ? 'rgba(234, 88, 12, 0.12)'
            : 'rgba(22, 163, 74, 0.12)'
        const confPct = defect.confidence_pct ?? Math.round(defect.confidence * 100)
        const sevLabel = defect.raw_severity || (isHigh ? 'Critical' : isMod ? 'Moderate' : 'Low')
        const labelText = `Detected Pothole · ${sevLabel} (${confPct}%)`
        const fontSize = Math.max(15, Math.round(imgW * 0.015))
        const padY = Math.round(fontSize * 0.35)
        const badgeH = fontSize + padY * 2
        const badgeW = labelText.length * (fontSize * 0.56) + 14
        const badgeY = Math.max(0, by - badgeH - 2)

        return (
          <g key={`def-${defect.defect_id || idx}`}>
            <rect
              x={bx}
              y={by}
              width={bw}
              height={bh}
              fill={fillTint}
              stroke={strokeColor}
              strokeWidth={Math.max(3, imgW * 0.003)}
              rx="3"
            />
            <rect x={bx} y={badgeY} width={badgeW} height={badgeH} fill={strokeColor} rx="3" />
            <text
              x={bx + 7}
              y={badgeY + fontSize + padY * 0.25}
              fill="#ffffff"
              fontSize={fontSize}
              fontWeight="700"
              fontFamily="system-ui, -apple-system, sans-serif"
            >
              {labelText}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

function ImageLightboxModal({
  src,
  alt,
  imgW,
  imgH,
  regions,
  detections,
  defects,
  initialShowBoxes,
  onClose,
}: {
  src: string
  alt: string
  imgW: number
  imgH: number
  regions?: TrashRegion[]
  detections?: TrashDetection[]
  defects?: RoadDefect[]
  initialShowBoxes: boolean
  onClose: () => void
}) {
  const [scale, setScale] = useState(1.0)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [dragging, setDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [showOverlay, setShowOverlay] = useState(initialShowBoxes)

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  function zoomBy(delta: number) {
    setScale(prev => Math.min(5.0, Math.max(0.5, Math.round((prev + delta) * 100) / 100)))
  }

  function resetView() {
    setScale(1.0)
    setPan({ x: 0, y: 0 })
  }

  return (
    <div
      className="lightbox-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="High-resolution evidence inspection modal"
      onClick={e => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div className="lightbox-toolbar">
        <div className="lightbox-title">
          <strong>High-Resolution Evidence Inspector</strong>
          <span>Scroll to zoom ({Math.round(scale * 100)}%) · Drag to pan</span>
        </div>
        <div className="lightbox-controls">
          <label className="lightbox-toggle">
            <input
              type="checkbox"
              checked={showOverlay}
              onChange={e => setShowOverlay(e.target.checked)}
            />
            <span>Show AI Detection Highlights</span>
          </label>
          <button type="button" className="button button-outline" onClick={() => zoomBy(-0.25)} aria-label="Zoom out">
            -
          </button>
          <button type="button" className="button button-outline" onClick={resetView} aria-label="Reset zoom to 100%">
            100%
          </button>
          <button type="button" className="button button-outline" onClick={() => zoomBy(0.25)} aria-label="Zoom in">
            +
          </button>
          <button type="button" className="button button-primary" onClick={onClose} aria-label="Close lightbox">
            Close (Esc)
          </button>
        </div>
      </div>

      <div
        className="lightbox-stage"
        onWheel={e => {
          e.preventDefault()
          zoomBy(e.deltaY < 0 ? 0.2 : -0.2)
        }}
        onMouseDown={e => {
          setDragging(true)
          setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y })
        }}
        onMouseMove={e => {
          if (!dragging) return
          setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y })
        }}
        onMouseUp={() => setDragging(false)}
        onMouseLeave={() => setDragging(false)}
        style={{ cursor: dragging ? 'grabbing' : 'grab' }}
      >
        <div
          style={{
            position: 'relative',
            aspectRatio: `${imgW} / ${imgH}`,
            maxWidth: '90vw',
            maxHeight: '80vh',
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})`,
            transformOrigin: 'center center',
            transition: dragging ? 'none' : 'transform 0.1s ease-out',
            userSelect: 'none',
          }}
        >
          <img
            src={src}
            alt={alt}
            draggable={false}
            style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
          />
          {showOverlay && (
            <SvgBoundingBoxes
              imgW={imgW}
              imgH={imgH}
              regions={regions}
              detections={detections}
              defects={defects}
            />
          )}
        </div>
      </div>
    </div>
  )
}

export function Evidence({
  report,
  showBoxes = true,
  selectedFrameIdx: controlledIdx,
  onSelectFrameIdx,
}: {
  report: Observation
  showBoxes?: boolean
  selectedFrameIdx?: number
  onSelectFrameIdx?: (idx: number) => void
}) {
  const [internalFrameIdx, setInternalFrameIdx] = useState(0)
  const [lightboxOpen, setLightboxOpen] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)

  const selectedFrameIdx = controlledIdx !== undefined ? controlledIdx : internalFrameIdx
  const setSelectedFrameIdx = (idx: number) => {
    setInternalFrameIdx(idx)
    onSelectFrameIdx?.(idx)
  }

  const frames = report.dualModelResult?.frames || []
  const safeIdx = frames.length > 0 ? Math.min(selectedFrameIdx, frames.length - 1) : 0
  const activeFrame = frames.length > 0 ? frames[safeIdx] : null

  const videoSrc =
    report.videoUrl ||
    (report.dualModelResult?.source_media?.type === 'video'
      ? report.dualModelResult?.source_media?.media_url
      : undefined)

  const activePhoto = activeFrame?.frame_url || report.photo
  const activeRegions = activeFrame ? activeFrame.trash_findings?.region_list || [] : report.regions
  const activeDetections = activeFrame ? activeFrame.trash_findings?.detections || [] : report.detections
  const activeDefects = activeFrame ? activeFrame.road_defect_findings?.defects || [] : report.defects
  const imgW = activeFrame?.width || report.imageWidth || 1280
  const imgH = activeFrame?.height || report.imageHeight || 720

  // Clean Vector-Based Civic Placeholder (Section 2.5)
  if (!activePhoto && !videoSrc) {
    return (
      <div className="evidence-placeholder">
        <div className="civic-placeholder-icon" aria-hidden="true">
          <svg
            width="36"
            height="36"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.7"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            <circle cx="12" cy="11" r="2.5" />
          </svg>
        </div>
        <strong>
          {report.source === 'Video observation'
            ? `Awaiting camera feed · ${report.clipTime || 'Frame'}`
            : 'No image attached'}
        </strong>
        <small>{report.category} · Barangay Santa Elena</small>
      </div>
    )
  }

  const hasBoxes =
    showBoxes &&
    Boolean(
      (activeDetections && activeDetections.length > 0) ||
        (activeRegions && activeRegions.length > 0) ||
        (activeDefects && activeDefects.length > 0),
    )

  function handleSelectFrame(idx: number, timestampSec: number) {
    setSelectedFrameIdx(idx)
    if (videoRef.current) {
      try {
        videoRef.current.currentTime = timestampSec
      } catch {
        // Ignore seek error if metadata not yet loaded
      }
    }
  }

  return (
    <div className="dual-media-container">
      {/* Fallback Video Player only if no extracted frame image is available */}
      {!activePhoto && videoSrc && (
        <div className="video-player-tier">
          <video ref={videoRef} src={videoSrc} controls className="synced-video-player" />
        </div>
      )}

      {/* Active Frame / Image with Scalable SVG Bounding Box Overlay */}
      {activePhoto && (
        <div
          className="evidence-canvas-wrapper"
          onClick={() => setLightboxOpen(true)}
          title="Click to open High-Resolution Pan & Zoom Inspector"
        >
          <div
            style={{
              position: 'relative',
              maxWidth: '100%',
              maxHeight: '100%',
              aspectRatio: `${imgW} / ${imgH}`,
              display: 'flex',
            }}
          >
            <img
              src={activePhoto}
              alt={`Submitted ${report.category.toLowerCase()} at ${report.location}`}
              style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
            />
            {hasBoxes && (
              <SvgBoundingBoxes
                imgW={imgW}
                imgH={imgH}
                regions={activeRegions}
                detections={activeDetections}
                defects={activeDefects}
              />
            )}
          </div>
          <button
            type="button"
            className="zoom-trigger-badge"
            onClick={e => {
              e.stopPropagation()
              setLightboxOpen(true)
            }}
            aria-label="Inspect full-screen with zoom and pan"
          >
            <svg
              width="13"
              height="13"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
              <line x1="11" y1="8" x2="11" y2="14" />
              <line x1="8" y1="11" x2="14" y2="11" />
            </svg>
            <span>Zoom</span>
          </button>
        </div>
      )}

      {/* Synchronized 1.0 FPS Frame Carousel / Stepper (Section 2.1) */}
      {frames.length > 1 && (
        <div className="frame-carousel-strip" role="region" aria-label="Sampled 1.0 FPS video frames">
          {frames.map((frame, idx) => {
            const isSelected = idx === safeIdx
            const itemCount =
              (frame.trash_findings?.total_items || 0) +
              (frame.road_defect_findings?.total_defects || 0)
            return (
              <button
                key={`frame-thumb-${frame.frame_index}-${idx}`}
                type="button"
                className={`frame-thumb-card ${isSelected ? 'active' : ''}`}
                onClick={() => handleSelectFrame(idx, frame.timestamp_sec)}
              >
                <img src={frame.frame_url} alt={`Frame at ${frame.timestamp_sec.toFixed(1)}s`} />
                <div className="frame-thumb-meta">
                  <strong>00:{String(Math.floor(frame.timestamp_sec)).padStart(2, '0')}</strong>
                  <span>{itemCount} det</span>
                </div>
              </button>
            )
          })}
        </div>
      )}

      {lightboxOpen && activePhoto && (
        <ImageLightboxModal
          src={activePhoto}
          alt={`Submitted ${report.category.toLowerCase()} at ${report.location}`}
          imgW={imgW}
          imgH={imgH}
          regions={activeRegions}
          detections={activeDetections}
          defects={activeDefects}
          initialShowBoxes={showBoxes}
          onClose={() => setLightboxOpen(false)}
        />
      )}
    </div>
  )
}

function RegionBadge({ region }: { region: TrashRegion }) {
  const isIndividual = region.category === 'individual_litter'
  const isPile = region.category === 'garbage_pile'
  const humanizedLabel = isPile
    ? `Garbage Pile (${region.item_count} items)`
    : isIndividual
      ? 'Individual Litter Item'
      : `Cluster of Litter (${region.item_count} items)`

  return (
    <div
      className="region-chip"
      style={{
        margin: '4px 0',
        padding: '8px 12px',
        background: 'rgba(235, 94, 40, 0.08)',
        borderRadius: '6px',
        borderLeft: '3px solid #eb5e28',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px' }}>
        <strong>{humanizedLabel}</strong>
        {isIndividual && region.model_score !== null ? (
          <span style={{ fontSize: '12px', fontWeight: 600, flexShrink: 0 }}>
            {(region.model_score * 100).toFixed(0)}% Confidence
          </span>
        ) : (
          <span style={{ fontSize: '12px', fontWeight: 600, color: '#eb5e28', flexShrink: 0 }}>
            Spatial Hazard Region
          </span>
        )}
      </div>
      {region.category_breakdown && Object.keys(region.category_breakdown).length > 0 && (
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '4px' }}>
          {Object.entries(region.category_breakdown).map(([cat, count]) => (
            <span
              key={cat}
              style={{
                fontSize: '11px',
                background: '#fff',
                border: '1px solid #ddd',
                padding: '1px 6px',
                borderRadius: '4px',
              }}
            >
              {cat}: {count}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function DefectBadge({ defect }: { defect: RoadDefect }) {
  const sevColor =
    defect.severity === 'high' ? '#d90429' : defect.severity === 'moderate' ? '#f77f00' : '#2b9348'
  const confText =
    defect.confidence >= 0.7
      ? 'High Confidence'
      : defect.confidence >= 0.4
        ? 'Moderate Confidence'
        : 'Detected'
  return (
    <div
      className="defect-chip"
      style={{
        margin: '4px 0',
        padding: '8px 12px',
        background: 'rgba(43, 147, 72, 0.08)',
        borderRadius: '6px',
        borderLeft: `3px solid ${sevColor}`,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px' }}>
        <strong>Detected Pothole · {confText}</strong>
        <span
          style={{
            fontSize: '12px',
            fontWeight: 700,
            color: sevColor,
            textTransform: 'uppercase',
            flexShrink: 0,
          }}
        >
          {defect.raw_severity || defect.severity}
        </span>
      </div>
      <div style={{ fontSize: '12px', marginTop: '3px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        <span>Est. Diameter: ~{defect.estimated_diameter_cm} cm</span>
        <span>·</span>
        <span>AI Score: {(defect.confidence * 100).toFixed(0)}%</span>
      </div>
    </div>
  )
}

function VerificationLifecycleBadge({ review }: { review: Observation['review'] }) {
  if (review === 'Validated') {
    return (
      <span className="lifecycle-badge lifecycle-verified">
        Barangay Verified · Work Order Scheduled
      </span>
    )
  }
  if (review === 'Rejected') {
    return <span className="lifecycle-badge lifecycle-rejected">Reviewed · Closed by Admin</span>
  }
  return (
    <span className="lifecycle-badge lifecycle-pending">
      AI Detection (Pending Barangay Verification)
    </span>
  )
}

function SearchAndFilterBar({
  query,
  onQueryChange,
  categoryFilter,
  onCategoryChange,
  statusFilter,
  onStatusChange,
  statusOptions,
  dateRange,
  onDateRangeChange,
  totalCount,
  countLabel,
  extraToggle,
}: {
  query: string
  onQueryChange: (val: string) => void
  categoryFilter: string
  onCategoryChange: (val: string) => void
  statusFilter?: string
  onStatusChange?: (val: string) => void
  statusOptions?: readonly string[]
  dateRange: string
  onDateRangeChange: (val: string) => void
  totalCount: number
  countLabel: string
  extraToggle?: React.ReactNode
}) {
  const [history, setHistory] = useState<string[]>(() => getSearchHistory())

  function handleSearchCommit(e: React.FormEvent) {
    e.preventDefault()
    if (query.trim()) {
      setHistory(pushSearchHistory(query))
    }
  }

  return (
    <div className="unified-filter-bar panel">
      <div className="filter-bar-top">
        <strong>
          {totalCount} {countLabel}
        </strong>
        {extraToggle}
      </div>
      <form className="filter-controls-grid" onSubmit={handleSearchCommit}>
        <div className="search-field">
          <svg
            width="15"
            height="15"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.7"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            value={query}
            onChange={e => onQueryChange(e.target.value)}
            onBlur={() => {
              if (query.trim()) setHistory(pushSearchHistory(query))
            }}
            placeholder="Search ID (OBS/INC), street name, or description…"
            aria-label="Search records"
          />
          {query && (
            <button
              type="button"
              className="clear-search-btn"
              onClick={() => onQueryChange('')}
              aria-label="Clear search query"
            >
              ×
            </button>
          )}
        </div>

        <label className="form-field compact-field">
          <span>Category</span>
          <select value={categoryFilter} onChange={e => onCategoryChange(e.target.value)}>
            <option value="All categories">All categories</option>
            {categories.map(c => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>

        {statusOptions && onStatusChange && (
          <label className="form-field compact-field">
            <span>Status</span>
            <select value={statusFilter} onChange={e => onStatusChange(e.target.value)}>
              <option value="All statuses">All statuses</option>
              {statusOptions.map(s => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
        )}

        <label className="form-field compact-field">
          <span>Date range</span>
          <select value={dateRange} onChange={e => onDateRangeChange(e.target.value)}>
            <option value="All time">All time</option>
            <option value="Last 24 hours">Last 24 hours</option>
            <option value="Past 7 days">Past 7 days</option>
            <option value="Past 30 days">Past 30 days</option>
          </select>
        </label>
      </form>

      {history.length > 0 && (
        <div className="search-history-chips" aria-label="Recent searches">
          <small>Recent searches:</small>
          {history.map(item => (
            <span key={item} className="search-chip">
              <button type="button" onClick={() => onQueryChange(item)}>
                {item}
              </button>
              <button
                type="button"
                aria-label={`Remove ${item} from search history`}
                onClick={() => setHistory(removeSearchHistoryItem(item))}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

export function HotspotLeaderboard({
  store,
  mode,
  onReportHere,
  onInspectIncidents,
}: {
  store: Store
  mode: 'user' | 'admin'
  onReportHere?: (location: string) => void
  onInspectIncidents?: () => void
}) {
  const [categoryFilter, setCategoryFilter] = useState<'All' | Category>('All')
  const entries = rankHotspots(store, categoryFilter)

  return (
    <section className="panel leaderboard-panel">
      <div className="panel-heading leaderboard-heading">
        <div>
          <div className="section-index">02 / MUNICIPAL HOTSPOT LEADERBOARD</div>
          <h2>Priority hotspots across Barangay Santa Elena</h2>
          <p>
            Ranked by validated observation frequency, recency, and AI-assessed hazard severity.
          </p>
        </div>
        <div className="leaderboard-filter-pills" role="group" aria-label="Hotspot category filter">
          <button
            type="button"
            className={`button ${categoryFilter === 'All' ? 'button-primary' : 'button-outline'}`}
            onClick={() => setCategoryFilter('All')}
          >
            All Hotspots
          </button>
          <button
            type="button"
            className={`button ${categoryFilter === 'Trash' ? 'button-primary' : 'button-outline'}`}
            onClick={() => setCategoryFilter('Trash')}
          >
            Trash Hotspots Only
          </button>
          <button
            type="button"
            className={`button ${categoryFilter === 'Pothole' ? 'button-primary' : 'button-outline'}`}
            onClick={() => setCategoryFilter('Pothole')}
          >
            Pothole Hotspots Only
          </button>
        </div>
      </div>

      <div className="table-scroll">
        <table className="leaderboard-table">
          <thead>
            <tr>
              <th>RANK</th>
              <th>LOCATION / SUB-DISTRICT</th>
              <th>PRIMARY ISSUE INDICATOR</th>
              <th>ACTIVE INCIDENTS &amp; FREQUENCY</th>
              <th>HOTSPOT PRIORITY SCORE</th>
              <th>TREND STATUS</th>
              <th>ACTION</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((spot, index) => {
              const prioColor =
                spot.priorityLevel === 'CRITICAL'
                  ? '#d90429'
                  : spot.priorityLevel === 'ELEVATED'
                    ? '#f77f00'
                    : '#2b9348'
              return (
                <tr key={spot.location}>
                  <td>
                    <span className="rank-box">{index + 1}</span>
                  </td>
                  <td>
                    <strong>{spot.location}</strong>
                    <div style={{ fontSize: '11px', color: '#64748b', marginTop: '2px' }}>
                      {spot.subDistrict}
                    </div>
                  </td>
                  <td>
                    <span className="issue-indicator-pill">{spot.primaryIssue}</span>
                    <div style={{ fontSize: '11px', color: '#64748b', marginTop: '3px' }}>
                      Trash: {spot.trashObservations} obs · Pothole: {spot.potholeObservations} obs
                    </div>
                  </td>
                  <td>
                    <strong>{spot.frequency} validated observations</strong>
                    <div style={{ fontSize: '11px', color: '#64748b', marginTop: '2px' }}>
                      {spot.incidents} active incident{spot.incidents === 1 ? '' : 's'} · Latest:{' '}
                      {formatDate(spot.latest)}
                    </div>
                  </td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <strong style={{ fontSize: '16px', color: prioColor }}>
                        {spot.hotspotScore.toFixed(1)}/100
                      </strong>
                      <span
                        className="priority-badge"
                        style={{
                          borderColor: prioColor,
                          color: prioColor,
                          fontWeight: 700,
                        }}
                      >
                        {spot.priorityLevel}
                      </span>
                    </div>
                  </td>
                  <td>
                    <span className={`trend-badge trend-${spot.trendStatus.toLowerCase()}`}>
                      {spot.trendStatus}
                    </span>
                  </td>
                  <td>
                    {mode === 'user' ? (
                      <button
                        type="button"
                        className="button button-outline"
                        onClick={() => onReportHere?.(spot.location)}
                      >
                        Report an Issue Here
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="button button-outline"
                        onClick={() => onInspectIncidents?.()}
                      >
                        Dispatch / Manage
                      </button>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {!entries.length && (
        <div className="empty-state">No active validated hotspots match the selected filter.</div>
      )}
    </section>
  )
}

function CitizenReportCard({
  report,
  store,
  showCitizenBoxes,
}: {
  report: Observation
  store: Store
  showCitizenBoxes: boolean
}) {
  const [selectedFrameIdx, setSelectedFrameIdx] = useState(0)
  const incident = store.incidents.find(i => i.id === report.incidentId)
  const frames = report.dualModelResult?.frames || []
  const safeIdx = frames.length > 0 ? Math.min(selectedFrameIdx, frames.length - 1) : 0
  const activeFrame = frames.length > 0 ? frames[safeIdx] : null

  const activeRegions = activeFrame ? activeFrame.trash_findings?.region_list || [] : report.regions
  const activeDefects = activeFrame ? activeFrame.road_defect_findings?.defects || [] : report.defects
  const cleanDescription = report.description.replace(/\s*\[Frame \d+ @ [\d.]+s\]$/i, '').trim() || report.category

  return (
    <article className="panel report-card">
      <Evidence
        report={report}
        showBoxes={showCitizenBoxes}
        selectedFrameIdx={safeIdx}
        onSelectFrameIdx={setSelectedFrameIdx}
      />
      <div className="report-card-body">
        <div className="report-meta">
          <span>{report.id}</span>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
            <VerificationLifecycleBadge review={report.review} />
            <Status value={incident?.status || report.review} />
          </div>
        </div>
        <h2>{report.category}</h2>
        <p className="location-copy">{report.location}</p>
        <p>{cleanDescription}</p>
        <small>
          Observed {formatDate(report.observedAt)} ·{' '}
          {frames.length > 1
            ? `Video evidence (${frames.length} extracted frames · Viewing 00:${String(
                Math.floor(activeFrame?.timestamp_sec ?? 0),
              ).padStart(2, '0')})`
            : report.source}
        </small>

        {/* Model A Trash Findings */}
        {activeRegions && activeRegions.length > 0 && (
          <div style={{ marginTop: '10px' }}>
            <small style={{ fontWeight: 700, textTransform: 'uppercase', color: '#475569' }}>
              AI Trash Detection Highlights (Model A)
            </small>
            {activeRegions.map(r => (
              <RegionBadge key={r.region_id} region={r} />
            ))}
          </div>
        )}

        {/* Model B Pothole Findings */}
        {activeDefects && activeDefects.length > 0 && (
          <div style={{ marginTop: '10px' }}>
            <small style={{ fontWeight: 700, textTransform: 'uppercase', color: '#475569' }}>
              AI Road Defect Highlights (Model B)
            </small>
            {activeDefects.map(d => (
              <DefectBadge key={d.defect_id} defect={d} />
            ))}
          </div>
        )}

        {report.hotspotScore !== undefined && (
          <div style={{ marginTop: '8px', fontSize: '12px', fontWeight: 600 }}>
            <span>Municipal Hotspot Priority: </span>
            <span
              style={{
                color:
                  report.priorityLevel === 'CRITICAL'
                    ? '#d90429'
                    : report.priorityLevel === 'ELEVATED'
                      ? '#f77f00'
                      : '#2b9348',
              }}
            >
              {report.hotspotScore}/100 [{report.priorityLevel}]
            </span>
          </div>
        )}

        <div className="report-update">
          <strong>{incident ? `Barangay Operations · ${incident.id}` : 'Verification Status'}</strong>
          <p>
            {incident?.note ||
              report.reviewNote ||
              'Your report is queued for barangay verification. Once validated, it updates municipal dispatch priorities.'}
          </p>
          {incident && (
            <small>
              Updated {formatDate(incident.updatedAt)} · {report.reviewNote}
            </small>
          )}
        </div>
        <div className="report-meta">
          <Status value={report.review} />
          {(!activeRegions || activeRegions.length === 0) &&
            (!activeDefects || activeDefects.length === 0) && (
              <small>
                Automated Model Detection (v5.2): {report.category} · {report.confidence}% AI Confidence
              </small>
            )}
        </div>
      </div>
    </article>
  )
}

export function MyReports({ store }: { store: Store }) {
  const [showCitizenBoxes, setShowCitizenBoxes] = useState(true)
  const [query, setQuery] = useState('')
  const [categoryFilter, setCategoryFilter] = useState('All categories')
  const [statusFilter, setStatusFilter] = useState('All statuses')
  const [dateRange, setDateRange] = useState('All time')

  const reports = store.observations
    .filter(o => o.reporter === residentId)
    .filter(o => {
      if (categoryFilter !== 'All categories' && o.category !== categoryFilter) return false
      const incident = store.incidents.find(i => i.id === o.incidentId)
      const effectiveStatus = incident?.status || o.review
      if (statusFilter !== 'All statuses' && effectiveStatus !== statusFilter) return false
      if (!withinDateRange(o.observedAt, dateRange)) return false
      if (query.trim()) {
        const hay = `${o.id} ${o.incidentId || ''} ${o.location} ${o.description} ${o.category}`.toLowerCase()
        if (!hay.includes(query.trim().toLowerCase())) return false
      }
      return true
    })
    .sort((a, b) => b.submittedAt.localeCompare(a.submittedAt))

  return (
    <>
      <SearchAndFilterBar
        query={query}
        onQueryChange={setQuery}
        categoryFilter={categoryFilter}
        onCategoryChange={setCategoryFilter}
        statusFilter={statusFilter}
        onStatusChange={setStatusFilter}
        statusOptions={['Pending review', 'Validated', 'Rejected', ...incidentStatuses]}
        dateRange={dateRange}
        onDateRangeChange={setDateRange}
        totalCount={reports.length}
        countLabel="citizen reports"
        extraToggle={
          <label className="citizen-box-toggle">
            <input
              type="checkbox"
              checked={showCitizenBoxes}
              onChange={e => setShowCitizenBoxes(e.target.checked)}
            />
            <span>Show AI Detection Highlights</span>
          </label>
        }
      />

      <div className="report-cards">
        {reports.map(report => (
          <CitizenReportCard
            key={report.id}
            report={report}
            store={store}
            showCitizenBoxes={showCitizenBoxes}
          />
        ))}
        {!reports.length && (
          <div className="panel empty-state">
            No citizen reports match your filter criteria. Submit a new observation to get started.
          </div>
        )}
      </div>
    </>
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
  const [selectedFrameIdx, setSelectedFrameIdx] = useState(0)
  const activeTarget = candidates.some(i => i.id === target) ? target : ''

  const frames = report.dualModelResult?.frames || []
  const safeIdx = frames.length > 0 ? Math.min(selectedFrameIdx, frames.length - 1) : 0
  const activeFrame = frames.length > 0 ? frames[safeIdx] : null
  const activeRegions = activeFrame ? activeFrame.trash_findings?.region_list || [] : report.regions
  const activeDefects = activeFrame ? activeFrame.road_defect_findings?.defects || [] : report.defects
  const cleanDescription = report.description.replace(/\s*\[Frame \d+ @ [\d.]+s\]$/i, '').trim() || report.category

  return (
    <article className="panel review-card">
      <Evidence
        report={report}
        showBoxes={true}
        selectedFrameIdx={safeIdx}
        onSelectFrameIdx={setSelectedFrameIdx}
      />
      <div className="review-body">
        <div className="report-meta">
          <span>
            {report.id} · {report.reporter}
          </span>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <VerificationLifecycleBadge review={report.review} />
            <Status value={report.review} />
          </div>
        </div>
        <h2>{report.category}</h2>
        <p className="location-copy">{report.location}</p>
        <p>{cleanDescription}</p>
        <small>
          Observed {formatDate(report.observedAt)} · Submitted {formatDate(report.submittedAt)} ·{' '}
          {frames.length > 1
            ? `Video evidence (${frames.length} extracted frames · Viewing 00:${String(
                Math.floor(activeFrame?.timestamp_sec ?? 0),
              ).padStart(2, '0')})`
            : report.source}
        </small>

        {/* AI Model Findings */}
        {activeRegions && activeRegions.length > 0 ? (
          <div style={{ margin: '10px 0', padding: '10px 12px', background: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
            <strong style={{ display: 'block', fontSize: '13px', marginBottom: '4px' }}>
              Model A: D-112 Spatial Grouping Findings
            </strong>
            {activeRegions.map(r => (
              <RegionBadge key={r.region_id} region={r} />
            ))}
          </div>
        ) : null}

        {activeDefects && activeDefects.length > 0 ? (
          <div style={{ margin: '10px 0', padding: '10px 12px', background: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
            <strong style={{ display: 'block', fontSize: '13px', marginBottom: '4px' }}>
              Model B: RF-DETR Cell 8A Pothole Findings
            </strong>
            {activeDefects.map(d => (
              <DefectBadge key={d.defect_id} defect={d} />
            ))}
          </div>
        ) : null}

        {report.hotspotScore !== undefined ? (
          <div style={{ margin: '8px 0', fontSize: '13px' }}>
            <strong>Calibrated Hotspot Priority Score: </strong>
            <span
              style={{
                fontWeight: 700,
                color:
                  report.priorityLevel === 'CRITICAL'
                    ? '#d90429'
                    : report.priorityLevel === 'ELEVATED'
                      ? '#f77f00'
                      : '#2b9348',
              }}
            >
              {report.hotspotScore}/100 [{report.priorityLevel}]
            </span>
          </div>
        ) : (
          <div className="detection-line">
            <strong>Automated Model Detection (v5.2)</strong>
            <span>
              {report.category} · {report.confidence}% AI Confidence
            </span>
            <small>Model confidence assists operator verification before incident linking.</small>
          </div>
        )}

        {candidates.length > 0 && (
          <p className="match-note">
            Possible duplicate: {candidates.length} active incident{candidates.length === 1 ? '' : 's'}{' '}
            with the same category and location. Confirm the same physical issue before linking.
          </p>
        )}
        <label className="form-field">
          Consolidate this observation
          <select value={activeTarget} onChange={e => setTarget(e.target.value)}>
            <option value="">Create a separate incident</option>
            {candidates.map(i => (
              <option key={i.id} value={i.id}>
                Link to {i.id} · {i.category} · {i.status}
              </option>
            ))}
          </select>
        </label>
        <label className="form-field">
          Review note <span className="muted">(required when rejecting)</span>
          <textarea
            maxLength={600}
            rows={2}
            value={note}
            onChange={e => setNote(e.target.value)}
            placeholder="Provide verification notes or explanation for the citizen"
          />
        </label>
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
        <div className="review-actions">
          <button
            className="button button-outline"
            onClick={() => {
              if (!note.trim()) {
                setError('Add a reason before rejecting this report.')
                return
              }
              onReview(report.id, 'reject', undefined, note)
            }}
          >
            Reject report
          </button>
          <button
            className="button button-primary"
            onClick={() => onReview(report.id, 'validate', activeTarget || undefined, note)}
          >
            {activeTarget ? 'Validate & link observation' : 'Validate & create incident'}
          </button>
        </div>
      </div>
    </article>
  )
}

export function ReviewQueue({
  store,
  onReview,
}: {
  store: Store
  onReview: (id: string, decision: 'validate' | 'reject', target: string | undefined, note: string) => void
}) {
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState('All categories')
  const [dateRange, setDateRange] = useState('All time')

  const pending = store.observations.filter(o => {
    if (o.review !== 'Pending review') return false
    if (filter !== 'All categories' && o.category !== filter) return false
    if (!withinDateRange(o.observedAt, dateRange)) return false
    if (query.trim()) {
      const hay = `${o.id} ${o.location} ${o.description} ${o.reporter} ${o.category}`.toLowerCase()
      if (!hay.includes(query.trim().toLowerCase())) return false
    }
    return true
  })

  return (
    <>
      <SearchAndFilterBar
        query={query}
        onQueryChange={setQuery}
        categoryFilter={filter}
        onCategoryChange={setFilter}
        dateRange={dateRange}
        onDateRangeChange={setDateRange}
        totalCount={pending.length}
        countLabel="observations awaiting review"
      />
      <div className="report-cards">
        {pending.map(report => (
          <ReviewCard
            key={`${report.id}-${duplicateCandidates(store, report)
              .map(i => i.id)
              .join('-')}`}
            store={store}
            report={report}
            onReview={onReview}
          />
        ))}
        {!pending.length && (
          <div className="panel empty-state">
            <h2>Review queue clear</h2>
            <p>
              No pending observations match this filter. Switch to Resident Portal to submit a new field report.
            </p>
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
            {id} · {incident.location}
          </small>
        </span>
        <span className="incident-summary-end">
          <span>{evidence.length} observations</span>
          <Status value={incident.status} />
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.7"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
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
              Incident status
              <select value={status} onChange={e => setStatus(e.target.value as IncidentStatus)}>
                {incidentStatuses.map(s => (
                  <option key={s}>{s}</option>
                ))}
              </select>
            </label>
            <label className="form-field">
              Operational status update
              <textarea
                required
                maxLength={600}
                rows={2}
                value={note}
                onChange={e => setNote(e.target.value)}
              />
            </label>
          </div>
          <button className="button button-primary" disabled={!note.trim()}>
            Save status update
          </button>
        </form>
        <h3>Consolidated evidence ({evidence.length})</h3>
        <div className="evidence-list">
          {evidence.map(o => (
            <div className="evidence-entry" key={o.id}>
              <Evidence report={o} showBoxes={true} />
              <div>
                <strong>{o.id}</strong>
                <p>{o.description}</p>
                <small>
                  {o.source} · {formatDate(o.observedAt)} · Automated Model Detection ({o.confidence}% AI Confidence)
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
  const [categoryFilter, setCategoryFilter] = useState('All categories')
  const [filter, setFilter] = useState('All statuses')
  const [dateRange, setDateRange] = useState('All time')

  const incidents = store.incidents.filter(i => {
    if (categoryFilter !== 'All categories' && i.category !== categoryFilter) return false
    if (filter !== 'All statuses' && i.status !== filter) return false
    if (!withinDateRange(i.updatedAt, dateRange)) return false
    if (query.trim()) {
      const hay = `${i.id} ${i.category} ${i.location} ${i.note}`.toLowerCase()
      if (!hay.includes(query.trim().toLowerCase())) return false
    }
    return true
  })

  return (
    <>
      <SearchAndFilterBar
        query={query}
        onQueryChange={setQuery}
        categoryFilter={categoryFilter}
        onCategoryChange={setCategoryFilter}
        statusFilter={filter}
        onStatusChange={setFilter}
        statusOptions={incidentStatuses}
        dateRange={dateRange}
        onDateRangeChange={setDateRange}
        totalCount={incidents.length}
        countLabel="incident records"
      />
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
          <div className="panel empty-state">No incidents match your search and filter criteria.</div>
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
  const [category, setCategory] = useState<Category>('Trash')
  const [generated, setGenerated] = useState(false)
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoPreviewUrl, setVideoPreviewUrl] = useState('')
  const [processingVideo, setProcessingVideo] = useState(false)
  const [videoError, setVideoError] = useState('')
  const videoInputRef = useRef<HTMLInputElement>(null)

  function handleSelectVideo(file: File | null) {
    setVideoFile(file)
    setVideoError('')
    if (file) {
      setVideoPreviewUrl(URL.createObjectURL(file))
    } else {
      setVideoPreviewUrl('')
    }
  }

  async function handleRealVideoProcess() {
    if (!videoFile) return
    setProcessingVideo(true)
    setVideoError('')

    try {
      const formData = new FormData()
      formData.append('file', videoFile)
      formData.append('location', location)
      formData.append('category', category)
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
        const serverVideoUrl = result.source_media?.media_url || videoPreviewUrl
        const firstFrame = result.frames[0]
        const singleObs: Observation = {
          id: `VID-${result.incident_id.slice(-6)}`,
          source: 'Video observation',
          reporter: 'Patrol Camera 01',
          location,
          category,
          observedAt: now,
          submittedAt: now,
          description: `1.0 FPS patrol video analysis (${result.frames.length} extracted frame${
            result.frames.length === 1 ? '' : 's'
          }). Category-selective ${
            category === 'Trash' ? 'Model A Trash' : 'Model B Pothole Cell 8A'
          } detection.`,
          photo: firstFrame.frame_url,
          videoUrl: serverVideoUrl,
          clipTime: `00:${String(Math.floor(firstFrame.timestamp_sec)).padStart(2, '0')}`,
          confidence: 90,
          review: 'Pending review',
          reviewNote: '',
          dualModelResult: result,
          detections: firstFrame.trash_findings?.detections || [],
          regions: firstFrame.trash_findings?.region_list || [],
          imageWidth: firstFrame.width || 1080,
          imageHeight: firstFrame.height || 1920,
          defects: firstFrame.road_defect_findings?.defects || [],
          hotspotScore: result.hotspot_score ?? firstFrame.frame_hotspot_score,
          priorityLevel: result.priority_level ?? firstFrame.frame_priority_level,
        }

        onAddRealObservations([singleObs])
        setGenerated(true)
      }
    } catch (err) {
      setVideoError(
        `Could not analyze video with backend server: ${
          err instanceof Error ? err.message : String(err)
        }. Ensure the backend server on port 8000 is running.`,
      )
    } finally {
      setProcessingVideo(false)
    }
  }

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <div className="section-index">PATROL CAMERA &amp; VIDEO INGESTION</div>
          <h2>Single-pass 1.0 FPS video detection pipeline</h2>
          <p>
            Upload patrol dashcam or street-inspection footage for category-targeted AI extraction and frame-by-frame triage.
          </p>
        </div>
      </div>
      <div className="form-body">
        <div
          style={{
            padding: '20px',
            background: '#f8fafc',
            borderRadius: '8px',
            border: '1px solid #cbd5e1',
            marginBottom: '20px',
          }}
        >
          <strong style={{ display: 'block', fontSize: '15px', marginBottom: '6px', color: '#0f172a' }}>
            Run Category-Targeted Video Detection
          </strong>
          <p style={{ fontSize: '13px', color: '#475569', marginBottom: '14px' }}>
            Select the target category and upload an MP4/MOV video. Selecting <strong>Trash</strong> runs Model A (Grounding DINO-T 3-Model Consensus + D-112 Spatial Grouping); selecting <strong>Pothole</strong> runs Model B (RF-DETR Nano + Cell 8A Vision Engine).
          </p>

          <div className="form-grid" style={{ marginTop: 0, marginBottom: '14px' }}>
            <label className="form-field">
              Patrol Camera Location
              <select value={location} onChange={e => setLocation(e.target.value)}>
                {locations.map(l => (
                  <option key={l}>{l}</option>
                ))}
              </select>
            </label>
            <label className="form-field">
              Target Issue Category &amp; AI Model
              <select value={category} onChange={e => setCategory(e.target.value as Category)}>
                {categories.map(c => (
                  <option key={c} value={c}>
                    {c === 'Trash'
                      ? 'Trash (Model A · Grounding DINO-T Stage 5)'
                      : 'Pothole (Model B · RF-DETR Nano Cell 8A)'}
                  </option>
                ))}
              </select>
            </label>
          </div>

          {videoPreviewUrl && (
            <div style={{ marginBottom: '14px' }}>
              <video
                src={videoPreviewUrl}
                controls
                muted
                style={{ maxWidth: '360px', maxHeight: '200px', borderRadius: '6px', border: '1px solid #cbd5e1' }}
              />
            </div>
          )}

          <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
            <input
              ref={videoInputRef}
              type="file"
              accept="video/mp4,video/quicktime,video/webm"
              style={{ display: 'none' }}
              onChange={e => handleSelectVideo(e.target.files?.[0] || null)}
            />
            <button
              type="button"
              className="button button-outline"
              onClick={() => videoInputRef.current?.click()}
            >
              {videoFile ? `Selected: ${videoFile.name}` : 'Select video file (.mp4)'}
            </button>
            <button
              type="button"
              className="button button-primary"
              disabled={!videoFile || processingVideo}
              onClick={handleRealVideoProcess}
            >
              {processingVideo
                ? `Extracting 1.0 FPS & Running ${
                    category === 'Trash' ? 'Model A (Trash)' : 'Model B (Pothole)'
                  }…`
                : `Process Video (${category} Model)`}
            </button>
          </div>
          {videoError && (
            <p className="form-error" style={{ marginTop: '10px' }}>
              {videoError}
            </p>
          )}
        </div>

        <div className="field-label" style={{ marginBottom: '8px' }}>
          Automated Patrol Camera Batch Ingestion
        </div>
        <div className="video-storyboard">
          <div>
            <span>00:04</span>
            <strong>Frame capture (1.0 FPS)</strong>
            <small>Shared single-pass extraction</small>
          </div>
          <div>
            <span>00:12</span>
            <strong>Category AI detection</strong>
            <small>Calibrated operating threshold</small>
          </div>
          <div>
            <span>00:23</span>
            <strong>Review queue staging</strong>
            <small>Synchronized frame &amp; bbox overlay</small>
          </div>
        </div>
        <div className="review-actions" style={{ marginTop: '16px' }}>
          <button
            className="button button-outline"
            onClick={() => {
              onGenerate(location, category)
              setGenerated(true)
            }}
          >
            Ingest patrol camera batch
          </button>
          {generated && (
            <button className="button button-primary" onClick={onReview}>
              Go to review queue
            </button>
          )}
        </div>
      </div>
    </section>
  )
}
