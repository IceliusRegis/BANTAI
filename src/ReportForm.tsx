import { useEffect, useRef, useState } from 'react'
import { categories, locations, computeHotspotScore } from './incidents'
import type {
  Category,
  NewReport,
  UnifiedIncidentResponse,
  TrashRegion,
  RoadDefect,
  UiDetectionBox,
  UiVideoFrame,
  TrashDetection,
} from './incidents'
import { MediaDetectionViewer } from './IncidentPanels'

function localTime() {
  const date = new Date()
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}T${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`
}

export type ReportSubmitData = NewReport & {
  mediaType?: 'image' | 'video'
  dualModelResult?: UnifiedIncidentResponse
  regions?: TrashRegion[]
  detections?: TrashDetection[]
  defects?: RoadDefect[]
  hotspotScore?: number
  priorityLevel?: 'CRITICAL' | 'ELEVATED' | 'ROUTINE'
  boxes?: UiDetectionBox[]
  frames?: UiVideoFrame[]
  potholeCount?: number
  trashCount?: number
  previewVideoUrl?: string
  imageWidth?: number
  imageHeight?: number
}

export default function ReportForm({ onSubmit }: { onSubmit: (report: ReportSubmitData) => void }) {
  const [category, setCategory] = useState<Category>('Pothole')
  const [location, setLocation] = useState(locations[0])
  const [otherLocation, setOtherLocation] = useState('')
  const [description, setDescription] = useState('')
  const [time, setTime] = useState(localTime)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [photo, setPhoto] = useState('')
  const [filename, setFilename] = useState('')
  const [mediaType, setMediaType] = useState<'image' | 'video'>('image')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [scanningStatus, setScanningStatus] = useState('')

  // Scanned AI findings
  const [boxes, setBoxes] = useState<UiDetectionBox[]>([])
  const [frames, setFrames] = useState<UiVideoFrame[]>([])
  const [potholeCount, setPotholeCount] = useState<number>(0)
  const [trashCount, setTrashCount] = useState<number>(0)
  const [previewVideoUrl, setPreviewVideoUrl] = useState<string | undefined>(undefined)
  const [dualModelResult, setDualModelResult] = useState<UnifiedIncidentResponse | undefined>(undefined)
  const [regions, setRegions] = useState<TrashRegion[]>([])
  const [detections, setDetections] = useState<TrashDetection[]>([])
  const [defects, setDefects] = useState<RoadDefect[]>([])
  const [imageWidth, setImageWidth] = useState<number | undefined>(undefined)
  const [imageHeight, setImageHeight] = useState<number | undefined>(undefined)

  const selection = useRef(0)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => () => { selection.current += 1 }, [])

  async function processSelectedFile(file: File) {
    const token = ++selection.current
    setError('')
    setPhoto('')
    setFilename(file.name)
    setSelectedFile(file)
    setBoxes([])
    setFrames([])
    setPotholeCount(0)
    setTrashCount(0)
    setRegions([])
    setDetections([])
    setDefects([])
    setPreviewVideoUrl(undefined)
    setDualModelResult(undefined)

    const isImage =
      ['image/jpeg', 'image/png', 'image/webp'].includes(file.type) ||
      file.name.match(/\.(jpg|jpeg|png|webp)$/i)
    const isVideo =
      ['video/mp4', 'video/webm', 'video/quicktime'].includes(file.type) ||
      file.name.match(/\.(mp4|mov|webm)$/i)

    if (!isImage && !isVideo) {
      setError('Please choose a valid photo (JPG, PNG) or video (MP4, MOV).')
      return
    }

    if (file.size > 80 * 1024 * 1024) {
      setError('File size must be under 80 MB.')
      return
    }

    setMediaType(isVideo ? 'video' : 'image')
    setLoading(true)

    // Local instant preview
    if (isImage) {
      try {
        const localPreview = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader()
          reader.onload = () => resolve(String(reader.result))
          reader.onerror = () => reject(new Error('Image could not be read.'))
          reader.readAsDataURL(file)
        })
        if (selection.current === token) setPhoto(localPreview)
      } catch {
        if (selection.current === token) setError('Unable to read selected image.')
      }
    } else {
      const localPreview = URL.createObjectURL(file)
      if (selection.current === token) setPhoto(localPreview)
    }

    // Call dual-model analysis endpoint
    setScanningStatus(
      isVideo
        ? 'Extracting frames & running dual-model detection…'
        : 'Running road defect & sanitation analysis…'
    )

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append(
        'location',
        location === 'Other location' ? otherLocation.trim() || 'Unspecified Street' : location
      )
      formData.append('sample_fps', '1.0')

      const res = await fetch('/api/v1/analyze', {
        method: 'POST',
        body: formData,
      })

      if (selection.current !== token) return

      if (res.ok) {
        const data: UnifiedIncidentResponse = await res.json()
        setDualModelResult(data)

        if (isVideo) {
          const mappedFrames: UiVideoFrame[] = (data.frames || []).map((f, i) => ({
            frameIndex: i + 1,
            timestampSec: f.timestamp_sec,
            clipTime: `00:${String(Math.floor(f.timestamp_sec)).padStart(2, '0')}`,
            frameUrl: f.frame_url,
            boxes: f.boxes || [],
            potholeCount: f.pothole_count ?? (f.road_defect_findings?.defects?.length || 0),
            trashCount: f.trash_count ?? (f.trash_findings?.total_items || 0),
            regions: f.trash_findings?.region_list || [],
            detections: f.trash_findings?.detections || [],
            defects: f.road_defect_findings?.defects || [],
            width: f.width,
            height: f.height,
            frameHotspotScore: f.frame_hotspot_score,
            framePriorityLevel: f.frame_priority_level,
          }))

          setFrames(mappedFrames)
          const totalP =
            data.summary?.total_potholes ??
            mappedFrames.reduce((sum, f) => sum + f.potholeCount, 0)
          const totalT =
            data.summary?.total_trash_items ??
            mappedFrames.reduce((sum, f) => sum + f.trashCount, 0)

          setPotholeCount(totalP)
          setTrashCount(totalT)
          setBoxes(mappedFrames[0]?.boxes || [])
          setRegions(mappedFrames[0]?.regions || [])
          setDetections(mappedFrames[0]?.detections || [])
          setDefects(mappedFrames[0]?.defects || [])
          setImageWidth(mappedFrames[0]?.width)
          setImageHeight(mappedFrames[0]?.height)

          if (mappedFrames[0]?.frameUrl) setPhoto(mappedFrames[0].frameUrl)
          if (data.preview_video_url) setPreviewVideoUrl(data.preview_video_url)

          if (totalP > 0 && totalT === 0) setCategory('Pothole')
          else if (totalT > 0 && totalP === 0) setCategory('Garbage pile')
        } else {
          // Photo
          const f0 = data.frames && data.frames[0]
          const detectedBoxes = f0?.boxes || []
          const pFound =
            f0?.pothole_count ?? (f0?.road_defect_findings?.defects?.length || 0)
          const tFound =
            f0?.trash_count ?? (f0?.trash_findings?.total_items || 0)

          setBoxes(detectedBoxes)
          setPotholeCount(pFound)
          setTrashCount(tFound)
          setRegions(f0?.trash_findings?.region_list || [])
          setDetections(f0?.trash_findings?.detections || [])
          setDefects(f0?.road_defect_findings?.defects || [])
          setImageWidth(f0?.width)
          setImageHeight(f0?.height)

          if (f0?.frame_url) setPhoto(f0.frame_url)

          if (pFound > 0 && tFound === 0) setCategory('Pothole')
          else if (tFound > 0 && pFound === 0) setCategory('Garbage pile')
        }
      }
    } catch (err) {
      // Graceful offline message
      console.warn('Backend server unavailable:', err)
    } finally {
      if (selection.current === token) {
        setLoading(false)
        setScanningStatus('')
      }
    }
  }

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    const place = location === 'Other location' ? otherLocation.trim() : location
    const observed = new Date(time)

    if (!place) {
      setError('Please specify the street location or nearby landmark.')
      return
    }
    if (!description.trim()) {
      setError('Please provide a brief description of the observed condition.')
      return
    }
    if (!photo && !selectedFile) {
      setError('Please attach a photo or video before submitting.')
      return
    }
    if (!Number.isFinite(observed.getTime()) || observed.getTime() > Date.now()) {
      setError('Observed time must be valid and cannot be in the future.')
      return
    }

    const calculatedScore =
      dualModelResult?.hotspot_score ??
      computeHotspotScore(trashCount > 0 ? 1 : 0, 0, 0, { high: potholeCount })

    const calculatedPriority: 'CRITICAL' | 'ELEVATED' | 'ROUTINE' =
      dualModelResult?.priority_level ??
      (potholeCount >= 2 || calculatedScore >= 70
        ? 'CRITICAL'
        : potholeCount > 0 || trashCount > 0
        ? 'ELEVATED'
        : 'ROUTINE')

    onSubmit({
      category,
      location: place,
      observedAt: observed.toISOString(),
      description: description.trim(),
      photo: photo || undefined,
      mediaType,
      boxes,
      frames: frames.length > 0 ? frames : undefined,
      potholeCount,
      trashCount,
      previewVideoUrl,
      dualModelResult,
      regions,
      detections,
      defects,
      hotspotScore: calculatedScore,
      priorityLevel: calculatedPriority,
      imageWidth,
      imageHeight,
    })
  }

  const hasMedia = Boolean(photo || selectedFile)

  return (
    <div className="submit-layout">
      <section className="panel">
        <div className="panel-heading">
          <div>
            <div className="section-index">COMMUNITY REPORTING</div>
            <h2>Submit an Incident Report</h2>
            <p>Report roadside potholes, road surface damage, or waste accumulation for barangay maintenance review.</p>
          </div>
        </div>

        <form className="form-body" onSubmit={handleSubmit}>
          {/* STEP 1: Media Attachment */}
          <div className="field-label">
            Step 1: Supporting Photo or Video <span>*</span>
          </div>

          <div className="dropzone" style={{ padding: '16px' }}>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp,video/mp4,video/webm,video/quicktime"
              style={{ display: 'none' }}
              onChange={e => {
                const f = e.target.files?.[0]
                if (f) void processSelectedFile(f)
              }}
            />

            {hasMedia ? (
              <div style={{ width: '100%', marginBottom: '12px' }}>
                <MediaDetectionViewer
                  photo={photo}
                  mediaType={mediaType}
                  boxes={boxes}
                  frames={frames}
                  potholeCount={potholeCount}
                  trashCount={trashCount}
                  previewVideoUrl={previewVideoUrl}
                  regions={regions}
                  detections={detections}
                  imageWidth={imageWidth}
                  imageHeight={imageHeight}
                  showAdminOverlay={false}
                />
              </div>
            ) : (
              <div
                className="upload-dropzone"
                style={{
                  cursor: 'pointer',
                  padding: '32px 16px',
                  background: '#f8faf9',
                  border: '2px dashed #b8d0c6',
                  borderRadius: '6px',
                }}
                onClick={() => fileInputRef.current?.click()}
              >
                <span style={{ fontSize: '32px', marginBottom: '8px' }}>📷 / 🎬</span>
                <strong style={{ fontSize: '15px', color: '#134e48' }}>
                  Click to select photo or video clip
                </strong>
                <small style={{ color: '#666' }}>Supports JPG, PNG, MP4, MOV</small>
              </div>
            )}

            <div
              style={{
                display: 'flex',
                gap: '10px',
                alignItems: 'center',
                flexWrap: 'wrap',
                marginTop: '10px',
              }}
            >
              <button
                type="button"
                className="button button-primary"
                onClick={() => fileInputRef.current?.click()}
                disabled={loading}
              >
                {hasMedia ? 'Change File' : 'Select File'}
              </button>
            </div>

            {filename && (
              <small style={{ display: 'block', marginTop: '6px', color: '#555' }}>
                Attached: {filename}
              </small>
            )}
            {scanningStatus && (
              <p className="notice" role="status" style={{ marginTop: '10px', fontWeight: 600 }}>
                ⏳ {scanningStatus}
              </p>
            )}
          </div>

          {/* STEP 2: Location and Description */}
          <div className="field-label" style={{ marginTop: '18px' }}>
            Step 2: Incident Location & Details <span>*</span>
          </div>

          <div className="form-grid">
            <label className="form-field">
              Street Corridor
              <select value={location} onChange={e => setLocation(e.target.value)}>
                {[...locations, 'Other location'].map(l => (
                  <option key={l}>{l}</option>
                ))}
              </select>
            </label>

            <label className="form-field">
              Report Category
              <select value={category} onChange={e => setCategory(e.target.value as Category)}>
                {categories.map(c => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </label>

            {location === 'Other location' && (
              <label className="form-field full-field">
                Specific street or landmark
                <input
                  required
                  maxLength={120}
                  value={otherLocation}
                  onChange={e => setOtherLocation(e.target.value)}
                  placeholder="e.g. Near corner market, opposite community clinic"
                />
              </label>
            )}

            <label className="form-field full-field">
              Observed Date & Time
              <input
                required
                type="datetime-local"
                max={localTime()}
                value={time}
                onChange={e => setTime(e.target.value)}
              />
            </label>

            <label className="form-field full-field">
              Detailed Description
              <textarea
                required
                maxLength={1000}
                rows={3}
                value={description}
                onChange={e => setDescription(e.target.value)}
                placeholder="Describe the issue, estimated dimensions, or impact on vehicular/pedestrian traffic."
              />
            </label>
          </div>

          {error && (
            <p className="form-error" role="alert" style={{ marginTop: '12px' }}>
              {error}
            </p>
          )}

          <div className="form-actions" style={{ marginTop: '16px' }}>
            <span style={{ fontSize: '13px', color: '#555' }}>
              Submissions are transmitted securely to the Barangay Santa Elena operations desk.
            </span>
            <button
              type="submit"
              className="button button-primary"
              style={{ fontSize: '14px', padding: '10px 22px', fontWeight: 750 }}
              disabled={loading || !hasMedia}
            >
              {loading ? 'Processing…' : 'Submit Report'}
            </button>
          </div>
        </form>
      </section>

      <aside className="panel guidance-panel">
        <div className="section-index">CIVIC REPORTING</div>
        <h2>Reporting Guidelines</h2>

        <div className="guide-item">
          <span>01</span>
          <p>
            <strong>Clear Visibility:</strong> Ensure the road defect or waste materials are centered and well-lit.
          </p>
        </div>

        <div className="guide-item">
          <span>02</span>
          <p>
            <strong>Automated Triage:</strong> The system automatically extracts bounding boxes, identifies materials, and indexes severity to recommend appropriate dispatch.
          </p>
        </div>

        <div className="guide-item">
          <span>03</span>
          <p>
            <strong>Track Progress:</strong> Monitor the lifecycle of your report under <strong>"My Reports"</strong> as maintenance teams inspect and resolve issues.
          </p>
        </div>

        <div
          style={{
            marginTop: '20px',
            padding: '12px',
            background: '#f0fdf4',
            borderRadius: '6px',
            border: '1px solid #bbf7d0',
          }}
        >
          <strong style={{ display: 'block', color: '#166534', fontSize: '12px' }}>
            Verified Citizen Submission
          </strong>
          <small style={{ color: '#15803d', fontSize: '11px' }}>
            Your account is authenticated for Barangay Santa Elena resident reporting.
          </small>
        </div>
      </aside>
    </div>
  )
}
