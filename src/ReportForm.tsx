import { useEffect, useRef, useState } from 'react'
import { categories, locations } from './incidents'
import type { Category, NewReport, UnifiedIncidentResponse, TrashRegion, TrashDetection, RoadDefect } from './incidents'

function localTime() {
  const date = new Date()
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16)
}

export type ReportSubmitData = NewReport & {
  dualModelResult?: UnifiedIncidentResponse
  regions?: TrashRegion[]
  defects?: RoadDefect[]
  hotspotScore?: number
  priorityLevel?: 'CRITICAL' | 'ELEVATED' | 'ROUTINE'
}

export default function ReportForm({
  onSubmit,
  initialLocation,
}: {
  onSubmit: (report: ReportSubmitData) => void
  initialLocation?: string
}) {
  const [category, setCategory] = useState<Category>('Trash')
  const [location, setLocation] = useState(() =>
    initialLocation && locations.includes(initialLocation) ? initialLocation : locations[0],
  )
  const [otherLocation, setOtherLocation] = useState(() =>
    initialLocation && !locations.includes(initialLocation) ? initialLocation : '',
  )
  const [description, setDescription] = useState('')
  const [time, setTime] = useState(localTime)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [photo, setPhoto] = useState('')
  const [videoPreviewUrl, setVideoPreviewUrl] = useState('')
  const [filename, setFilename] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [analyzingStatus, setAnalyzingStatus] = useState('')
  const selection = useRef(0)

  useEffect(() => {
    if (initialLocation) {
      if (locations.includes(initialLocation)) {
        setLocation(initialLocation)
      } else {
        setLocation('Other location')
        setOtherLocation(initialLocation)
      }
    }
  }, [initialLocation])

  useEffect(() => () => {
    selection.current += 1
  }, [])

  async function chooseFile(file?: File) {
    const token = ++selection.current
    setError('')
    setPhoto('')
    setVideoPreviewUrl('')
    setFilename('')
    setSelectedFile(null)
    if (!file) {
      setLoading(false)
      return
    }

    const isImage = ['image/jpeg', 'image/png', 'image/webp'].includes(file.type)
    const isVideo =
      ['video/mp4', 'video/webm', 'video/quicktime'].includes(file.type) ||
      Boolean(file.name.match(/\.(mp4|mov|webm)$/i))

    if (!isImage && !isVideo) {
      setError('Select a JPG, PNG, WEBP image or an MP4/MOV/WEBM video file.')
      return
    }

    if (file.size > 50 * 1024 * 1024) {
      setError('File must be smaller than 50 MB.')
      return
    }

    setSelectedFile(file)
    setFilename(file.name)
    setLoading(true)

    try {
      if (isImage) {
        const data = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader()
          reader.onload = () => resolve(String(reader.result))
          reader.onerror = () => reject(new Error('Image could not be read.'))
          reader.readAsDataURL(file)
        })
        if (selection.current === token) {
          setPhoto(data)
        }
      } else {
        const objUrl = URL.createObjectURL(file)
        if (selection.current === token) {
          setVideoPreviewUrl(objUrl)
        }
      }
    } catch {
      if (selection.current === token) {
        setError('File could not be opened. Select another media file.')
      }
    } finally {
      if (selection.current === token) {
        setLoading(false)
      }
    }
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    const place = location === 'Other location' ? otherLocation.trim() : location
    const observed = new Date(time)
    if (!place || !description.trim() || (!photo && !selectedFile)) {
      setError('Select a supporting photo or video file, location, and description before submitting.')
      return
    }
    if (!Number.isFinite(observed.getTime()) || observed.getTime() > Date.now()) {
      setError('The observation time must be valid and cannot be in the future.')
      return
    }

    let dualModelResult: UnifiedIncidentResponse | undefined = undefined
    let detections: TrashDetection[] | undefined = undefined
    let regions: TrashRegion[] | undefined = undefined
    let imageWidth: number | undefined = undefined
    let imageHeight: number | undefined = undefined
    let defects: RoadDefect[] | undefined = undefined
    let hotspotScore: number | undefined = undefined
    let priorityLevel: 'CRITICAL' | 'ELEVATED' | 'ROUTINE' | undefined = undefined
    let primaryPhoto = photo
    let finalVideoUrl: string | undefined = videoPreviewUrl || undefined

    if (selectedFile) {
      setLoading(true)
      setAnalyzingStatus(
        category === 'Trash'
          ? 'Running BANTAI Model A (Grounding DINO-T Trash Detection & D-112 Spatial Grouping)…'
          : 'Running BANTAI Model B (RF-DETR Nano + Cell 8A Pothole Detection)…',
      )
      try {
        const formData = new FormData()
        formData.append('file', selectedFile)
        formData.append('category', category)
        formData.append('location', place)
        formData.append('description', description.trim())

        const res = await fetch('/api/v1/analyze', {
          method: 'POST',
          body: formData,
        })

        if (res.ok) {
          const data: UnifiedIncidentResponse = await res.json()
          dualModelResult = data
          hotspotScore = data.hotspot_score
          priorityLevel = data.priority_level
          if (data.source_media?.type === 'video' && data.source_media?.media_url) {
            finalVideoUrl = data.source_media.media_url
          }
          if (data.frames && data.frames.length > 0) {
            detections = data.frames[0].trash_findings?.detections || []
            regions = data.frames[0].trash_findings?.region_list || []
            imageWidth = data.frames[0].width
            imageHeight = data.frames[0].height
            defects = data.frames[0].road_defect_findings?.defects || []
            if (!primaryPhoto && data.frames[0].frame_url) {
              primaryPhoto = data.frames[0].frame_url
            }
          }
        }
      } catch {
        // Proceed with local attachment if backend server is offline
      } finally {
        setLoading(false)
        setAnalyzingStatus('')
      }
    }

    onSubmit({
      category,
      location: place,
      observedAt: observed.toISOString(),
      description: description.trim(),
      photo: primaryPhoto || undefined,
      videoUrl: finalVideoUrl,
      dualModelResult,
      detections,
      regions,
      imageWidth,
      imageHeight,
      defects,
      hotspotScore,
      priorityLevel,
    })
  }

  return (
    <div className="submit-layout">
      <section className="panel">
        <div className="panel-heading">
          <div>
            <div className="section-index">NEW CITIZEN OBSERVATION</div>
            <h2>Submit an incident report</h2>
            <p>Upload field evidence of street trash or road potholes in Barangay Santa Elena.</p>
          </div>
        </div>
        <form className="form-body" onSubmit={handleSubmit}>
          <div className="field-label">
            Supporting photo or video <span>*</span>
          </div>
          <div className="dropzone">
            {photo ? (
              <img className="upload-preview" src={photo} alt="Selected incident evidence" />
            ) : videoPreviewUrl && selectedFile ? (
              <div className="video-upload-preview" aria-label="Selected video file">
                <video
                  src={videoPreviewUrl}
                  controls
                  muted
                  style={{ maxWidth: '280px', maxHeight: '150px', borderRadius: '6px', marginBottom: '10px', border: '1px solid #cbd5e1' }}
                />
                <span style={{ fontSize: '13px', fontWeight: 600, color: '#1e293b', display: 'block', marginBottom: '8px' }}>
                  {selectedFile.name} (Video ready for 1.0 FPS frame extraction)
                </span>
              </div>
            ) : (
              <div className="civic-upload-placeholder" aria-label="Select photo or video file">
                <div className="drop-icon">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" />
                    <circle cx="12" cy="13" r="3" />
                  </svg>
                </div>
                <strong>Select photo or video file</strong>
                <p>Attach clear street-level evidence for automated AI verification</p>
              </div>
            )}
            <label className="button button-outline">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              Choose media file
              <input
                className="sr-only"
                type="file"
                accept="image/jpeg,image/png,image/webp,video/mp4,video/webm,video/quicktime"
                onChange={event => void chooseFile(event.target.files?.[0])}
              />
            </label>
            <small>{filename || 'Supported formats: JPG, PNG, WEBP or MP4/MOV video (1.0 FPS frame sampling)'}</small>
          </div>
          <div className="form-grid">
            <label className="form-field">
              Issue category (Selects Target AI Model)
              <select value={category} onChange={e => setCategory(e.target.value as Category)}>
                {categories.map(c => (
                  <option key={c} value={c}>
                    {c === 'Trash' ? 'Trash (Model A · Grounding DINO-T)' : 'Pothole (Model B · RF-DETR Nano Cell 8A)'}
                  </option>
                ))}
              </select>
            </label>
            <label className="form-field">
              Location
              <select value={location} onChange={e => setLocation(e.target.value)}>
                {[...locations, 'Other location'].map(l => (
                  <option key={l} value={l}>
                    {l}
                  </option>
                ))}
              </select>
            </label>
            {location === 'Other location' && (
              <label className="form-field full-field">
                Street or landmark
                <input
                  required
                  maxLength={120}
                  value={otherLocation}
                  onChange={e => setOtherLocation(e.target.value)}
                  placeholder="Enter a precise street or landmark in Barangay Santa Elena"
                />
              </label>
            )}
            <label className="form-field full-field">
              Date and time observed
              <input
                required
                type="datetime-local"
                max={localTime()}
                value={time}
                onChange={e => setTime(e.target.value)}
              />
            </label>
            <label className="form-field full-field">
              Description
              <textarea
                required
                maxLength={1000}
                rows={4}
                value={description}
                onChange={e => setDescription(e.target.value)}
                placeholder="Describe the street issue and nearby landmarks. Avoid personal information."
              />
            </label>
          </div>
          {analyzingStatus && (
            <p className="notice" role="status">
              {analyzingStatus}
            </p>
          )}
          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
          <div className="form-actions">
            <span>Category-targeted AI analysis runs automatically upon submission.</span>
            <button
              className="button button-primary"
              disabled={loading || (!photo && !selectedFile)}
            >
              {loading ? (analyzingStatus ? 'Running AI Model…' : 'Reading media…') : 'Submit report'}
            </button>
          </div>
        </form>
      </section>
      <aside className="panel guidance-panel">
        <div className="section-index">CATEGORY-SELECTIVE AI PIPELINE</div>
        <h2>Targeted single-model verification</h2>
        <div className="guide-item">
          <span>01</span>
          <p>
            <strong>Single-Pass Media Ingestion:</strong> Photos are EXIF-normalized directly; videos are sampled at 1.0 FPS for frame-by-frame inspection.
          </p>
        </div>
        <div className="guide-item">
          <span>02</span>
          <p>
            <strong>Trash Category (Model A):</strong> Runs Grounding DINO-T 3-Model Consensus with D-112 Disjoint Spatial Grouping (Individual Litter, Cluster of Litter, Garbage Pile).
          </p>
        </div>
        <div className="guide-item">
          <span>03</span>
          <p>
            <strong>Pothole Category (Model B):</strong> Runs RF-DETR Nano (v5) + Cell 8A Vision Engine with manhole rejection, DBSCAN crater fusion, and severity indexing.
          </p>
        </div>
        <p className="prototype-note">
          Routing strictly by issue category prevents cross-domain false positives. Barangay verification in the Review Queue updates municipal hotspot priorities.
        </p>
      </aside>
    </div>
  )
}
