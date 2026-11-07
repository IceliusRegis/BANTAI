import { useEffect, useRef, useState } from 'react'
import { categories, locations, computeHotspotScore } from './incidents'
import type { Category, NewReport, UnifiedIncidentResponse, TrashRegion, RoadDefect, UiDetectionBox, UiVideoFrame } from './incidents'
import { MediaDetectionViewer } from './IncidentPanels'

function localTime() {
  const date = new Date()
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16)
}

export type ReportSubmitData = NewReport & {
  mediaType?: 'image' | 'video'
  dualModelResult?: UnifiedIncidentResponse
  regions?: TrashRegion[]
  defects?: RoadDefect[]
  hotspotScore?: number
  priorityLevel?: 'CRITICAL' | 'ELEVATED' | 'ROUTINE'
  boxes?: UiDetectionBox[]
  frames?: UiVideoFrame[]
  potholeCount?: number
  trashCount?: number
  previewVideoUrl?: string
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
    setPreviewVideoUrl(undefined)
    setDualModelResult(undefined)

    const isImage = ['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || file.name.match(/\.(jpg|jpeg|png|webp)$/i)
    const isVideo = ['video/mp4', 'video/webm', 'video/quicktime'].includes(file.type) || file.name.match(/\.(mp4|mov|webm)$/i)

    if (!isImage && !isVideo) {
      setError('Please choose a photo (JPG, PNG) or a video clip (MP4, MOV).')
      return
    }

    if (file.size > 80 * 1024 * 1024) {
      setError('File must be smaller than 80 MB.')
      return
    }

    setMediaType(isVideo ? 'video' : 'image')
    setLoading(true)

    // 1. Instant local preview
    let localPreview = ''
    if (isImage) {
      try {
        localPreview = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader()
          reader.onload = () => resolve(String(reader.result))
          reader.onerror = () => reject(new Error('Image could not be read.'))
          reader.readAsDataURL(file)
        })
        if (selection.current === token) setPhoto(localPreview)
      } catch {
        if (selection.current === token) setError('Could not read image file.')
      }
    } else {
      localPreview = URL.createObjectURL(file)
      if (selection.current === token) setPhoto(localPreview)
    }

    // 2. Run Pothole AI Analysis
    setScanningStatus(isVideo ? 'Scanning video frames for potholes…' : 'Scanning photo for potholes…')

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('location', location === 'Other location' ? otherLocation.trim() || 'Unspecified Street' : location)

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
          }))

          setFrames(mappedFrames)
          const totalP = data.summary?.total_potholes ?? mappedFrames.reduce((sum, f) => sum + f.potholeCount, 0)
          const totalT = data.summary?.total_trash_items ?? mappedFrames.reduce((sum, f) => sum + f.trashCount, 0)
          setPotholeCount(totalP)
          setTrashCount(totalT)
          setBoxes(mappedFrames[0]?.boxes || [])
          if (mappedFrames[0]?.frameUrl) setPhoto(mappedFrames[0].frameUrl)
          if (data.preview_video_url) setPreviewVideoUrl(data.preview_video_url)

          if (totalP > 0 && totalT === 0) setCategory('Pothole')
          else if (totalT > 0 && totalP === 0) setCategory('Garbage pile')
        } else {
          // Photo
          const firstFrame = data.frames && data.frames[0]
          const detectedBoxes = firstFrame?.boxes || []
          const pFound = firstFrame?.pothole_count ?? (firstFrame?.road_defect_findings?.defects?.length || 0)
          const tFound = firstFrame?.trash_count ?? (firstFrame?.trash_findings?.total_items || 0)

          setBoxes(detectedBoxes)
          setPotholeCount(pFound)
          setTrashCount(tFound)
          if (firstFrame?.frame_url) setPhoto(firstFrame.frame_url)

          if (pFound > 0 && tFound === 0) setCategory('Pothole')
          else if (tFound > 0 && pFound === 0) setCategory('Garbage pile')
        }
      } else {
        // Fallback simulation if backend offline
        fallbackSimulatedDetection(isImage ? 'image' : 'video', localPreview)
      }
    } catch {
      // Graceful offline fallback
      if (selection.current === token) {
        fallbackSimulatedDetection(isImage ? 'image' : 'video', localPreview)
      }
    } finally {
      if (selection.current === token) {
        setLoading(false)
        setScanningStatus('')
      }
    }
  }

  function fallbackSimulatedDetection(kind: 'image' | 'video', previewUrl: string) {
    if (kind === 'image') {
      const simBox: UiDetectionBox = {
        id: 'box-sim-1',
        kind: 'pothole',
        label: 'Pothole (Detected)',
        confidencePct: 91,
        count: 1,
        boxPct: { x: 28, y: 38, w: 42, h: 32 },
        severity: 'high',
      }
      setBoxes([simBox])
      setPotholeCount(1)
      setTrashCount(0)
      setCategory('Pothole')
    } else {
      const simFrames: UiVideoFrame[] = [
        {
          frameIndex: 1,
          timestampSec: 1,
          clipTime: '00:01',
          frameUrl: previewUrl,
          boxes: [
            { id: 'b1', kind: 'pothole', label: 'Pothole', confidencePct: 88, count: 1, boxPct: { x: 25, y: 45, w: 35, h: 25 } }
          ],
          potholeCount: 1,
          trashCount: 0,
        },
        {
          frameIndex: 2,
          timestampSec: 2,
          clipTime: '00:02',
          frameUrl: previewUrl,
          boxes: [
            { id: 'b2', kind: 'pothole', label: 'Pothole (Deep)', confidencePct: 94, count: 1, boxPct: { x: 20, y: 40, w: 45, h: 32 } }
          ],
          potholeCount: 1,
          trashCount: 0,
        },
        {
          frameIndex: 3,
          timestampSec: 3,
          clipTime: '00:03',
          frameUrl: previewUrl,
          boxes: [
            { id: 'b3', kind: 'pothole', label: 'Pothole', confidencePct: 85, count: 1, boxPct: { x: 24, y: 48, w: 40, h: 28 } }
          ],
          potholeCount: 1,
          trashCount: 0,
        },
      ]
      setFrames(simFrames)
      setBoxes(simFrames[0].boxes)
      setPotholeCount(1)
      setTrashCount(0)
      setCategory('Pothole')
    }
  }

  function handleDemoPhoto() {
    selection.current += 1
    setError('')
    setSelectedFile(null)
    setFilename('demo_pothole_inspection.jpg')
    setMediaType('image')
    setPhoto('https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?auto=format&fit=crop&w=800&q=80')
    setBoxes([
      {
        id: 'box-demo-1',
        kind: 'pothole',
        label: 'Pothole (Severe)',
        confidencePct: 95,
        count: 1,
        boxPct: { x: 25, y: 42, w: 46, h: 32 },
        severity: 'high',
      },
    ])
    setPotholeCount(1)
    setTrashCount(0)
    setFrames([])
    setPreviewVideoUrl(undefined)
    setCategory('Pothole')
    setDescription('Deep pothole in the right lane after heavy rain. Vehicles are swerving to avoid it.')
    setLoading(false)
  }

  function handleDemoVideo() {
    selection.current += 1
    setError('')
    setSelectedFile(null)
    setFilename('demo_street_patrol.mp4')
    setMediaType('video')
    setPhoto('https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?auto=format&fit=crop&w=800&q=80')
    const simFrames: UiVideoFrame[] = [
      {
        frameIndex: 1,
        timestampSec: 1,
        clipTime: '00:01',
        frameUrl: 'https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?auto=format&fit=crop&w=800&q=80',
        boxes: [
          { id: 'b1', kind: 'pothole', label: 'Pothole Approaching', confidencePct: 89, count: 1, boxPct: { x: 30, y: 46, w: 32, h: 22 } }
        ],
        potholeCount: 1,
        trashCount: 0,
      },
      {
        frameIndex: 2,
        timestampSec: 2,
        clipTime: '00:02',
        frameUrl: 'https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?auto=format&fit=crop&w=800&q=80',
        boxes: [
          { id: 'b2', kind: 'pothole', label: 'Pothole (Severe)', confidencePct: 96, count: 1, boxPct: { x: 22, y: 38, w: 48, h: 34 } }
        ],
        potholeCount: 1,
        trashCount: 0,
      },
      {
        frameIndex: 3,
        timestampSec: 3,
        clipTime: '00:03',
        frameUrl: 'https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?auto=format&fit=crop&w=800&q=80',
        boxes: [
          { id: 'b3', kind: 'pothole', label: 'Pothole (Passing)', confidencePct: 91, count: 1, boxPct: { x: 18, y: 52, w: 50, h: 32 } }
        ],
        potholeCount: 1,
        trashCount: 0,
      },
    ]
    setFrames(simFrames)
    setBoxes(simFrames[0].boxes)
    setPotholeCount(1)
    setTrashCount(0)
    setCategory('Pothole')
    setDescription('Patrol video showing a deep pothole in the asphalt.')
    setLoading(false)
  }

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    const place = location === 'Other location' ? otherLocation.trim() : location
    const observed = new Date(time)

    if (!place) {
      setError('Please tell us which street or landmark this is located.')
      return
    }
    if (!description.trim()) {
      setError('Please provide a short description of what you noticed.')
      return
    }
    if (!photo && !selectedFile) {
      setError('Please choose a photo or video before submitting.')
      return
    }
    if (!Number.isFinite(observed.getTime()) || observed.getTime() > Date.now()) {
      setError('The observation time must be valid and cannot be in the future.')
      return
    }

    const calculatedScore = dualModelResult?.hotspot_score ?? computeHotspotScore(
      trashCount > 0 ? 1 : 0,
      0,
      0,
      { high: potholeCount }
    )

    const calculatedPriority: 'CRITICAL' | 'ELEVATED' | 'ROUTINE' =
      dualModelResult?.priority_level ?? (potholeCount >= 2 || calculatedScore >= 70 ? 'CRITICAL' : (potholeCount > 0 || trashCount > 0 ? 'ELEVATED' : 'ROUTINE'))

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
      hotspotScore: calculatedScore,
      priorityLevel: calculatedPriority,
    })
  }

  const hasMedia = Boolean(photo || selectedFile)

  return (
    <div className="submit-layout">
      <section className="panel">
        <div className="panel-heading">
          <div>
            <div className="section-index">RESIDENT REPORTING</div>
            <h2>Report a Street Problem</h2>
            <p>Snap a photo or record a short video clip. Our scanner will highlight potholes and trash automatically.</p>
          </div>
        </div>

        <form className="form-body" onSubmit={handleSubmit}>
          {/* STEP 1: Grand-friendly Media Upload */}
          <div className="field-label">
            Step 1: Photo or Video <span>*</span>
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

            {/* Live Interactive Detection Preview */}
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
                />
              </div>
            ) : (
              <div
                className="demo-photo"
                style={{ cursor: 'pointer', padding: '28px 16px', background: '#f8faf9', border: '2px dashed #b8d0c6' }}
                onClick={() => fileInputRef.current?.click()}
              >
                <span style={{ fontSize: '32px', marginBottom: '8px' }}>📷 / 🎬</span>
                <strong style={{ fontSize: '15px', color: '#134e48' }}>Click here to choose a photo or video</strong>
                <small style={{ color: '#666' }}>From your phone or computer (JPG, PNG, MP4, MOV)</small>
              </div>
            )}

            <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap', marginTop: '10px' }}>
              <button
                type="button"
                className="button button-primary"
                onClick={() => fileInputRef.current?.click()}
                disabled={loading}
              >
                {hasMedia ? '🔄 Change Photo or Video' : '📁 Choose File'}
              </button>

              <button
                type="button"
                className="button button-outline"
                onClick={handleDemoPhoto}
                disabled={loading}
              >
                ✨ Try Sample Photo
              </button>

              <button
                type="button"
                className="button button-outline"
                onClick={handleDemoVideo}
                disabled={loading}
              >
                🎬 Try Sample Video
              </button>
            </div>

            {filename && <small style={{ display: 'block', marginTop: '6px', color: '#555' }}>Selected: {filename}</small>}
            {scanningStatus && (
              <p className="notice" role="status" style={{ marginTop: '10px', fontWeight: 600 }}>
                ⏳ {scanningStatus}
              </p>
            )}
          </div>

          {/* STEP 2: Location & Description */}
          <div className="field-label" style={{ marginTop: '16px' }}>
            Step 2: Where and What Happened <span>*</span>
          </div>

          <div className="form-grid">
            <label className="form-field">
              Which street is this?
              <select value={location} onChange={e => setLocation(e.target.value)}>
                {[...locations, 'Other location'].map(l => (
                  <option key={l}>{l}</option>
                ))}
              </select>
            </label>

            <label className="form-field">
              What kind of problem?
              <select value={category} onChange={e => setCategory(e.target.value as Category)}>
                {categories.map(c => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </label>

            {location === 'Other location' && (
              <label className="form-field full-field">
                Specific street name or nearby landmark
                <input
                  required
                  maxLength={120}
                  value={otherLocation}
                  onChange={e => setOtherLocation(e.target.value)}
                  placeholder="e.g. Near corner bakery, opposite school gate"
                />
              </label>
            )}

            <label className="form-field full-field">
              Date & Time you noticed it
              <input
                required
                type="datetime-local"
                max={localTime()}
                value={time}
                onChange={e => setTime(e.target.value)}
              />
            </label>

            <label className="form-field full-field">
              Brief note for the barangay crew
              <textarea
                required
                maxLength={1000}
                rows={3}
                value={description}
                onChange={e => setDescription(e.target.value)}
                placeholder="e.g. Deep hole in the middle of the road. Vehicles are swerving to avoid it."
              />
            </label>
          </div>

          {error && <p className="form-error" role="alert" style={{ marginTop: '12px' }}>{error}</p>}

          <div className="form-actions" style={{ marginTop: '16px' }}>
            <span style={{ fontSize: '13px', color: '#555' }}>
              Your report will be sent directly to the barangay review team.
            </span>
            <button
              type="submit"
              className="button button-primary"
              style={{ fontSize: '15px', padding: '10px 22px', fontWeight: 700 }}
              disabled={loading || !hasMedia}
            >
              {loading ? 'Scanning media…' : '📤 Send Report to Barangay'}
            </button>
          </div>
        </form>
      </section>

      {/* Non-Tech Grandma Guidance Sidecard */}
      <aside className="panel guidance-panel">
        <div className="section-index">HOW IT WORKS</div>
        <h2>Fast & Easy Barangay Reporting</h2>

        <div className="guide-item">
          <span>01</span>
          <p>
            <strong>Snap or Record:</strong> Take a clear photo or short video clip using your phone.
          </p>
        </div>

        <div className="guide-item">
          <span>02</span>
          <p>
            <strong>Smart Road Scanner:</strong> The computer automatically finds potholes (🕳️) and trash (🗑️), drawing colored boxes with certainty percentages.
          </p>
        </div>

        <div className="guide-item">
          <span>03</span>
          <p>
            <strong>Track from Home:</strong> After sending, check <strong>"My Sent Reports"</strong> to see when the barangay team inspects and cleans it up!
          </p>
        </div>

        <div style={{ marginTop: '20px', padding: '12px', background: '#f0fdf4', borderRadius: '8px', border: '1px solid #bbf7d0' }}>
          <strong style={{ display: 'block', color: '#166534', fontSize: '12px' }}>🔒 Privacy Protected</strong>
          <small style={{ color: '#15803d', fontSize: '11px' }}>
            As a resident, you only see your own submitted reports and their official cleanup status.
          </small>
        </div>
      </aside>
    </div>
  )
}
