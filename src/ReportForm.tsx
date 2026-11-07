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

export default function ReportForm({ onSubmit }: { onSubmit: (report: ReportSubmitData) => void }) {
  const [category, setCategory] = useState<Category>('Trash')
  const [location, setLocation] = useState(locations[0])
  const [otherLocation, setOtherLocation] = useState('')
  const [description, setDescription] = useState('')
  const [time, setTime] = useState(localTime)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [photo, setPhoto] = useState('')
  const [filename, setFilename] = useState('')
  const [demoPhoto, setDemoPhoto] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [analyzingStatus, setAnalyzingStatus] = useState('')
  const selection = useRef(0)

  useEffect(() => () => { selection.current += 1 }, [])

  async function chooseFile(file?: File) {
    const token = ++selection.current
    setError(''); setPhoto(''); setFilename(''); setDemoPhoto(false); setSelectedFile(null)
    if (!file) { setLoading(false); return }

    const isImage = ['image/jpeg', 'image/png', 'image/webp'].includes(file.type)
    const isVideo = ['video/mp4', 'video/webm', 'video/quicktime'].includes(file.type) || file.name.match(/\.(mp4|mov|webm)$/i)

    if (!isImage && !isVideo) {
      setError('Choose a JPG, PNG, WEBP image or an MP4/MOV video.')
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
        if (selection.current === token) { setPhoto(data) }
      } else {
        // Video placeholder thumbnail
        if (selection.current === token) { setPhoto('') }
      }
    } catch {
      if (selection.current === token) setError('File could not be opened. Choose another file.')
    } finally {
      if (selection.current === token) setLoading(false)
    }
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()
    const place = location === 'Other location' ? otherLocation.trim() : location
    const observed = new Date(time)
    if (!place || !description.trim() || (!photo && !demoPhoto && !selectedFile)) {
      setError('Add media, location, and description before submitting.')
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

    // Attempt live API dual-model analysis if a real file was uploaded
    if (selectedFile) {
      setLoading(true)
      setAnalyzingStatus(category === 'Trash' ? 'Running BANTAI Model A (Grounding DINO-T Trash Detection & D-112 Spatial Grouping)…' : 'Running BANTAI Model B (RF-DETR Nano + Cell 8A Pothole Detection)…')
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
        // Fallback gracefully to offline state
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

  return <div className="submit-layout">
    <section className="panel"><div className="panel-heading"><div><div className="section-index">NEW OBSERVATION</div><h2>Submit an incident report</h2><p>Tell the barangay what you noticed and where.</p></div></div>
      <form className="form-body" onSubmit={handleSubmit}>
        <div className="field-label">Supporting photo or video <span>*</span></div>
        <div className="dropzone">
          {photo ? (
            <img className="upload-preview" src={photo} alt="Selected incident evidence" />
          ) : selectedFile ? (
            <div className="demo-photo" aria-label="Selected media file">
              🎬<span>{selectedFile.name} (Video ready for single-pass analysis)</span>
            </div>
          ) : (
            <div className="demo-photo" aria-label="Illustrative photo placeholder">
              ▧<span>{demoPhoto ? `${category} · demo image placeholder` : 'Add a clear photo or video clip'}</span>
            </div>
          )}
          <label className="button button-outline">
            Choose media file
            <input className="sr-only" type="file" accept="image/jpeg,image/png,image/webp,video/mp4,video/webm,video/quicktime" onChange={event => void chooseFile(event.target.files?.[0])} />
          </label>
          <button type="button" className="text-button" onClick={() => { selection.current += 1; setPhoto(''); setFilename(''); setSelectedFile(null); setDemoPhoto(true); setLoading(false); setError('') }}>Use demo placeholder</button>
          <small>{filename || 'JPG, PNG, WEBP or MP4 video · Single-pass 1.0 FPS ingestion'}</small>
        </div>
        <div className="form-grid"><label className="form-field">Issue category<select value={category} onChange={e => setCategory(e.target.value as Category)}>{categories.map(c => <option key={c}>{c}</option>)}</select></label>
          <label className="form-field">Location<select value={location} onChange={e => setLocation(e.target.value)}>{[...locations, 'Other location'].map(l => <option key={l}>{l}</option>)}</select></label>
          {location === 'Other location' && <label className="form-field full-field">Street or landmark<input required maxLength={120} value={otherLocation} onChange={e => setOtherLocation(e.target.value)} placeholder="Enter a precise street or landmark" /></label>}
          <label className="form-field full-field">Date and time observed<input required type="datetime-local" max={localTime()} value={time} onChange={e => setTime(e.target.value)} /></label>
          <label className="form-field full-field">Description<textarea required maxLength={1000} rows={4} value={description} onChange={e => setDescription(e.target.value)} placeholder="Describe the issue and nearby landmarks. Avoid personal information." /></label>
        </div>
        {analyzingStatus && <p className="notice" role="status">{analyzingStatus}</p>}
        {error && <p className="form-error" role="alert">{error}</p>}
        <div className="form-actions"><span>Category-targeted AI analysis runs automatically upon submission.</span><button className="button button-primary" disabled={loading || (!photo && !demoPhoto && !selectedFile)}>{loading ? (analyzingStatus ? 'Analyzing AI Models…' : 'Reading media…') : 'Submit report'}</button></div>
      </form>
    </section>
    <aside className="panel guidance-panel"><div className="section-index">AI DUAL-MODEL PIPELINE</div><h2>Category-targeted AI analysis.</h2>
      <div className="guide-item"><span>01</span><p><strong>Single-Pass Ingestion:</strong> Photos are staged directly; videos are sampled at 1.0 FPS once.</p></div>
      <div className="guide-item"><span>02</span><p><strong>Model A (Trash Detector):</strong> 3-Model Consensus with D-112 Disjoint Spatial Grouping.</p></div>
      <div className="guide-item"><span>03</span><p><strong>Model B (Road Defects):</strong> RF-DETR Nano (v5) + Cell 8A Vision Engine with anti-manhole filtering, DBSCAN crater fusion &amp; severity indexing.</p></div>
      <p className="prototype-note">Admin validation in the Review Queue confirms detections and updates barangay hotspot priorities.</p>
    </aside>
  </div>
}
