import { useEffect, useRef, useState } from 'react'
import { categories, locations } from './incidents'
import type { Category, NewReport } from './incidents'

function localTime() {
  const date = new Date()
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16)
}

export default function ReportForm({ onSubmit }: { onSubmit: (report: NewReport) => void }) {
  const [category, setCategory] = useState<Category>('Garbage pile')
  const [location, setLocation] = useState(locations[0])
  const [otherLocation, setOtherLocation] = useState('')
  const [description, setDescription] = useState('')
  const [time, setTime] = useState(localTime)
  const [photo, setPhoto] = useState('')
  const [filename, setFilename] = useState('')
  const [demoPhoto, setDemoPhoto] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const selection = useRef(0)
  useEffect(() => () => { selection.current += 1 }, [])

  async function choosePhoto(file?: File) {
    const token = ++selection.current
    setError(''); setPhoto(''); setFilename(''); setDemoPhoto(false)
    if (!file) { setLoading(false); return }
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || file.size > 8 * 1024 * 1024) {
      setError('Choose a JPG, PNG, or WEBP image smaller than 8 MB.'); setLoading(false); return
    }
    setLoading(true)
    try {
      const bitmap = await createImageBitmap(file)
      bitmap.close()
      const data = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve(String(reader.result))
        reader.onerror = () => reject(new Error('Image could not be read.'))
        reader.readAsDataURL(file)
      })
      if (selection.current === token) { setPhoto(data); setFilename(file.name) }
    } catch {
      if (selection.current === token) setError('This image could not be opened. Choose another file.')
    } finally { if (selection.current === token) setLoading(false) }
  }

  return <div className="submit-layout">
    <section className="panel"><div className="panel-heading"><div><div className="section-index">NEW OBSERVATION</div><h2>Submit an incident report</h2><p>Tell the barangay what you noticed and where.</p></div></div>
      <form className="form-body" onSubmit={event => {
        event.preventDefault()
        const place = location === 'Other location' ? otherLocation.trim() : location
        const observed = new Date(time)
        if (!place || !description.trim() || (!photo && !demoPhoto)) { setError('Add a photo, location, and description before submitting.'); return }
        if (!Number.isFinite(observed.getTime()) || observed.getTime() > Date.now()) { setError('The observation time must be valid and cannot be in the future.'); return }
        onSubmit({ category, location: place, observedAt: observed.toISOString(), description: description.trim(), photo: photo || undefined })
      }}>
        <div className="field-label">Supporting photo <span>*</span></div>
        <div className="dropzone">{photo ? <img className="upload-preview" src={photo} alt="Selected incident evidence" /> : <div className="demo-photo" aria-label="Illustrative photo placeholder">▧<span>{demoPhoto ? `${category} · demo image placeholder` : 'Add a clear photo of the issue'}</span></div>}
          <label className="button button-outline">Choose photo<input className="sr-only" type="file" accept="image/jpeg,image/png,image/webp" onChange={event => void choosePhoto(event.target.files?.[0])} /></label>
          <button type="button" className="text-button" onClick={() => { selection.current += 1; setPhoto(''); setFilename(''); setDemoPhoto(true); setLoading(false); setError('') }}>Use demo photo placeholder</button>
          <small>{filename || 'JPG, PNG, WEBP · Up to 8 MB · Kept in this browser session only'}</small>
        </div>
        <div className="form-grid"><label className="form-field">Issue category<select value={category} onChange={e => setCategory(e.target.value as Category)}>{categories.map(c => <option key={c}>{c}</option>)}</select></label>
          <label className="form-field">Location<select value={location} onChange={e => setLocation(e.target.value)}>{[...locations, 'Other location'].map(l => <option key={l}>{l}</option>)}</select></label>
          {location === 'Other location' && <label className="form-field full-field">Street or landmark<input required maxLength={120} value={otherLocation} onChange={e => setOtherLocation(e.target.value)} placeholder="Enter a precise street or landmark" /></label>}
          <label className="form-field full-field">Date and time observed<input required type="datetime-local" max={localTime()} value={time} onChange={e => setTime(e.target.value)} /></label>
          <label className="form-field full-field">Description<textarea required maxLength={1000} rows={4} value={description} onChange={e => setDescription(e.target.value)} placeholder="Describe the issue and nearby landmarks. Avoid personal information." /></label>
        </div>
        {error && <p className="form-error" role="alert">{error}</p>}
        <div className="form-actions"><span>Admin validation is required before a report affects rankings.</span><button className="button button-primary" disabled={loading || (!photo && !demoPhoto)}>{loading ? 'Reading image…' : 'Submit report'}</button></div>
      </form>
    </section>
    <aside className="panel guidance-panel"><div className="section-index">WHAT HAPPENS NEXT</div><h2>One report. A clear next step.</h2>
      <div className="guide-item"><span>01</span><p>Your report enters the admin review queue with a simulated detection result.</p></div>
      <div className="guide-item"><span>02</span><p>An admin validates the observation and checks whether it belongs to an existing incident.</p></div>
      <div className="guide-item"><span>03</span><p>Track inspection, cleanup, or maintenance updates in My reports.</p></div>
      <p className="prototype-note">Wireframe only: no computer vision model runs. The selected category supplies the simulated detection label.</p>
    </aside>
  </div>
}
