import { useEffect, useMemo, useRef, useState } from 'react'
import IncidentApp from './IncidentApp'

type Priority = 'Critical' | 'High' | 'Moderate' | 'Low'
type View = 'dashboard' | 'map' | 'records' | 'submit' | 'detail'
type Hotspot = {
  id: string
  name: string
  area: string
  score: number
  level: Priority
  flags: number
  confidence: number
  last: string
  position: { left: string; top: string }
}

const hotspots: Hotspot[] = [
  { id: 'h1', name: 'J. P. Rizal St. near Market', area: 'Barangay Santa Elena', score: 94, level: 'Critical', flags: 18, confidence: 96, last: 'May 18, 2025', position: { left: '48%', top: '39%' } },
  { id: 'h2', name: 'Shoe Avenue corner E. Dela Paz', area: 'Barangay Santa Elena', score: 86, level: 'High', flags: 14, confidence: 91, last: 'May 18, 2025', position: { left: '68%', top: '57%' } },
  { id: 'h3', name: 'P. Burgos St. riverside', area: 'Barangay Santa Elena', score: 73, level: 'High', flags: 11, confidence: 89, last: 'May 17, 2025', position: { left: '29%', top: '65%' } },
  { id: 'h4', name: 'Marikina Bridge approach', area: 'Barangay Santa Elena', score: 58, level: 'Moderate', flags: 8, confidence: 83, last: 'May 16, 2025', position: { left: '77%', top: '28%' } },
  { id: 'h5', name: 'Kapitan Moy St. alley', area: 'Barangay Santa Elena', score: 42, level: 'Moderate', flags: 5, confidence: 78, last: 'May 15, 2025', position: { left: '36%', top: '23%' } },
  { id: 'h6', name: 'Champagnat St. junction', area: 'Barangay Santa Elena', score: 21, level: 'Low', flags: 2, confidence: 72, last: 'May 13, 2025', position: { left: '22%', top: '43%' } },
]

const navItems: { label: string; view: View; icon: string }[] = [
  { label: 'Dashboard', view: 'dashboard', icon: 'grid' },
  { label: 'Hotspot Map', view: 'map', icon: 'map' },
  { label: 'Records', view: 'records', icon: 'list' },
  { label: 'Submit Photo', view: 'submit', icon: 'upload' },
]

function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const paths: Record<string, React.ReactNode> = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
    map: <><path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3z" /><path d="M9 3v15M15 6v15" /></>,
    list: <><path d="M8 6h13M8 12h13M8 18h13" /><path d="M3 6h.01M3 12h.01M3 18h.01" strokeWidth="3" /></>,
    upload: <><path d="M12 16V3m-5 5 5-5 5 5" /><path d="M4 16v4h16v-4" /></>,
    arrow: <><path d="M5 12h14m-6-6 6 6-6 6" /></>,
    chevron: <><path d="m9 18 6-6-6-6" /></>,
    back: <><path d="M19 12H5m6 6-6-6 6-6" /></>,
    filter: <><path d="M4 7h16M7 12h10m-7 5h4" /></>,
    calendar: <><rect x="3" y="5" width="18" height="16" rx="2" /><path d="M7 3v4m10-4v4M3 10h18" /></>,
    image: <><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="m21 15-5-5L5 21" /></>,
    search: <><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></>,
    download: <><path d="M12 3v12m-5-5 5 5 5-5M4 17v4h16v-4" /></>,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v5m0-8h.01" /></>,
    check: <><path d="m4 12 5 5L20 6" /></>,
    location: <><path d="M20 10c0 5-8 11-8 11S4 15 4 10a8 8 0 1 1 16 0Z" /><circle cx="12" cy="10" r="2.5" /></>,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}

function PriorityBadge({ level }: { level: Priority }) {
  return <span className={`priority-badge priority-${level.toLowerCase()}`}><span className={`priority-shape shape-${level.toLowerCase()}`} aria-hidden="true" />{level}</span>
}

function PhotoPlaceholder({ label = 'Street-level photo placeholder', className = '' }: { label?: string; className?: string }) {
  return <div className={`photo-placeholder ${className}`} role="img" aria-label={label}><Icon name="image" size={22} /><span>PHOTO</span></div>
}

function Header({ view, navigate }: { view: View; navigate: (view: View) => void }) {
  return <header className="site-header">
    <div className="header-top page-width">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true"><span className="brand-mark-top" /><span className="brand-mark-bottom" /></div>
        <div className="brand-copy"><div className="brand-name">DUMPING HOTSPOT <span>DETECTION</span></div><div className="brand-subtitle">& PRIORITIZATION AGENT</div></div>
      </div>
      <div className="header-right"><span className="pilot-pill"><span className="pilot-dot" /> PILOT BARANGAY</span><span className="header-location">Santa Elena, Marikina City</span><div className="header-avatar" aria-label="Barangay operations user">BO</div></div>
    </div>
    <nav className="nav-bar" aria-label="Main navigation"><div className="page-width nav-inner">
      {navItems.map(item => <button key={item.view} type="button" className={`nav-item ${view === item.view || (view === 'detail' && item.view === 'dashboard') ? 'active' : ''}`} aria-current={view === item.view ? 'page' : undefined} onClick={() => navigate(item.view)}><Icon name={item.icon} size={17} /><span>{item.label}</span></button>)}
      <span className="nav-context">SANITATION OPERATIONS <span className="context-divider">/</span> MAY 2025</span>
    </div></nav>
  </header>
}

function PageHeading({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: React.ReactNode }) {
  return <div className="page-heading"><div><div className="eyebrow"><span className="eyebrow-line" />{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action && <div className="heading-action">{action}</div>}</div>
}

function SummaryCard({ label, value, note, icon, emphasis = false }: { label: string; value: string; note: string; icon: string; emphasis?: boolean }) {
  return <div className={`summary-card ${emphasis ? 'summary-emphasis' : ''}`}><div className="summary-top"><span>{label}</span><span className="summary-icon"><Icon name={icon} size={18} /></span></div><div className="summary-value">{value}</div><div className="summary-note">{note}</div></div>
}

function MapCanvas({ navigate, select, selectedId, compact = false }: { navigate: (view: View, id?: string) => void; select?: (id: string) => void; selectedId?: string; compact?: boolean }) {
  const [zoom, setZoom] = useState(1)
  return <div className={`map-canvas ${compact ? 'map-compact' : ''}`} aria-label="Illustrative map of Santa Elena hotspots">
    <div className="map-world" style={{ transform: `scale(${zoom})` }}>
      <div className="river"><span>MARIKINA RIVER</span></div>
      <div className="road road-a"><span>J. P. RIZAL ST.</span></div><div className="road road-b"><span>SHOE AVENUE</span></div><div className="road road-c"><span>P. BURGOS ST.</span></div><div className="road road-d" /><div className="road road-e" /><div className="road road-f" /><div className="road road-g" /><div className="road road-h" />
      <div className="map-area-label area-one">SANTA ELENA</div><div className="map-area-label area-two">SAN ROQUE</div><div className="map-area-label area-three">CALUMPANG</div>
      <div className="map-park">MARIKINA<br />SPORTS CENTER</div>
      {hotspots.map((spot, index) => <button type="button" key={spot.id} className={`map-marker marker-${spot.level.toLowerCase()} ${selectedId === spot.id ? 'marker-selected' : ''}`} style={{ left: spot.position.left, top: spot.position.top }} aria-label={`${spot.name}, rank ${index + 1}, ${spot.level} priority, score ${spot.score}`} onClick={() => select ? select(spot.id) : navigate('detail', spot.id)} title={`${spot.name} · ${spot.level}`}><span className={`priority-shape shape-${spot.level.toLowerCase()}`} aria-hidden="true" /><span className="marker-number">{index + 1}</span></button>)}
    </div>
    {!compact && <div className="map-zoom" aria-label="Map zoom controls"><button type="button" aria-label="Zoom in" onClick={() => setZoom(z => Math.min(1.4, +(z + 0.2).toFixed(1)))}>+</button><button type="button" aria-label="Zoom out" onClick={() => setZoom(z => Math.max(1, +(z - 0.2).toFixed(1)))}>−</button></div>}
    <span className="map-watermark">SANTA ELENA · MARIKINA</span>
  </div>
}

function MapLegend() {
  return <div className="map-legend" aria-label="Priority marker legend">{(['Critical', 'High', 'Moderate', 'Low'] as Priority[]).map(level => <span key={level}><span className={`priority-shape shape-${level.toLowerCase()}`} aria-hidden="true" />{level}</span>)}</div>
}

function QueueRow({ spot, rank, navigate }: { spot: Hotspot; rank: number; navigate: (view: View, id?: string) => void }) {
  return <button type="button" className="queue-row" onClick={() => navigate('detail', spot.id)} aria-label={`View details for rank ${rank}, ${spot.name}, ${spot.level} priority, score ${spot.score}`}>
    <span className="rank-box">{String(rank).padStart(2, '0')}</span>
    <PhotoPlaceholder label={`Photo placeholder for ${spot.name}`} className="queue-photo" />
    <span className="queue-location"><strong>{spot.name}</strong><small><Icon name="location" size={13} />{spot.area}</small></span>
    <span className="queue-score"><strong>{spot.score}<small>/100</small></strong><span>PRIORITY SCORE</span></span>
    <span className="queue-priority"><PriorityBadge level={spot.level} /></span>
    <span className="queue-meta"><strong>{spot.flags} flags <span className="meta-separator">·</span> {spot.confidence}% avg.</strong><small>Last: {spot.last}</small></span>
    <span className="queue-arrow"><Icon name="chevron" size={19} /></span>
  </button>
}

function Dashboard({ navigate }: { navigate: (view: View, id?: string) => void }) {
  const [level, setLevel] = useState('All priorities')
  const [range, setRange] = useState('All time')
  const [sort, setSort] = useState('Priority score')
  const visible = useMemo(() => {
    let list = hotspots.filter(spot => level === 'All priorities' || spot.level === level)
    if (range === 'Today') list = list.filter(spot => spot.last === 'May 18, 2025')
    if (range === 'Last 7 days') list = list.filter(spot => parseInt(spot.last.split(' ')[1]) >= 12)
    return [...list].sort((a, b) => sort === 'Flag frequency' ? b.flags - a.flags : sort === 'Confidence' ? b.confidence - a.confidence : b.score - a.score)
  }, [level, range, sort])
  return <main className="page-width main-content">
    <PageHeading eyebrow="OPERATIONS OVERVIEW / 01" title="Cleanup dashboard" description="A clear view of where your team is needed most. Review, prioritize, and respond." action={<button className="button button-primary" onClick={() => navigate('submit')}><Icon name="upload" size={17} /> Submit a photo</button>} />
    <div className="summary-grid" aria-label="Hotspot summary">
      <SummaryCard label="TOTAL FLAGGED LOCATIONS" value="06" note="Across Santa Elena" icon="location" />
      <SummaryCard label="RECURRING HOTSPOTS" value="04" note="Flagged 5+ times" icon="list" />
      <SummaryCard label="NEW FLAGS TODAY" value="03" note="May 18, 2025" icon="calendar" />
      <SummaryCard label="HIGHEST PRIORITY SCORE" value="94" note="J. P. Rizal St. near Market" icon="arrow" emphasis />
    </div>
    <div className="dashboard-grid">
      <section className="panel queue-panel" aria-labelledby="queue-title"><div className="panel-heading"><div><div className="section-index">01 / ACTION QUEUE</div><h2 id="queue-title">Prioritized cleanup queue</h2><p>Ranked by recurrence and detection confidence</p></div><span className="count-pill">{visible.length} LOCATIONS</span></div>
        <div className="filters" aria-label="Queue filters"><span className="filters-label"><Icon name="filter" size={16} /> FILTERS</span><label>Priority <select value={level} onChange={e => setLevel(e.target.value)}><option>All priorities</option><option>Critical</option><option>High</option><option>Moderate</option><option>Low</option></select></label><label>Date range <select value={range} onChange={e => setRange(e.target.value)}><option>All time</option><option>Today</option><option>Last 7 days</option></select></label><label>Sort by <select value={sort} onChange={e => setSort(e.target.value)}><option>Priority score</option><option>Flag frequency</option><option>Confidence</option></select></label></div>
        <div className="queue-list">{visible.length ? visible.map(spot => <QueueRow key={spot.id} spot={spot} rank={hotspots.indexOf(spot) + 1} navigate={navigate} />) : <div className="empty-state">No hotspots match these filters.</div>}</div>
        <div className="panel-footer"><span>Showing {visible.length} of {hotspots.length} flagged locations</span><button className="text-button" onClick={() => navigate('records')}>View all records <Icon name="arrow" size={16} /></button></div>
      </section>
      <section className="panel overview-map-panel" aria-labelledby="overview-map-title"><div className="panel-heading"><div><div className="section-index">02 / GEOGRAPHIC VIEW</div><h2 id="overview-map-title">Hotspot map</h2><p>Santa Elena, Marikina City</p></div><button className="icon-button" type="button" aria-label="Open full hotspot map" onClick={() => navigate('map')}><Icon name="arrow" size={18} /></button></div><MapCanvas compact navigate={navigate} /><div className="map-panel-bottom"><MapLegend /><button className="button button-outline map-full-button" onClick={() => navigate('map')}>Explore full map <Icon name="arrow" size={16} /></button></div></section>
    </div>
    <div className="dashboard-note"><Icon name="info" size={16} /><span>Priority scores are decision-support estimates based on flag frequency and image detection confidence. Verify conditions on site before dispatch.</span></div>
  </main>
}

function MapView({ navigate }: { navigate: (view: View, id?: string) => void }) {
  const [selected, setSelected] = useState('h1')
  const spot = hotspots.find(s => s.id === selected)!
  return <main className="page-width main-content"><PageHeading eyebrow="GEOGRAPHIC VIEW / 02" title="Hotspot map" description="Explore flagged locations across the pilot barangay. Select a marker to review its priority." action={<span className="count-pill large-pill">06 ACTIVE LOCATIONS</span>} />
    <div className="map-view-layout"><section className="panel full-map-panel" aria-label="Interactive hotspot map"><div className="map-toolbar"><div><strong>Santa Elena, Marikina City</strong><span>Illustrative area map · Select a marker for details</span></div><MapLegend /></div><MapCanvas navigate={navigate} select={setSelected} selectedId={selected} /></section>
      <aside className="panel selected-panel" aria-labelledby="selected-title"><div className="section-index">SELECTED HOTSPOT</div><div className="selected-head"><span className="rank-box">{String(hotspots.indexOf(spot) + 1).padStart(2, '0')}</span><PriorityBadge level={spot.level} /></div><h2 id="selected-title">{spot.name}</h2><p className="muted"><Icon name="location" size={15} />{spot.area}</p><PhotoPlaceholder label={`Latest photo placeholder for ${spot.name}`} className="selected-photo" /><div className="selected-stats"><div><small>PRIORITY SCORE</small><strong>{spot.score}<span>/100</span></strong></div><div><small>TOTAL FLAGS</small><strong>{spot.flags}</strong></div><div><small>AVG. CONFIDENCE</small><strong>{spot.confidence}%</strong></div><div><small>LAST FLAGGED</small><strong className="date-stat">{spot.last}</strong></div></div><button className="button button-primary button-wide" onClick={() => navigate('detail', spot.id)}>View full details <Icon name="arrow" size={17} /></button></aside>
    </div>
  </main>
}

function DetailView({ spot, navigate }: { spot: Hotspot; navigate: (view: View, id?: string) => void }) {
  const rank = hotspots.indexOf(spot) + 1
  return <main className="page-width main-content detail-content"><button className="breadcrumb" onClick={() => navigate('dashboard')}><Icon name="back" size={16} /> Dashboard <span>/</span> Hotspot details</button><div className="detail-title-row"><div><div className="eyebrow"><span className="eyebrow-line" /> HOTSPOT PROFILE / {String(rank).padStart(2, '0')}</div><h1>{spot.name}</h1><p className="muted"><Icon name="location" size={16} />{spot.area} <span className="dot-separator">·</span> Last flagged {spot.last}</p></div><PriorityBadge level={spot.level} /></div>
    <div className="detail-grid"><div className="detail-main"><section className="panel detail-panel" aria-labelledby="breakdown-title"><div className="panel-heading"><div><div className="section-index">01 / PRIORITY ANALYSIS</div><h2 id="breakdown-title">Priority score breakdown</h2><p>Two signals inform this location's cleanup rank</p></div></div><div className="score-breakdown"><div className="big-score"><span>PRIORITY SCORE</span><strong>{spot.score}<small>/100</small></strong><PriorityBadge level={spot.level} /></div><div className="score-factors"><div><div className="factor-head"><span>Flag frequency</span><strong>{spot.flags} reports</strong></div><div className="progress-track"><div style={{ width: `${Math.min(100, spot.flags * 5)}%` }} /></div><small>Repeated flags indicate a recurring site</small></div><div><div className="factor-head"><span>Detection confidence</span><strong>{spot.confidence}% average</strong></div><div className="progress-track"><div style={{ width: `${spot.confidence}%` }} /></div><small>Average confidence across flagged photos</small></div></div></div></section>
      <section className="panel detail-panel" aria-labelledby="gallery-title"><div className="panel-heading"><div><div className="section-index">02 / VISUAL EVIDENCE</div><h2 id="gallery-title">Flagged photos</h2><p>Recent image references for this location</p></div><span className="count-pill">{Math.min(spot.flags, 3)} RECENT</span></div><div className="gallery-grid">{[0, 1, 2].map((n) => <div className="gallery-item" key={n}><PhotoPlaceholder label={`Flagged image ${n + 1} placeholder for ${spot.name}`} /><div className="gallery-meta"><span>May {18 - n}, 2025 · {['09:42 AM', '02:16 PM', '11:08 AM'][n]}</span><strong><span className="result-dot" />Dumping Detected · {Math.max(68, spot.confidence - n * 3)}%</strong></div></div>)}</div></section></div>
      <aside className="detail-side"><section className="panel detail-panel" aria-labelledby="timeline-title"><div className="section-index">03 / RECURRENCE</div><h2 id="timeline-title">Flags over time</h2><p className="panel-description">Reports received in the last 6 weeks</p><div className="chart" role="img" aria-label="Bar chart showing recurring flags across April and May, increasing in recent weeks"><div className="chart-grid"><span>6</span><span>4</span><span>2</span><span>0</span></div><div className="chart-bars">{[28, 47, 38, 65, 53, 85].map((height, i) => <div className="chart-column" key={i}><span className="chart-bar" style={{ height: `${height}%` }} /><small>{['APR 7', 'APR 14', 'APR 21', 'APR 28', 'MAY 5', 'MAY 12'][i]}</small></div>)}</div></div><div className="chart-note"><span className="result-dot" /> Consistent activity across recent weeks</div></section><section className="panel detail-panel quick-action"><div className="section-index">NEXT STEP</div><h2>Ready to respond?</h2><p>Use this location and its image history to plan a field visit.</p><button className="button button-outline button-wide" onClick={() => navigate('map')}>Locate on map <Icon name="arrow" size={17} /></button></section></aside>
    </div></main>
}

type RecordItem = { id: string; location: string; timestamp: string; classification: string; confidence: number; rank: string }
const records: RecordItem[] = [
  { id: 'REC-2025-0184', location: 'J. P. Rizal St. near Market', timestamp: 'May 18, 2025 · 09:42', classification: 'Dumping Detected', confidence: 98, rank: '#01' },
  { id: 'REC-2025-0183', location: 'Shoe Avenue corner E. Dela Paz', timestamp: 'May 18, 2025 · 08:14', classification: 'Dumping Detected', confidence: 93, rank: '#02' },
  { id: 'REC-2025-0182', location: 'Marikina Bridge approach', timestamp: 'May 18, 2025 · 07:35', classification: 'Clean', confidence: 87, rank: '—' },
  { id: 'REC-2025-0181', location: 'P. Burgos St. riverside', timestamp: 'May 17, 2025 · 14:16', classification: 'Dumping Detected', confidence: 91, rank: '#03' },
  { id: 'REC-2025-0180', location: 'Kapitan Moy St. alley', timestamp: 'May 16, 2025 · 11:08', classification: 'Dumping Detected', confidence: 81, rank: '#05' },
  { id: 'REC-2025-0179', location: 'Champagnat St. junction', timestamp: 'May 15, 2025 · 16:22', classification: 'Clean', confidence: 92, rank: '—' },
  { id: 'REC-2025-0178', location: 'Marikina Bridge approach', timestamp: 'May 14, 2025 · 10:05', classification: 'Dumping Detected', confidence: 83, rank: '#04' },
  { id: 'REC-2025-0177', location: 'J. P. Rizal St. near Market', timestamp: 'May 13, 2025 · 09:30', classification: 'Dumping Detected', confidence: 95, rank: '#01' },
]

function RecordsView({ navigate }: { navigate: (view: View, id?: string) => void }) {
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<'newest' | 'oldest' | 'confidence'>('newest')
  const [page, setPage] = useState(1)
  const filtered = useMemo(() => records.filter(record => `${record.id} ${record.location} ${record.classification}`.toLowerCase().includes(query.toLowerCase())).sort((a, b) => sort === 'confidence' ? b.confidence - a.confidence : sort === 'oldest' ? records.indexOf(b) - records.indexOf(a) : records.indexOf(a) - records.indexOf(b)), [query, sort])
  const perPage = 5
  const pages = Math.max(1, Math.ceil(filtered.length / perPage))
  function exportCsv() {
    const header = ['Record ID', 'Location', 'Timestamp', 'Image Reference', 'Classification', 'Confidence %', 'Computed Priority Rank']
    const csv = [header, ...filtered.map(r => [r.id, r.location, r.timestamp, `${r.id}.jpg`, r.classification, String(r.confidence), r.rank])].map(row => row.map(value => `"${value.replace(/"/g, '""')}"`).join(',')).join('\n')
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
    const link = document.createElement('a'); link.href = url; link.download = 'santa-elena-incident-records.csv'; link.click(); URL.revokeObjectURL(url)
  }
  return <main className="page-width main-content"><PageHeading eyebrow="INCIDENT HISTORY / 03" title="Incident records" description="Browse image classifications and priority outcomes from recent reports." action={<button className="button button-outline" onClick={exportCsv}><Icon name="download" size={17} /> Export CSV</button>} />
    <section className="panel records-panel" aria-labelledby="records-title"><div className="records-toolbar"><div><div className="section-index">REPORT LOG</div><h2 id="records-title">All records <span className="count-pill">{filtered.length} RECORDS</span></h2></div><div className="records-controls"><label className="search-field"><Icon name="search" size={18} /><span className="sr-only">Search records</span><input value={query} onChange={e => { setQuery(e.target.value); setPage(1) }} placeholder="Search records or locations" /></label><label className="sort-field">Sort <select value={sort} onChange={e => { setSort(e.target.value as typeof sort); setPage(1) }}><option value="newest">Newest first</option><option value="oldest">Oldest first</option><option value="confidence">Confidence</option></select></label></div></div>
      <div className="table-scroll"><table><thead><tr><th scope="col">RECORD ID</th><th scope="col">LOCATION</th><th scope="col">TIMESTAMP</th><th scope="col">IMAGE REFERENCE</th><th scope="col">CLASSIFICATION</th><th scope="col">CONFIDENCE</th><th scope="col">PRIORITY RANK</th></tr></thead><tbody>{filtered.slice((page - 1) * perPage, page * perPage).map(record => <tr key={record.id}><td><span className="record-id">{record.id}</span></td><td><button className="table-location" onClick={() => { const spot = hotspots.find(s => s.name === record.location); if (spot) navigate('detail', spot.id) }}>{record.location}</button></td><td>{record.timestamp}</td><td><span className="image-ref"><Icon name="image" size={15} />{record.id}.jpg</span></td><td><span className={`classification ${record.classification === 'Clean' ? 'clean' : ''}`}><span className="result-dot" />{record.classification}</span></td><td><strong>{record.confidence}%</strong></td><td><span className="table-rank">{record.rank}</span></td></tr>)}</tbody></table>{filtered.length === 0 && <div className="empty-state">No records match your search.</div>}</div><div className="table-footer"><span>Showing {filtered.length ? (page - 1) * perPage + 1 : 0}–{Math.min(page * perPage, filtered.length)} of {filtered.length} records</span><div className="pagination"><button aria-label="Previous page" disabled={page === 1} onClick={() => setPage(p => p - 1)}>‹</button><span>Page {page} of {pages}</span><button aria-label="Next page" disabled={page === pages} onClick={() => setPage(p => p + 1)}>›</button></div></div></section>
    <div className="dashboard-note"><Icon name="info" size={16} /><span>Records shown are sample data for this wireframe. Classifications and priority ranks are simulated.</span></div>
  </main>
}

function SubmitView({ navigate }: { navigate: (view: View, id?: string) => void }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [location, setLocation] = useState('')
  const [time, setTime] = useState(() => { const date = new Date(); return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}T${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}` })
  const [result, setResult] = useState(false)
  const [dragging, setDragging] = useState(false)
  useEffect(() => { if (!file) { setPreview(''); return } const url = URL.createObjectURL(file); setPreview(url); return () => URL.revokeObjectURL(url) }, [file])
  function reset() { setFile(null); setLocation(''); setResult(false); if (inputRef.current) inputRef.current.value = '' }
  return <main className="page-width main-content"><PageHeading eyebrow="NEW REPORT / 04" title="Submit a photo" description="Add a street-level photo or simulated CCTV capture to help identify a potential dumping site." />
    <div className="submit-layout"><section className="panel submit-panel" aria-labelledby="submit-title"><div className="panel-heading"><div><div className="section-index">01 / REPORT DETAILS</div><h2 id="submit-title">New image report</h2><p>Complete the fields below to simulate a submission</p></div></div>
      {!result ? <form onSubmit={e => { e.preventDefault(); if (file && location && time) setResult(true) }}><div className="form-body"><label className="field-label" htmlFor="photo-input">Street-level photo <span>*</span></label><div className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); const incoming = e.dataTransfer.files[0]; if (incoming?.type.startsWith('image/')) setFile(incoming) }}><input ref={inputRef} id="photo-input" type="file" accept="image/*" className="sr-only" onChange={e => setFile(e.target.files?.[0] || null)} />{preview ? <img className="upload-preview" src={preview} alt="Preview of selected report image" /> : <span className="drop-icon"><Icon name="upload" size={25} /></span>}<strong>{file ? file.name : 'Drop a photo here'}</strong><p>{file ? 'Image selected. Choose another file to replace it.' : 'or choose a file from your device'}</p><button type="button" className="button button-outline" onClick={() => inputRef.current?.click()}>Take photo / Choose file</button><small>JPG, PNG or WEBP · Image files only</small></div><div className="form-grid"><label className="form-field">Location / street name <span>*</span><input required value={location} onChange={e => setLocation(e.target.value)} placeholder="e.g. J. P. Rizal St. near Market" /></label><label className="form-field">Date and time <span>*</span><input required type="datetime-local" value={time} onChange={e => setTime(e.target.value)} /></label></div><div className="form-actions"><span>* Required fields</span><button type="submit" className="button button-primary" disabled={!file || !location || !time}>Analyze report <Icon name="arrow" size={17} /></button></div></div></form> : <div className="result-state"><div className="result-banner"><span className="result-check"><Icon name="check" size={20} /></span><div><strong>Report analyzed</strong><p>Simulated classification is ready for review.</p></div></div><div className="result-detail"><div>{preview && <img className="result-image" src={preview} alt="Submitted report preview" />}</div><div className="result-details-list"><span className="section-index">CLASSIFICATION RESULT</span><h3><span className="result-dot" /> Dumping Detected</h3><div><span>Confidence score</span><strong>94%</strong></div><div><span>Location</span><strong>{location}</strong></div><div><span>Priority score change</span><strong>68 → 74 <span className="increase">+6 points</span></strong></div><p>This is a simulated result for wireframe demonstration only.</p></div></div><div className="result-actions"><button className="button button-outline" onClick={reset}>Submit another photo</button><button className="button button-primary" onClick={() => navigate('dashboard')}>Back to dashboard <Icon name="arrow" size={16} /></button></div></div>}
    </section><aside className="submit-side"><div className="panel guidance-panel"><div className="section-index">FIELD GUIDE</div><h2>Before you submit</h2><div className="guide-item"><span>01</span><p>Capture the site clearly, with the dumped materials visible in frame.</p></div><div className="guide-item"><span>02</span><p>Enter the closest street name or landmark so crews can locate it.</p></div><div className="guide-item"><span>03</span><p>Review the result and verify conditions before dispatching a crew.</p></div></div><div className="side-note"><Icon name="info" size={18} /><p>This prototype simulates image analysis. No photo is uploaded to a server or saved.</p></div></aside></div>
  </main>
}

function Footer() { return <footer className="site-footer"><div className="page-width footer-inner"><span>DUMPING HOTSPOT DETECTION & PRIORITIZATION AGENT</span><span>PILOT WIREFRAME · SANTA ELENA, MARIKINA CITY</span></div></footer> }

function parseHash(): { view: View; id: string } {
  const [part, id] = window.location.hash.replace(/^#\/?/, '').split('/')
  return { view: (['dashboard', 'map', 'records', 'submit', 'detail'].includes(part) ? part : 'dashboard') as View, id: id || 'h1' }
}

function LegacyApp() {
  const [route, setRoute] = useState(parseHash)
  useEffect(() => { const onHash = () => setRoute(parseHash()); window.addEventListener('hashchange', onHash); return () => window.removeEventListener('hashchange', onHash) }, [])
  function navigate(view: View, id?: string) { const hash = `#/${view}${id ? `/${id}` : ''}`; if (window.location.hash === hash) { setRoute({ view, id: id || 'h1' }); window.scrollTo(0, 0) } else { window.location.hash = hash; window.scrollTo(0, 0) } }
  const spot = hotspots.find(s => s.id === route.id) || hotspots[0]
  return <div className="app-shell"><Header view={route.view} navigate={navigate} />{route.view === 'dashboard' && <Dashboard navigate={navigate} />}{route.view === 'map' && <MapView navigate={navigate} />}{route.view === 'records' && <RecordsView navigate={navigate} />}{route.view === 'submit' && <SubmitView navigate={navigate} />}{route.view === 'detail' && <DetailView spot={spot} navigate={navigate} />}<Footer /></div>
}

export default IncidentApp
