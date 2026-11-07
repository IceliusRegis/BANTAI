export const categories = ['Pothole', 'Garbage pile', 'Road surface damage'] as const
export type Category = typeof categories[number]

export const locations = [
  'J. P. Rizal St. near Market',
  'Shoe Avenue corner E. Dela Paz',
  'P. Burgos St. riverside',
  'Marikina Bridge approach',
  'Kapitan Moy St. alley',
  'Champagnat St. junction',
] as const

export const incidentStatuses = ['Open', 'For inspection', 'In progress', 'Resolved'] as const
export type IncidentStatus = typeof incidentStatuses[number]

export type UiBoxPct = {
  x: number
  y: number
  w: number
  h: number
}

export type UiDetectionBox = {
  id: string
  kind: 'pothole' | 'trash'
  label: string
  category?: string
  severity?: 'low' | 'moderate' | 'high' | 'critical'
  confidencePct: number
  count: number
  boxPct: UiBoxPct
  estimatedDiameterCm?: number
  categoryBreakdown?: Record<string, number>
}

export type UiVideoFrame = {
  frameIndex: number
  timestampSec: number
  clipTime: string
  frameUrl: string
  boxes: UiDetectionBox[]
  potholeCount: number
  trashCount: number
  detections?: TrashDetection[]
  regions?: TrashRegion[]
  defects?: RoadDefect[]
  width?: number
  height?: number
  frameHotspotScore?: number
  framePriorityLevel?: 'CRITICAL' | 'ELEVATED' | 'ROUTINE'
}

export type TrashDetection = {
  detection_id: number
  category_id: number
  label: string
  bantai_category: string
  model_score: number
  bbox: [number, number, number, number]
  box_pct?: UiBoxPct
  confidence_pct?: number
}

export type TrashRegion = {
  region_id: number
  category: 'individual_litter' | 'group_of_individual_litter' | 'garbage_pile'
  display_label: string
  item_count: number
  bbox: [number, number, number, number]
  box_pct?: UiBoxPct
  model_score: number | null
  confidence_pct?: number
  member_indices: number[]
  category_breakdown?: Record<string, number>
}

export type TrashFindings = {
  model_version: string
  total_items: number
  regions: {
    garbage_piles: number
    groups_of_litter: number
    individual_items: number
  }
  items_by_bantai_category: Record<string, number>
  region_list: TrashRegion[]
  detections: TrashDetection[]
}

export type RoadDefect = {
  defect_id: number
  type: 'pothole'
  label?: string
  severity: 'low' | 'moderate' | 'high' | 'critical'
  estimated_diameter_cm: number
  bbox: [number, number, number, number]
  box_pct?: UiBoxPct
  confidence: number
  confidence_pct?: number
}

export type RoadDefectFindings = {
  model_version: string
  total_defects: number
  defects: RoadDefect[]
}

export type DualModelFrameResult = {
  frame_index: number
  timestamp_sec: number
  frame_url: string
  width?: number
  height?: number
  trash_findings: TrashFindings
  road_defect_findings: RoadDefectFindings
  boxes?: UiDetectionBox[]
  pothole_count?: number
  trash_count?: number
  frame_hotspot_score?: number
  frame_priority_level?: 'CRITICAL' | 'ELEVATED' | 'ROUTINE'
}

export type UnifiedIncidentResponse = {
  incident_id: string
  source_media: {
    filename: string
    type: 'image' | 'video'
    sha256?: string
    width?: number
    height?: number
    fps?: number
    duration_sec?: number
  }
  preview_video_url?: string | null
  location: {
    street_name: string
    latitude: number
    longitude: number
  }
  hotspot_score: number
  priority_level: 'CRITICAL' | 'ELEVATED' | 'ROUTINE'
  dispatch_recommendation: string
  summary?: {
    total_frames: number
    total_extracted_frames?: number
    total_potholes: number
    total_trash_items: number
    total_garbage_piles: number
    total_litter_groups: number
    total_individual_litter: number
    potholes_by_severity?: Record<string, number>
  }
  frames: DualModelFrameResult[]
}

export type Observation = {
  id: string
  reporter: string
  source: 'Resident photo' | 'Patrol dashcam' | 'CCTV feed'
  location: string
  category: Category
  observedAt: string
  submittedAt: string
  description: string
  photo?: string
  clipTime?: string
  confidence: number
  review: 'Pending review' | 'Validated' | 'Rejected'
  incidentId?: string
  reviewNote: string
  mediaType?: 'image' | 'video'
  boxes?: UiDetectionBox[]
  frames?: UiVideoFrame[]
  potholeCount?: number
  trashCount?: number
  previewVideoUrl?: string
  dualModelResult?: UnifiedIncidentResponse
  regions?: TrashRegion[]
  detections?: TrashDetection[]
  defects?: RoadDefect[]
  hotspotScore?: number
  priorityLevel?: 'CRITICAL' | 'ELEVATED' | 'ROUTINE'
  imageWidth?: number
  imageHeight?: number
}

export function computeHotspotScore(
  piles: number,
  groups: number,
  litter: number,
  potholesBySeverity: { low?: number; moderate?: number; high?: number } = {}
): number {
  const wPile = 35.0
  const wGroup = 15.0
  const wLitter = 3.0
  const wLow = 25.0
  const wMod = 32.5
  const wHigh = 40.0

  const score =
    (wPile * Math.max(0, piles)) +
    (wGroup * Math.max(0, groups)) +
    (wLitter * Math.max(0, litter)) +
    (wLow * (potholesBySeverity.low || 0)) +
    (wMod * (potholesBySeverity.moderate || 0)) +
    (wHigh * (potholesBySeverity.high || 0))

  return Math.min(100, Math.round(score * 10) / 10)
}

export type Incident = {
  id: string
  category: Category
  location: string
  status: IncidentStatus
  note: string
  updatedAt: string
}

export type Store = {
  observations: Observation[]
  incidents: Incident[]
}

export type RankedIncident = Incident & {
  frequency: number
  latest: string
}

export type NewReport = Pick<
  Observation,
  'location' | 'category' | 'observedAt' | 'description' | 'photo'
> & {
  mediaType?: 'image' | 'video'
  boxes?: UiDetectionBox[]
  frames?: UiVideoFrame[]
  potholeCount?: number
  trashCount?: number
  previewVideoUrl?: string
  dualModelResult?: UnifiedIncidentResponse
  regions?: TrashRegion[]
  detections?: TrashDetection[]
  defects?: RoadDefect[]
  hotspotScore?: number
  priorityLevel?: 'CRITICAL' | 'ELEVATED' | 'ROUTINE'
  imageWidth?: number
  imageHeight?: number
}

export const residentId = 'Maria Santos (Verified Resident)'

export function normalizeLocation(value: string) {
  return value.trim().replace(/\s+/g, ' ').toLowerCase()
}

function hoursAgo(h: number, now = new Date()): string {
  return new Date(now.getTime() - h * 3600000).toISOString()
}

export function seedStore(now = new Date()): Store {
  const ago = (hours: number) => new Date(now.getTime() - hours * 3600000).toISOString()
  const incidents: Incident[] = [
    { id: 'INC-001', category: 'Garbage pile', location: locations[0], status: 'Open', note: 'Awaiting barangay cleanup assessment.', updatedAt: ago(3) },
    { id: 'INC-002', category: 'Pothole', location: locations[1], status: 'For inspection', note: 'Maintenance team will inspect the reported road defect.', updatedAt: ago(5) },
    { id: 'INC-003', category: 'Road surface damage', location: locations[3], status: 'In progress', note: 'Inspection completed. Maintenance coordination is in progress.', updatedAt: ago(8) },
    { id: 'INC-004', category: 'Garbage pile', location: locations[2], status: 'Resolved', note: 'Cleanup completed and verified by the barangay team.', updatedAt: ago(12) },
  ]
  const observations: Observation[] = []
  for (const [index, incident] of incidents.entries()) {
    const count = [4, 3, 3, 2][index]
    for (let n = 0; n < count; n++) {
      const isPothole = incident.category === 'Pothole'
      const sampleBoxes: UiDetectionBox[] = isPothole ? [
        {
          id: `box-pot-${n}`,
          kind: 'pothole',
          label: 'Pothole',
          severity: 'moderate',
          confidencePct: 90 - n,
          count: 1,
          boxPct: { x: 38 + n * 4, y: 55 + n * 3, w: 18, h: 12 },
          estimatedDiameterCm: 36.5,
        }
      ] : [
        {
          id: `box-tr-${n}`,
          kind: 'trash',
          label: 'Garbage Pile',
          category: 'garbage_pile',
          confidencePct: 92 - n,
          count: 4,
          boxPct: { x: 32 + n * 3, y: 52 + n * 4, w: 24, h: 18 },
        }
      ]

      observations.push({
        id: `OBS-${String(observations.length + 1).padStart(3, '0')}`,
        reporter: n === 0 ? residentId : 'Sample resident', source: 'Resident photo',
        location: incident.location, category: incident.category, observedAt: ago(3 + index * 3 + n * 20), submittedAt: ago(2 + index * 3 + n * 20),
        description: `Sample observation of ${incident.category.toLowerCase()} near this location.`,
        confidence: 90 - n, review: 'Validated', incidentId: incident.id, reviewNote: 'Validated by barangay admin.',
        mediaType: 'image',
        boxes: sampleBoxes,
        potholeCount: isPothole ? 1 : 0,
        trashCount: isPothole ? 0 : 4,
        hotspotScore: isPothole ? 62.5 : 85.0,
        priorityLevel: isPothole ? 'ELEVATED' : 'CRITICAL',
      })
    }
  }

  // Pending user observation
  observations.unshift({
    id: 'OBS-013',
    reporter: 'Sample resident',
    source: 'Resident photo',
    location: locations[0],
    category: 'Garbage pile',
    observedAt: ago(1),
    submittedAt: ago(0.5),
    description: 'Piled garbage and plastic litter beside the market entrance. Please check whether this is the existing incident.',
    confidence: 93,
    review: 'Pending review',
    reviewNote: '',
    mediaType: 'image',
    boxes: [
      {
        id: 'box-p13',
        kind: 'trash',
        label: 'Garbage Pile',
        category: 'garbage_pile',
        confidencePct: 93,
        count: 5,
        boxPct: { x: 34, y: 50, w: 28, h: 22 },
        categoryBreakdown: { Plastic: 3, Paper: 2 },
      }
    ],
    potholeCount: 0,
    trashCount: 5,
    hotspotScore: 85.0,
    priorityLevel: 'CRITICAL',
  })

  return { incidents, observations }
}

// Only validated observations contribute. Resolved incidents stay in the history.
export function rankIncidents(store: Store): RankedIncident[] {
  return store.incidents.filter(i => i.status !== 'Resolved').map(incident => {
    const evidence = store.observations.filter(o => o.review === 'Validated' && o.incidentId === incident.id)
    return { ...incident, frequency: evidence.length, latest: evidence.map(o => o.observedAt).sort().slice(-1)[0] || incident.updatedAt }
  }).filter(i => i.frequency > 0).sort((a, b) => b.frequency - a.frequency || b.latest.localeCompare(a.latest) || a.id.localeCompare(b.id))
}

export type HotspotRanking = {
  location: string
  frequency: number
  latest: string
  incidents: number
  potholesCount: number
  trashCount: number
  priorityLevel: 'Urgent' | 'High' | 'Moderate'
}

export function rankHotspots(store: Store): HotspotRanking[] {
  const groups = new Map<string, HotspotRanking>()
  for (const incident of rankIncidents(store)) {
    const key = normalizeLocation(incident.location)
    const previous = groups.get(key) || {
      location: incident.location,
      frequency: 0,
      latest: '',
      incidents: 0,
      potholesCount: 0,
      trashCount: 0,
      priorityLevel: 'Moderate',
    }

    const validated = store.observations.filter(o => o.review === 'Validated' && normalizeLocation(o.location) === key)
    const pCount = validated.reduce((sum, o) => sum + (o.potholeCount || (o.category === 'Pothole' ? 1 : 0)), 0)
    const tCount = validated.reduce((sum, o) => sum + (o.trashCount || (o.category === 'Garbage pile' ? 3 : 0)), 0)

    const priorityLevel = (pCount >= 2 || tCount >= 5 || validated.length >= 4)
      ? 'Urgent'
      : (validated.length >= 2 ? 'High' : 'Moderate')

    groups.set(key, {
      location: previous.location,
      frequency: previous.frequency + incident.frequency,
      latest: previous.latest > incident.latest ? previous.latest : incident.latest,
      incidents: previous.incidents + 1,
      potholesCount: pCount,
      trashCount: tCount,
      priorityLevel,
    })
  }
  return [...groups.values()].sort((a, b) => b.frequency - a.frequency || b.latest.localeCompare(a.latest) || a.location.localeCompare(b.location))
}

export function duplicateCandidates(store: Store, report: Observation) {
  return store.incidents.filter(i => i.status !== 'Resolved' && i.category === report.category && normalizeLocation(i.location) === normalizeLocation(report.location))
}

export function addResidentReport(store: Store, report: NewReport, id: string, now: string): Store {
  const observation: Observation = {
    ...report,
    location: report.location.trim(),
    description: report.description.trim(),
    id,
    submittedAt: now,
    reporter: residentId,
    source: report.mediaType === 'video' ? 'Video observation' : 'Resident photo',
    confidence: report.confidence || 92,
    review: 'Pending review',
    reviewNote: '',
  }
  return { ...store, observations: [observation, ...store.observations] }
}

export function reviewReport(store: Store, reportId: string, decision: 'validate' | 'reject', targetId: string | undefined, note: string, now: string, newId: string): Store {
  const report = store.observations.find(o => o.id === reportId)
  if (!report || report.review !== 'Pending review') return store
  if (decision === 'reject') {
    if (!note.trim()) return store
    return { ...store, observations: store.observations.map(o => o.id === reportId ? { ...o, review: 'Rejected', reviewNote: note.trim() } : o) }
  }
  const target = targetId ? duplicateCandidates(store, report).find(i => i.id === targetId) : undefined
  if (targetId && !target) return store
  const id = target?.id || newId
  return {
    incidents: target ? store.incidents : [...store.incidents, { id, category: report.category, location: report.location, status: 'Open', note: note.trim() || 'Validated and awaiting inspection.', updatedAt: now }],
    observations: store.observations.map(o => o.id === reportId ? { ...o, review: 'Validated', incidentId: id, reviewNote: note.trim() || (target ? 'Linked to an existing incident.' : 'Validated as a new incident.') } : o),
  }
}

export function updateIncident(store: Store, id: string, status: IncidentStatus, note: string, now: string): Store {
  if (!incidentStatuses.includes(status) || !note.trim()) return store
  return { ...store, incidents: store.incidents.map(i => i.id === id ? { ...i, status, note: note.trim(), updatedAt: now } : i) }
}

export function addVideoObservations(store: Store, location: string, category: Category, now: string, batch: string): Store {
  const isPothole = category === 'Pothole'
  const observations: Observation[] = ['00:04', '00:12', '00:23'].map((clipTime, index) => {
    const sec = [4, 12, 23][index]
    const boxes: UiDetectionBox[] = isPothole ? [
      {
        id: `pot-${batch}-${index + 1}`,
        kind: 'pothole',
        label: 'Pothole',
        severity: index === 1 ? 'high' : 'moderate',
        confidencePct: [89, 94, 91][index],
        count: 1,
        boxPct: { x: 40 + index * 2, y: 62 + index * 2, w: 18, h: 12 },
        estimatedDiameterCm: 42.0,
      }
    ] : [
      {
        id: `tr-${batch}-${index + 1}`,
        kind: 'trash',
        label: index === 0 ? 'Plastic Litter' : 'Garbage Pile',
        category: index === 0 ? 'individual_litter' : 'garbage_pile',
        confidencePct: [88, 93, 90][index],
        count: index === 0 ? 1 : 4,
        boxPct: { x: 35 + index * 3, y: 55 + index * 2, w: 22, h: 16 },
      }
    ]

    return {
      id: `VID-${batch}-${index + 1}`,
      source: 'Video observation',
      reporter: 'Street Patrol Camera',
      location,
      category,
      observedAt: new Date(new Date(now).getTime() - (23 - [4, 12, 23][index]) * 1000).toISOString(),
      submittedAt: now,
      description: `Simulated ${category.toLowerCase()} observation at clip position ${clipTime}. Review before counting toward incident frequency.`,
      clipTime,
      confidence: [89, 94, 91][index],
      review: 'Pending review',
      reviewNote: '',
      mediaType: 'video',
      boxes,
      potholeCount: isPothole ? 1 : 0,
      trashCount: isPothole ? 0 : (index === 0 ? 1 : 4),
      hotspotScore: isPothole ? 62.5 : 85.0,
      priorityLevel: isPothole ? 'ELEVATED' : 'CRITICAL',
    }
  })

  return { ...store, observations: [...observations, ...store.observations] }
}
