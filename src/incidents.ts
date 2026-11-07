export const categories = ['Trash', 'Pothole'] as const
export type Category = typeof categories[number]
export const locations = [
  'J. P. Rizal St. near Market',
  'Shoe Avenue corner E. Dela Paz',
  'P. Burgos St. riverside',
  'Marikina Bridge approach',
]
export const incidentStatuses = ['Open', 'For inspection', 'In progress', 'Resolved'] as const
export type IncidentStatus = typeof incidentStatuses[number]

export type TrashDetection = {
  detection_id: number
  category_id: number
  label: string
  bantai_category: string
  model_score: number
  bbox: [number, number, number, number]
}

export type TrashRegion = {
  region_id: number
  category: 'individual_litter' | 'group_of_individual_litter' | 'garbage_pile'
  display_label: string
  item_count: number
  bbox: [number, number, number, number]
  model_score: number | null
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
  detections?: TrashDetection[]
}

export type RoadDefect = {
  defect_id: number
  type: 'pothole'
  severity: 'low' | 'moderate' | 'high'
  estimated_diameter_cm: number
  bbox: [number, number, number, number]
  label?: string
  raw_severity?: string
  xyxy?: [number, number, number, number]
  box_pct?: { x: number; y: number; w: number; h: number }
  boxPct?: { x: number; y: number; w: number; h: number }
  confidence: number
  confidence_pct?: number
  confidencePct?: number
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
  frame_hotspot_score?: number
  frame_priority_level?: 'CRITICAL' | 'ELEVATED' | 'ROUTINE'
}

export type UnifiedIncidentResponse = {
  incident_id: string
  source_media: {
    filename: string
    type: 'video' | 'image'
    fps?: number
    duration_sec?: number
    media_url?: string
  }
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
    total_garbage_piles: number
    total_litter_groups: number
    total_individual_litter: number
    total_potholes: number
  }
  frames: DualModelFrameResult[]
}

export type Observation = {
  id: string
  reporter: string
  source: 'Resident photo' | 'Video observation'
  location: string
  category: Category
  observedAt: string
  submittedAt: string
  description: string
  photo?: string
  videoUrl?: string
  clipTime?: string
  confidence: number
  review: 'Pending review' | 'Validated' | 'Rejected'
  incidentId?: string
  reviewNote: string
  dualModelResult?: UnifiedIncidentResponse
  detections?: TrashDetection[]
  regions?: TrashRegion[]
  imageWidth?: number
  imageHeight?: number
  defects?: RoadDefect[]
  hotspotScore?: number
  priorityLevel?: 'CRITICAL' | 'ELEVATED' | 'ROUTINE'
}

export function computeHotspotScore(
  piles: number,
  groups: number,
  litter: number,
  potholesBySeverity: { low?: number; moderate?: number; high?: number } = {},
): number {
  const W_pile = 35.0
  const W_group = 15.0
  const W_litter = 3.0
  const W_pothole: Record<string, number> = { low: 25.0, moderate: 32.5, high: 40.0 }
  let score = piles * W_pile + groups * W_group + litter * W_litter
  for (const [sev, count] of Object.entries(potholesBySeverity)) {
    score += (count || 0) * (W_pothole[sev] || 30.0)
  }
  return Math.min(100.0, Math.max(0.0, Math.round(score * 10) / 10))
}

export type Incident = {
  id: string
  category: Category
  location: string
  status: IncidentStatus
  note: string
  updatedAt: string
}

export type Store = { observations: Observation[]; incidents: Incident[] }
export type RankedIncident = Incident & { frequency: number; latest: string }

export type HotspotLeaderboardEntry = {
  location: string
  subDistrict: string
  frequency: number
  latest: string
  incidents: number
  trashIncidents: number
  potholeIncidents: number
  trashObservations: number
  potholeObservations: number
  primaryIssue: 'Trash Concentration' | 'Road Surface Hazard' | 'Dual-Hazard Zone'
  hotspotScore: number
  priorityLevel: 'CRITICAL' | 'ELEVATED' | 'ROUTINE'
  trendStatus: 'Escalating' | 'New' | 'Stable' | 'Resolving'
}

export type NewReport = Pick<
  Observation,
  | 'location'
  | 'category'
  | 'observedAt'
  | 'description'
  | 'photo'
  | 'videoUrl'
  | 'dualModelResult'
  | 'detections'
  | 'regions'
  | 'imageWidth'
  | 'imageHeight'
  | 'defects'
  | 'hotspotScore'
  | 'priorityLevel'
>

export const residentId = 'Citizen (Brgy. Santa Elena)'

export function normalizeLocation(value: string) {
  return value.trim().replace(/\s+/g, ' ').toLowerCase()
}

const STORE_STORAGE_KEY = 'bantai_municipal_store_v2'

export function seedStore(now = new Date()): Store {
  const ago = (hours: number) => new Date(now.getTime() - hours * 3600000).toISOString()
  const incidents: Incident[] = [
    {
      id: 'INC-001',
      category: 'Trash',
      location: locations[0],
      status: 'Open',
      note: 'Awaiting barangay sanitation crew dispatch.',
      updatedAt: ago(3),
    },
    {
      id: 'INC-002',
      category: 'Pothole',
      location: locations[1],
      status: 'For inspection',
      note: 'Road engineering team scheduled to inspect crater depth.',
      updatedAt: ago(5),
    },
    {
      id: 'INC-003',
      category: 'Pothole',
      location: locations[3],
      status: 'In progress',
      note: 'Field inspection completed. Asphalt patching crew coordinated.',
      updatedAt: ago(8),
    },
    {
      id: 'INC-004',
      category: 'Trash',
      location: locations[2],
      status: 'Resolved',
      note: 'Waste clearance completed and verified by barangay inspector.',
      updatedAt: ago(12),
    },
  ]
  const observations: Observation[] = []
  for (const [index, incident] of incidents.entries()) {
    const count = [4, 3, 3, 2][index]
    for (let n = 0; n < count; n++) {
      observations.push({
        id: `OBS-${String(observations.length + 1).padStart(3, '0')}`,
        reporter: n === 0 ? residentId : 'Verified Citizen Reporter',
        source: 'Resident photo',
        location: incident.location,
        category: incident.category,
        observedAt: ago(3 + index * 3 + n * 20),
        submittedAt: ago(2 + index * 3 + n * 20),
        description: `Field report of ${incident.category.toLowerCase()} accumulation near ${incident.location}.`,
        confidence: 90 - n,
        review: 'Validated',
        incidentId: incident.id,
        reviewNote: 'Validated by barangay operations.',
        hotspotScore: incident.category === 'Trash' ? 50.0 : 65.0,
        priorityLevel: incident.category === 'Trash' ? 'ELEVATED' : 'ELEVATED',
      })
    }
  }
  observations.unshift({
    id: 'OBS-013',
    reporter: 'Verified Citizen Reporter',
    source: 'Resident photo',
    location: locations[0],
    category: 'Trash',
    observedAt: ago(1),
    submittedAt: ago(0.5),
    description: 'Uncollected waste bags and scattered packaging beside the public market entrance.',
    confidence: 93,
    review: 'Pending review',
    reviewNote: '',
    hotspotScore: 53.0,
    priorityLevel: 'ELEVATED',
  })
  return { incidents, observations }
}

function deduplicateMultiFrameObservations(observations: Observation[]): Observation[] {
  const seenIncidentIds = new Set<string>()
  const seenBaseIds = new Set<string>()
  const result: Observation[] = []

  for (const obs of observations) {
    const dualId = obs.dualModelResult?.incident_id
    const isSplitVideoObs =
      (obs.dualModelResult?.frames && obs.dualModelResult.frames.length > 1) ||
      /\s*\[Frame \d+ @ [\d.]+s\]$/i.test(obs.description)

    if (isSplitVideoObs) {
      const baseId = obs.id.replace(/-\d+$/, '')
      if (dualId && seenIncidentIds.has(dualId)) {
        continue
      }
      if (seenBaseIds.has(baseId)) {
        continue
      }
      if (dualId) seenIncidentIds.add(dualId)
      seenBaseIds.add(baseId)

      const firstFrame = obs.dualModelResult?.frames?.[0]
      result.push({
        ...obs,
        id: baseId,
        description: obs.description.replace(/\s*\[Frame \d+ @ [\d.]+s\]$/i, '').trim() || obs.category,
        photo: firstFrame?.frame_url || obs.photo,
        detections: firstFrame?.trash_findings?.detections || obs.detections,
        regions: firstFrame?.trash_findings?.region_list || obs.regions,
        defects: firstFrame?.road_defect_findings?.defects || obs.defects,
        hotspotScore: obs.dualModelResult?.hotspot_score ?? obs.hotspotScore,
        priorityLevel: obs.dualModelResult?.priority_level ?? obs.priorityLevel,
      })
    } else {
      result.push(obs)
    }
  }
  return result
}

export function loadPersistedStore(): Store {
  if (typeof window === 'undefined' || !window.localStorage) {
    return seedStore()
  }
  try {
    const raw = window.localStorage.getItem(STORE_STORAGE_KEY)
    if (!raw) return seedStore()
    const parsed = JSON.parse(raw) as Store
    if (!parsed || !Array.isArray(parsed.incidents) || !Array.isArray(parsed.observations)) {
      return seedStore()
    }
    return {
      ...parsed,
      observations: deduplicateMultiFrameObservations(parsed.observations),
    }
  } catch {
    return seedStore()
  }
}

export function savePersistedStore(store: Store): void {
  if (typeof window === 'undefined' || !window.localStorage) return
  try {
    window.localStorage.setItem(STORE_STORAGE_KEY, JSON.stringify(store))
  } catch {
    // Ignore storage quota errors
  }
}

export function resetPersistedStore(): Store {
  const fresh = seedStore()
  savePersistedStore(fresh)
  return fresh
}

// Only validated observations contribute. Resolved incidents stay in the history.
export function rankIncidents(store: Store): RankedIncident[] {
  return store.incidents
    .filter(i => i.status !== 'Resolved')
    .map(incident => {
      const evidence = store.observations.filter(
        o => o.review === 'Validated' && o.incidentId === incident.id,
      )
      return {
        ...incident,
        frequency: evidence.length,
        latest:
          evidence
            .map(o => o.observedAt)
            .sort()
            .slice(-1)[0] || incident.updatedAt,
      }
    })
    .filter(i => i.frequency > 0)
    .sort(
      (a, b) =>
        b.frequency - a.frequency ||
        b.latest.localeCompare(a.latest) ||
        a.id.localeCompare(b.id),
    )
}

export function rankHotspots(
  store: Store,
  categoryFilter: Category | 'All' = 'All',
): HotspotLeaderboardEntry[] {
  const groups = new Map<
    string,
    {
      location: string
      frequency: number
      latest: string
      incidents: number
      trashIncidents: number
      potholeIncidents: number
      trashObservations: number
      potholeObservations: number
      maxObsHotspotScore: number
      hasOpen: boolean
      hasInProgress: boolean
    }
  >()

  const activeRanked = rankIncidents(store).filter(
    inc => categoryFilter === 'All' || inc.category === categoryFilter,
  )

  for (const incident of activeRanked) {
    const key = normalizeLocation(incident.location)
    const previous = groups.get(key) || {
      location: incident.location,
      frequency: 0,
      latest: '',
      incidents: 0,
      trashIncidents: 0,
      potholeIncidents: 0,
      trashObservations: 0,
      potholeObservations: 0,
      maxObsHotspotScore: 0,
      hasOpen: false,
      hasInProgress: false,
    }

    const isTrash = incident.category === 'Trash'
    const linkedObs = store.observations.filter(
      o => o.review === 'Validated' && o.incidentId === incident.id,
    )
    const maxScoreInIncident = linkedObs.reduce(
      (acc, o) => Math.max(acc, o.hotspotScore ?? 0),
      0,
    )

    groups.set(key, {
      location: previous.location,
      frequency: previous.frequency + incident.frequency,
      latest: previous.latest > incident.latest ? previous.latest : incident.latest,
      incidents: previous.incidents + 1,
      trashIncidents: previous.trashIncidents + (isTrash ? 1 : 0),
      potholeIncidents: previous.potholeIncidents + (isTrash ? 0 : 1),
      trashObservations: previous.trashObservations + (isTrash ? incident.frequency : 0),
      potholeObservations: previous.potholeObservations + (isTrash ? 0 : incident.frequency),
      maxObsHotspotScore: Math.max(previous.maxObsHotspotScore, maxScoreInIncident),
      hasOpen: previous.hasOpen || incident.status === 'Open',
      hasInProgress: previous.hasInProgress || incident.status === 'In progress',
    })
  }

  return [...groups.values()]
    .map(g => {
      let primaryIssue: HotspotLeaderboardEntry['primaryIssue'] = 'Trash Concentration'
      if (g.trashIncidents > 0 && g.potholeIncidents > 0) {
        primaryIssue = 'Dual-Hazard Zone'
      } else if (g.potholeIncidents > 0) {
        primaryIssue = 'Road Surface Hazard'
      }

      // Compute calibrated municipal hotspot score combining incident density & model severity
      const baseCalc = Math.min(
        100,
        Math.round(
          (g.trashObservations * 18.0 +
            g.potholeObservations * 26.0 +
            g.incidents * 12.0) *
            10,
        ) / 10,
      )
      const hotspotScore = Math.max(baseCalc, g.maxObsHotspotScore)
      const priorityLevel: 'CRITICAL' | 'ELEVATED' | 'ROUTINE' =
        hotspotScore >= 75 ? 'CRITICAL' : hotspotScore >= 40 ? 'ELEVATED' : 'ROUTINE'

      let trendStatus: HotspotLeaderboardEntry['trendStatus'] = 'Stable'
      if (g.frequency >= 4 && g.hasOpen) trendStatus = 'Escalating'
      else if (g.hasInProgress) trendStatus = 'Resolving'
      else if (g.frequency <= 2 && g.hasOpen) trendStatus = 'New'

      return {
        location: g.location,
        subDistrict: 'Brgy. Santa Elena · Marikina City',
        frequency: g.frequency,
        latest: g.latest,
        incidents: g.incidents,
        trashIncidents: g.trashIncidents,
        potholeIncidents: g.potholeIncidents,
        trashObservations: g.trashObservations,
        potholeObservations: g.potholeObservations,
        primaryIssue,
        hotspotScore,
        priorityLevel,
        trendStatus,
      }
    })
    .sort(
      (a, b) =>
        b.frequency - a.frequency ||
        b.latest.localeCompare(a.latest) ||
        a.location.localeCompare(b.location),
    )
}

export function duplicateCandidates(store: Store, report: Observation) {
  return store.incidents.filter(
    i =>
      i.status !== 'Resolved' &&
      i.category === report.category &&
      normalizeLocation(i.location) === normalizeLocation(report.location),
  )
}

export function addResidentReport(store: Store, report: NewReport, id: string, now: string): Store {
  const observation: Observation = {
    ...report,
    location: report.location.trim(),
    description: report.description.trim(),
    id,
    submittedAt: now,
    reporter: residentId,
    source: 'Resident photo',
    confidence: 92,
    review: 'Pending review',
    reviewNote: '',
  }
  return { ...store, observations: [observation, ...store.observations] }
}

export function reviewReport(
  store: Store,
  reportId: string,
  decision: 'validate' | 'reject',
  targetId: string | undefined,
  note: string,
  now: string,
  newId: string,
): Store {
  const report = store.observations.find(o => o.id === reportId)
  if (!report || report.review !== 'Pending review') return store
  if (decision === 'reject') {
    if (!note.trim()) return store
    return {
      ...store,
      observations: store.observations.map(o =>
        o.id === reportId ? { ...o, review: 'Rejected', reviewNote: note.trim() } : o,
      ),
    }
  }
  const target = targetId
    ? duplicateCandidates(store, report).find(i => i.id === targetId)
    : undefined
  if (targetId && !target) return store
  const id = target?.id || newId
  return {
    incidents: target
      ? store.incidents
      : [
          ...store.incidents,
          {
            id,
            category: report.category,
            location: report.location,
            status: 'Open',
            note: note.trim() || 'Validated and awaiting field inspection.',
            updatedAt: now,
          },
        ],
    observations: store.observations.map(o =>
      o.id === reportId
        ? {
            ...o,
            review: 'Validated',
            incidentId: id,
            reviewNote:
              note.trim() ||
              (target ? 'Linked to an existing active incident.' : 'Validated as a new municipal incident.'),
          }
        : o,
    ),
  }
}

export function updateIncident(
  store: Store,
  id: string,
  status: IncidentStatus,
  note: string,
  now: string,
): Store {
  if (!incidentStatuses.includes(status) || !note.trim()) return store
  return {
    ...store,
    incidents: store.incidents.map(i =>
      i.id === id ? { ...i, status, note: note.trim(), updatedAt: now } : i,
    ),
  }
}

export function addVideoObservations(
  store: Store,
  location: string,
  category: Category,
  now: string,
  batch: string,
): Store {
  const observations: Observation[] = ['00:04', '00:12', '00:23'].map((clipTime, index) => ({
    id: `VID-${batch}-${index + 1}`,
    source: 'Video observation',
    reporter: 'Patrol Camera 01',
    location,
    category,
    observedAt: new Date(new Date(now).getTime() - (23 - [4, 12, 23][index]) * 1000).toISOString(),
    submittedAt: now,
    description: `Automated patrol camera ${category.toLowerCase()} capture at clip timestamp ${clipTime}. Queued for barangay verification.`,
    clipTime,
    confidence: [89, 94, 91][index],
    review: 'Pending review',
    reviewNote: '',
  }))
  return { ...store, observations: [...observations, ...store.observations] }
}
