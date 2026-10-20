export const categories = ['Garbage pile', 'Pothole', 'Road surface damage'] as const
export type Category = typeof categories[number]
export const locations = ['J. P. Rizal St. near Market', 'Shoe Avenue corner E. Dela Paz', 'P. Burgos St. riverside', 'Marikina Bridge approach']
export const incidentStatuses = ['Open', 'For inspection', 'In progress', 'Resolved'] as const
export type IncidentStatus = typeof incidentStatuses[number]
export type Observation = {
  id: string; reporter: string; source: 'Resident photo' | 'Video observation'; location: string;
  category: Category; observedAt: string; submittedAt: string; description: string;
  photo?: string; clipTime?: string; confidence: number;
  review: 'Pending review' | 'Validated' | 'Rejected'; incidentId?: string; reviewNote: string;
}
export type Incident = { id: string; category: Category; location: string; status: IncidentStatus; note: string; updatedAt: string }
export type Store = { observations: Observation[]; incidents: Incident[] }
export type RankedIncident = Incident & { frequency: number; latest: string }
export type NewReport = Pick<Observation, 'location' | 'category' | 'observedAt' | 'description' | 'photo'>
export const residentId = 'Demo resident'
export function normalizeLocation(value: string) { return value.trim().replace(/\s+/g, ' ').toLowerCase() }

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
    for (let n = 0; n < count; n++) observations.push({
      id: `OBS-${String(observations.length + 1).padStart(3, '0')}`,
      reporter: n === 0 ? residentId : 'Sample resident', source: 'Resident photo',
      location: incident.location, category: incident.category, observedAt: ago(3 + index * 3 + n * 20), submittedAt: ago(2 + index * 3 + n * 20),
      description: `Sample observation of ${incident.category.toLowerCase()} near this location.`,
      confidence: 90 - n, review: 'Validated', incidentId: incident.id, reviewNote: 'Validated by barangay admin.',
    })
  }
  observations.unshift({ id: 'OBS-013', reporter: 'Sample resident', source: 'Resident photo', location: locations[0], category: 'Garbage pile', observedAt: ago(1), submittedAt: ago(.5), description: 'Another observation of piled garbage beside the market entrance. Please check whether this is the existing incident.', confidence: 93, review: 'Pending review', reviewNote: '' })
  return { incidents, observations }
}

// Only validated observations contribute. Resolved incidents stay in the history.
export function rankIncidents(store: Store): RankedIncident[] {
  return store.incidents.filter(i => i.status !== 'Resolved').map(incident => {
    const evidence = store.observations.filter(o => o.review === 'Validated' && o.incidentId === incident.id)
    return { ...incident, frequency: evidence.length, latest: evidence.map(o => o.observedAt).sort().slice(-1)[0] || incident.updatedAt }
  }).filter(i => i.frequency > 0).sort((a, b) => b.frequency - a.frequency || b.latest.localeCompare(a.latest) || a.id.localeCompare(b.id))
}

export function rankHotspots(store: Store) {
  const groups = new Map<string, { location: string; frequency: number; latest: string; incidents: number }>()
  for (const incident of rankIncidents(store)) {
    const key = normalizeLocation(incident.location)
    const previous = groups.get(key) || { location: incident.location, frequency: 0, latest: '', incidents: 0 }
    groups.set(key, { location: previous.location, frequency: previous.frequency + incident.frequency, latest: previous.latest > incident.latest ? previous.latest : incident.latest, incidents: previous.incidents + 1 })
  }
  return [...groups.values()].sort((a, b) => b.frequency - a.frequency || b.latest.localeCompare(a.latest) || a.location.localeCompare(b.location))
}

export function duplicateCandidates(store: Store, report: Observation) {
  return store.incidents.filter(i => i.status !== 'Resolved' && i.category === report.category && normalizeLocation(i.location) === normalizeLocation(report.location))
}

export function addResidentReport(store: Store, report: NewReport, id: string, now: string): Store {
  const observation: Observation = { ...report, location: report.location.trim(), description: report.description.trim(), id, submittedAt: now, reporter: residentId, source: 'Resident photo', confidence: 92, review: 'Pending review', reviewNote: '' }
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
  const observations: Observation[] = ['00:04', '00:12', '00:23'].map((clipTime, index) => ({
    id: `VID-${batch}-${index + 1}`, source: 'Video observation', reporter: 'Demo camera', location, category,
    observedAt: new Date(new Date(now).getTime() - (23 - [4, 12, 23][index]) * 1000).toISOString(), submittedAt: now,
    description: `Simulated ${category.toLowerCase()} observation at clip position ${clipTime}. Review before counting toward incident frequency.`,
    clipTime, confidence: [89, 94, 91][index], review: 'Pending review', reviewNote: '',
  }))
  return { ...store, observations: [...observations, ...store.observations] }
}
