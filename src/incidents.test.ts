import { test } from 'node:test'
import assert from 'node:assert/strict'
import { seedStore, addResidentReport, reviewReport, updateIncident, rankIncidents, rankHotspots, addVideoObservations } from './incidents.ts'

test('user submission, duplicate consolidation, status update and resolved rankings', () => {
  const now = '2026-10-13T04:00:00.000Z'
  let store = seedStore(new Date(now))
  const before = rankIncidents(store)[0].frequency
  store = addResidentReport(store, { category: 'Garbage pile', location: 'J. P. Rizal St. near Market', observedAt: now, description: 'Demo report' }, 'TEST-1', now)
  assert.equal(rankIncidents(store)[0].frequency, before)
  store = reviewReport(store, 'TEST-1', 'validate', 'INC-001', 'Same physical issue', now, 'NEW-1')
  assert.equal(store.incidents.length, 4)
  assert.equal(rankIncidents(store)[0].frequency, before + 1)
  assert.equal(store.observations.find(o => o.id === 'TEST-1')?.incidentId, 'INC-001')
  const same = reviewReport(store, 'TEST-1', 'validate', 'INC-001', '', now, 'NEW-2')
  assert.equal(same, store)
  store = updateIncident(store, 'INC-001', 'Resolved', 'Cleanup completed', now)
  assert.equal(store.incidents[0].note, 'Cleanup completed')
  assert.ok(!rankIncidents(store).some(i => i.id === 'INC-001'))
  assert.ok(!rankHotspots(store).some(i => i.location === 'J. P. Rizal St. near Market'))
})

test('ties use recency, rejected and video observations wait for validation', () => {
  const now = '2026-10-13T04:00:00.000Z'
  let store = seedStore(new Date(now))
  assert.deepEqual(rankIncidents(store).map(i => i.id), ['INC-001', 'INC-002', 'INC-003'])
  const frequencies = rankIncidents(store).map(i => i.frequency)
  store = addVideoObservations(store, 'J. P. Rizal St. near Market', 'Garbage pile', now, 'test')
  assert.deepEqual(rankIncidents(store).map(i => i.frequency), frequencies)
  assert.equal(store.observations.filter(o => o.source === 'Video observation').length, 3)
  assert.equal(reviewReport(store, 'VID-test-1', 'reject', undefined, '', now, 'unused'), store)
  store = reviewReport(store, 'VID-test-1', 'reject', undefined, 'Not enough evidence', now, 'unused')
  assert.equal(store.observations.find(o => o.id === 'VID-test-1')?.review, 'Rejected')
  assert.deepEqual(rankIncidents(store).map(i => i.frequency), frequencies)
})

test('different categories cannot be consolidated and new incidents can be created', () => {
  const now = '2026-10-13T04:00:00.000Z'
  let store = seedStore(new Date(now))
  store = addResidentReport(store, { category: 'Pothole', location: 'J. P. Rizal St. near Market', observedAt: now, description: 'Separate anomaly' }, 'TEST-2', now)
  assert.equal(reviewReport(store, 'TEST-2', 'validate', 'INC-001', '', now, 'NEW'), store)
  store = reviewReport(store, 'TEST-2', 'validate', undefined, '', now, 'NEW')
  assert.equal(store.incidents.length, 5)
  assert.equal(store.observations.find(o => o.id === 'TEST-2')?.incidentId, 'NEW')
  assert.equal(rankHotspots(store)[0].incidents, 2)
})
