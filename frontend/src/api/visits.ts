import { api } from "./client";

export type Visit = {
  id: number;
  child_id: number;
  visit_number: number;
  visit_date: string; // YYYY-MM-DD
  age_months: number;
  created_at: string;
};

export type VisitCreate = {
  visit_date: string; // YYYY-MM-DD
  age_months: number;
};

export type VisitUpdate = {
  visit_date?: string; // YYYY-MM-DD
  age_months?: number;
};

export async function listVisits(childId: number): Promise<Visit[]> {
  const res = await api.get<Visit[]>(`/children/${childId}/visits`);
  return res.data;
}

export async function createVisit(childId: number, payload: VisitCreate): Promise<Visit> {
  const res = await api.post<Visit>(`/children/${childId}/visits`, payload);
  return res.data;
}

export async function getVisit(visitId: string): Promise<Visit> {
  const res = await api.get<Visit>(`/visits/${visitId}`);
  return res.data;
}

export async function updateVisit(visitId: string, payload: VisitUpdate): Promise<Visit> {
  const res = await api.patch<Visit>(`/visits/${visitId}`, payload);
  return res.data;
}

export async function deleteVisit(visitId: string): Promise<void> {
  await api.delete(`/visits/${visitId}`);
}
