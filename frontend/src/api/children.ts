import { api } from "./client";

export type Child = {
  id: number;
  pseudo_id: string;
  birthdate: string; // ISO date
  sex: string;
  clinic_id: string;
  created_at: string;
};

export type ChildCreate = {
  pseudo_id: string;
  birthdate: string; // YYYY-MM-DD
  sex: string;
  clinic_id?: string;
};

export async function listChildren(): Promise<Child[]> {
  const res = await api.get<Child[]>("/children");
  return res.data;
}

export async function createChild(payload: ChildCreate): Promise<Child> {
  const res = await api.post<Child>("/children", payload);
  return res.data;
}

export async function getChild(childId: number): Promise<Child> {
  const res = await api.get<Child>(`/children/${childId}`);
  return res.data;
}

export async function deleteChild(childId: number): Promise<void> {
  await api.delete(`/children/${childId}`);
}

