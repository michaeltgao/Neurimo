import { api } from "./client";

export type Questionnaire = {
  id?: number;
  visit_id?: number;
  regression: boolean;
  seizures: boolean;
  motor_delay: boolean;
  global_delay: boolean;
  family_history_asd_ndd: boolean;
  dysmorphic_features: boolean;
  macrocephaly: boolean;
  microcephaly: boolean;
  notes?: string | null;
};

export type Report = {
  visit: {
    id: number;
    child_id: number;
    visit_date: string;
    age_months: number;
  };
  asd_risk_bucket: "low" | "medium" | "med-high" | "high" | string;
  explanations: string[];
  prior_visits: {
    id: number;
    age_months: number;
    visit_date: string;
    asd_risk_bucket: string;
  }[];
};

export async function submitQuestionnaire(visitId: string, payload: Questionnaire): Promise<Questionnaire> {
  const res = await api.post(`/visits/${visitId}/questionnaire`, payload);
  return res.data;
}

export async function getQuestionnaire(visitId: string): Promise<Questionnaire> {
  const res = await api.get(`/visits/${visitId}/questionnaire`);
  return res.data;
}

export async function getReport(visitId: string): Promise<Report> {
  const res = await api.get<Report>(`/visits/${visitId}/report`);
  return res.data;
}
