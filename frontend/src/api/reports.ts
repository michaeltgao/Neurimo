import { api } from "./client";

export type LikertValue = "always" | "often" | "sometimes" | "rarely" | "never";

export type QuestionnaireResponses = Record<string, LikertValue>;

export type FamilyHistoryData = Record<string, Record<string, boolean>>;

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
  responses?: QuestionnaireResponses | null;
  family_history?: FamilyHistoryData | null;
  notes?: string | null;
};

export type ExplanationsByTask = {
  joint_attention: string[];
  imitation: string[];
  free_play: string[];
  questionnaire: string[];
  general: string[];
};

export type Report = {
  visit: {
    id: number;
    child_id: number;
    visit_date: string;
    age_months: number;
  };
  asd_risk_bucket: "low" | "moderate" | "moderate-high" | "high" | string;
  risk_score: number | null;
  explanations: string[];
  explanations_by_task?: ExplanationsByTask | null;
  prior_visits: {
    id: number;
    age_months: number;
    visit_date: string;
    asd_risk_bucket: string;
    risk_score: number | null;
    visit_number: number;
    is_current: boolean;
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
