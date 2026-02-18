import { useState } from "react";
import type { Questionnaire, LikertValue, QuestionnaireResponses } from "../api/reports";

type Props = {
  initial?: Questionnaire;
  onSubmit: (q: Questionnaire) => Promise<void>;
  onUnsave?: () => void;
};

const LIKERT_OPTIONS: { value: LikertValue; label: string }[] = [
  { value: "always", label: "Always" },
  { value: "often", label: "Often" },
  { value: "sometimes", label: "Sometimes" },
  { value: "rarely", label: "Rarely" },
  { value: "never", label: "Never" },
];

// Section icons
const SocialIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
  </svg>
);

const MessageIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  </svg>
);

const RepeatIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="17 1 21 5 17 9" />
    <path d="M3 11V9a4 4 0 0 1 4-4h14" />
    <polyline points="7 23 3 19 7 15" />
    <path d="M21 13v2a4 4 0 0 1-4 4H3" />
  </svg>
);

const EyeIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
);

const SECTION_ICONS = [SocialIcon, MessageIcon, RepeatIcon, EyeIcon];

type QuestionSection = {
  title: string;
  subtitle: string;
  questions: { key: string; text: string }[];
};

const SECTIONS: QuestionSection[] = [
  {
    title: "Social Interaction",
    subtitle: "Questions about how the child interacts with others",
    questions: [
      { key: "social_responds_to_name", text: "Does the child respond when you call their name?" },
      { key: "social_eye_contact", text: "Does the child make eye contact during interactions?" },
      { key: "social_interest_in_children", text: "Does the child show interest in other children?" },
      { key: "social_smile_response", text: "Does the child smile in response to your smile?" },
      { key: "social_share_enjoyment", text: "Does the child share enjoyment or interests with you?" },
    ],
  },
  {
    title: "Communication & Language",
    subtitle: "Questions about the child's communication abilities",
    questions: [
      { key: "comm_gestures", text: "Does the child use gestures (pointing, waving) to communicate?" },
      { key: "comm_verbal_requests", text: "Does the child respond to simple verbal requests?" },
      { key: "comm_show_things", text: "Does the child try to show you things they find interesting?" },
      { key: "comm_babble_words", text: "Does the child babble or attempt to use words?" },
      { key: "comm_imitate", text: "Does the child imitate sounds or actions?" },
    ],
  },
  {
    title: "Repetitive Behaviors & Routines",
    subtitle: "Questions about repetitive behaviors and adherence to routines",
    questions: [
      { key: "rep_repetitive_movements", text: "Does the child engage in repetitive movements (hand flapping, rocking)?" },
      { key: "rep_insist_sameness", text: "Does the child insist on sameness or become upset by changes?" },
      { key: "rep_intense_interests", text: "Does the child have unusually intense interests in specific objects or topics?" },
      { key: "rep_line_up_objects", text: "Does the child line up toys or objects in specific patterns?" },
      { key: "rep_difficulty_transitions", text: "Does the child have difficulty with transitions between activities?" },
    ],
  },
  {
    title: "Sensory Responses",
    subtitle: "Questions about the child's responses to sensory input",
    questions: [
      { key: "sens_unusual_reactions", text: "Does the child have unusual reactions to sounds, textures, or lights?" },
      { key: "sens_seek_sensory", text: "Does the child seek out certain sensory experiences (spinning, touching)?" },
      { key: "sens_distressed_noises", text: "Does the child become distressed by everyday noises?" },
      { key: "sens_food_texture", text: "Does the child have strong preferences or aversions to certain foods or textures?" },
    ],
  },
];

function likertColor(_value: LikertValue): { bg: string; border: string; text: string } {
  // All selected options use the same black/gray styling
  return { bg: "var(--color-bg-tertiary)", border: "var(--color-text)", text: "var(--color-text)" };
}

export default function QuestionnaireForm({ initial, onSubmit, onUnsave }: Props) {
  const [responses, setResponses] = useState<QuestionnaireResponses>(
    initial?.responses ?? {}
  );
  const [status, setStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [err, setErr] = useState("");

  function setAnswer(key: string, value: LikertValue) {
    // Don't do anything if clicking the same answer
    if (responses[key] === value) return;

    // Reset to unsaved when data changes
    if (status === "saved") {
      setStatus("idle");
      onUnsave?.();
    }
    setResponses((prev) => ({ ...prev, [key]: value }));
  }

  const totalQuestions = SECTIONS.reduce((sum, s) => sum + s.questions.length, 0);
  const answeredCount = Object.keys(responses).length;
  const allAnswered = answeredCount === totalQuestions;

  async function handleSave() {
    setStatus("saving");
    setErr("");
    try {
      const payload: Questionnaire = {
        ...initial, // Preserve existing data (including family_history)
        regression: initial?.regression ?? false,
        seizures: initial?.seizures ?? false,
        motor_delay: initial?.motor_delay ?? false,
        global_delay: initial?.global_delay ?? false,
        family_history_asd_ndd: initial?.family_history_asd_ndd ?? false,
        dysmorphic_features: initial?.dysmorphic_features ?? false,
        macrocephaly: initial?.macrocephaly ?? false,
        microcephaly: initial?.microcephaly ?? false,
        responses,
      };
      await onSubmit(payload);
      setStatus("saved");
    } catch (e: unknown) {
      setStatus("error");
      const err = e as { response?: { data?: { detail?: string } }; message?: string };
      setErr(err?.response?.data?.detail ?? err?.message ?? "Save failed");
    }
  }

  return (
    <form onSubmit={(e) => e.preventDefault()}>
      {/* Section header */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 12, marginBottom: 20 }}>
        <div style={{
          padding: 10,
          backgroundColor: "var(--color-bg-tertiary)",
          borderRadius: 10,
          color: "var(--color-text-secondary)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
            <polyline points="10 9 9 9 8 9" />
          </svg>
        </div>
        <div style={{ flex: 1 }}>
          <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 2 }}>Behavioral Assessment</h2>
          <p style={{ color: "var(--color-text-secondary)", fontSize: "0.8125rem", margin: 0 }}>
            Answer all questions about the child's development
          </p>
        </div>
        <div style={{
          padding: "6px 12px",
          backgroundColor: allAnswered ? "#dcfce7" : "var(--color-bg-tertiary)",
          borderRadius: 20,
          fontSize: "0.75rem",
          fontWeight: 600,
          color: allAnswered ? "#166534" : "var(--color-text-secondary)",
        }}>
          {answeredCount}/{totalQuestions}
        </div>
      </div>

      <div style={{ display: "grid", gap: 20 }}>
        {SECTIONS.map((section, sectionIdx) => {
          const SectionIcon = SECTION_ICONS[sectionIdx];
          const sectionAnswered = section.questions.filter(q => responses[q.key]).length;
          const sectionTotal = section.questions.length;

          return (
            <div
              key={section.title}
              style={{
                backgroundColor: "#f9fafb",
                borderRadius: 12,
                padding: "20px 20px 16px",
                border: "1px solid var(--color-border-light)",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                <div style={{
                  padding: 8,
                  backgroundColor: "#fff",
                  borderRadius: 8,
                  color: "var(--color-text-secondary)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  border: "1px solid var(--color-border)",
                }}>
                  <SectionIcon />
                </div>
                <div style={{ flex: 1 }}>
                  <h3 style={{ fontSize: "0.9375rem", fontWeight: 600, marginBottom: 2 }}>
                    {section.title}
                  </h3>
                  <p style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)", margin: 0 }}>
                    {section.subtitle}
                  </p>
                </div>
                <div style={{
                  fontSize: "0.6875rem",
                  fontWeight: 600,
                  padding: "4px 8px",
                  backgroundColor: sectionAnswered === sectionTotal ? "#dcfce7" : "#fff",
                  color: sectionAnswered === sectionTotal ? "#166534" : "var(--color-text-secondary)",
                  borderRadius: 4,
                  border: "1px solid var(--color-border)",
                }}>
                  {sectionAnswered}/{sectionTotal}
                </div>
              </div>

              <div style={{ display: "grid", gap: 0 }}>
                {section.questions.map((q, idx) => {
                  const selected = responses[q.key];
                  return (
                    <div
                      key={q.key}
                      style={{
                        padding: "14px 0",
                        borderTop: idx > 0 ? "1px solid #e5e7eb" : "none",
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "flex-start", gap: 8, marginBottom: 10 }}>
                        <span style={{
                          fontSize: "0.875rem",
                          fontWeight: 600,
                          color: "var(--color-text)",
                        }}>
                          {idx + 1}.
                        </span>
                        <span style={{ fontSize: "0.875rem", fontWeight: 500, color: "#1f2937", lineHeight: 1.4 }}>
                          {q.text}
                        </span>
                      </div>
                      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginLeft: 28 }}>
                        {LIKERT_OPTIONS.map((opt) => {
                          const isSelected = selected === opt.value;
                          const colors = likertColor(opt.value);
                          return (
                            <button
                              key={opt.value}
                              type="button"
                              onClick={() => setAnswer(q.key, opt.value)}
                              style={{
                                padding: "5px 14px",
                                borderRadius: 16,
                                border: isSelected
                                  ? `2px solid ${colors.border}`
                                  : "1px solid #d1d5db",
                                backgroundColor: isSelected ? colors.bg : "#fff",
                                color: isSelected ? colors.text : "#6b7280",
                                fontSize: "0.75rem",
                                fontWeight: isSelected ? 600 : 400,
                                cursor: "pointer",
                                transition: "all 0.15s ease",
                                outline: "none",
                              }}
                            >
                              {opt.label}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      <div
        style={{
          marginTop: 24,
          padding: 16,
          backgroundColor: "var(--color-bg-secondary)",
          borderRadius: 10,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {status === "saved" ? (
            <button
              type="button"
              onClick={() => {
                setStatus("idle");
                onUnsave?.();
              }}
              style={{
                padding: "10px 20px",
                backgroundColor: "var(--color-bg-tertiary)",
                color: "var(--color-text-secondary)",
                borderColor: "var(--color-border)",
                cursor: "pointer",
              }}
            >
              ✓ Saved
            </button>
          ) : (
            <button
              type="button"
              onClick={handleSave}
              disabled={status === "saving" || !allAnswered}
              className="primary"
              style={{
                opacity: allAnswered ? 1 : 0.5,
                padding: "10px 20px",
              }}
            >
              {status === "saving" ? "Saving..." : "Save Questionnaire"}
            </button>
          )}

          {status === "error" && (
            <span style={{ color: "var(--color-error)", fontSize: "0.875rem" }}>{err}</span>
          )}
        </div>

        <span style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)" }}>
          {allAnswered ? "All questions answered" : `${totalQuestions - answeredCount} remaining`}
        </span>
      </div>
    </form>
  );
}
