import { useState } from "react";
import type { Questionnaire } from "../api/reports";

type Props = {
  initial?: Questionnaire;
  onSubmit: (q: Questionnaire) => Promise<void>;
};

const defaultQ: Questionnaire = {
  regression: false,
  seizures: false,
  motor_delay: false,
  global_delay: false,
  family_history_asd_ndd: false,
  dysmorphic_features: false,
  macrocephaly: false,
  microcephaly: false,
  notes: "",
};

export default function QuestionnaireForm({ initial, onSubmit }: Props) {
  const [q, setQ] = useState<Questionnaire>(initial ?? defaultQ);
  const [status, setStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [err, setErr] = useState<string>("");

  function toggle(key: keyof Omit<Questionnaire, "notes">) {
    setQ((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("saving");
    setErr("");
    try {
      await onSubmit(q);
      setStatus("saved");
    } catch (e: unknown) {
      setStatus("error");
      const err = e as { response?: { data?: { detail?: string } }; message?: string };
      setErr(err?.response?.data?.detail ?? err?.message ?? "Save failed");
    }
  }

  const checkboxItems = [
    { key: "regression" as const, label: "Regression (lost skills)" },
    { key: "seizures" as const, label: "Seizures" },
    { key: "motor_delay" as const, label: "Motor delay" },
    { key: "global_delay" as const, label: "Global delay" },
    { key: "family_history_asd_ndd" as const, label: "Family history of ASD/NDD" },
    { key: "dysmorphic_features" as const, label: "Dysmorphic features" },
    { key: "macrocephaly" as const, label: "Macrocephaly" },
    { key: "microcephaly" as const, label: "Microcephaly" },
  ];

  return (
    <form onSubmit={handleSubmit}>
      <div
        style={{
          border: "1px solid var(--color-border)",
          borderRadius: 8,
          overflow: "hidden",
          marginBottom: 20,
        }}
      >
        {checkboxItems.map((item, index) => (
          <label
            key={item.key}
            style={{
              display: "flex",
              alignItems: "center",
              padding: "12px 16px",
              cursor: "pointer",
              borderBottom: index < checkboxItems.length - 1 ? "1px solid var(--color-border)" : "none",
              backgroundColor: q[item.key] ? "var(--color-bg-secondary)" : "transparent",
              transition: "background-color 0.15s ease",
              fontSize: "0.875rem",
            }}
          >
            <input
              type="checkbox"
              checked={q[item.key]}
              onChange={() => toggle(item.key)}
              style={{ width: 16, height: 16, marginRight: 12 }}
            />
            {item.label}
          </label>
        ))}
      </div>

      <div style={{ marginBottom: 20 }}>
        <label
          htmlFor="notes"
          style={{ display: "block", marginBottom: 6, fontSize: "0.875rem", fontWeight: 500 }}
        >
          Additional notes
        </label>
        <textarea
          id="notes"
          value={q.notes ?? ""}
          onChange={(e) => setQ((prev) => ({ ...prev, notes: e.target.value }))}
          rows={3}
          placeholder="Any additional observations..."
          style={{ resize: "vertical" }}
        />
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <button type="submit" disabled={status === "saving"} className="primary">
          {status === "saving" ? "Saving..." : "Save questionnaire"}
        </button>

        {status === "saved" && (
          <span style={{ color: "var(--color-success)", fontSize: "0.875rem" }}>Saved</span>
        )}
        {status === "error" && (
          <span style={{ color: "var(--color-error)", fontSize: "0.875rem" }}>{err}</span>
        )}
      </div>
    </form>
  );
}
