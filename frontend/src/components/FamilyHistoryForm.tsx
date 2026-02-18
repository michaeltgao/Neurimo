import { useState } from "react";

export type FamilyHistoryData = Record<string, Record<string, boolean>>;

type Props = {
  initial?: FamilyHistoryData;
  onSubmit: (data: FamilyHistoryData) => Promise<void>;
  onUnsave?: () => void;
};

const UsersIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
  </svg>
);

const CONDITIONS = [
  { key: "anxiety", label: "Anxiety" },
  { key: "adhd", label: "ADHD/ADD" },
  { key: "asd", label: "Autism Spectrum Disorder" },
  { key: "bipolar", label: "Bipolar Disorder" },
  { key: "depression", label: "Depression" },
  { key: "epilepsy", label: "Epilepsy/Seizure Disorder" },
  { key: "genetic", label: "Genetic Condition" },
  { key: "intellectual_disability", label: "Intellectual Disability" },
  { key: "language_disorder", label: "Language Disorder" },
  { key: "learning_disability", label: "Learning Disability" },
  { key: "tics", label: "Motor or Vocal Tics" },
  { key: "psychosis", label: "Psychosis or Schizophrenia" },
];

const FAMILY_MEMBERS = [
  { key: "mother", label: "Mother" },
  { key: "father", label: "Father" },
  { key: "brother", label: "Brother" },
  { key: "sister", label: "Sister" },
  { key: "grandparent", label: "Grandparent" },
  { key: "aunt_uncle", label: "Aunt/Uncle" },
  { key: "other", label: "Other" },
];

export default function FamilyHistoryForm({ initial, onSubmit, onUnsave }: Props) {
  const [data, setData] = useState<FamilyHistoryData>(initial ?? {});
  const [status, setStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [err, setErr] = useState("");

  function toggle(condition: string, member: string) {
    // Reset to unsaved when data changes
    if (status === "saved") {
      setStatus("idle");
      onUnsave?.();
    }
    setData((prev) => {
      const conditionData = prev[condition] ?? {};
      const current = conditionData[member] ?? false;
      return {
        ...prev,
        [condition]: {
          ...conditionData,
          [member]: !current,
        },
      };
    });
  }

  function isChecked(condition: string, member: string): boolean {
    return data[condition]?.[member] ?? false;
  }

  async function handleSave() {
    setStatus("saving");
    setErr("");
    try {
      await onSubmit(data);
      setStatus("saved");
    } catch (e: unknown) {
      setStatus("error");
      const error = e as { response?: { data?: { detail?: string } }; message?: string };
      setErr(error?.response?.data?.detail ?? error?.message ?? "Save failed");
    }
  }

  // Count how many conditions are checked
  const checkedCount = Object.values(data).reduce((sum, members) => {
    return sum + Object.values(members).filter(Boolean).length;
  }, 0);

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
          <UsersIcon />
        </div>
        <div style={{ flex: 1 }}>
          <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 2 }}>Family History</h2>
          <p style={{ color: "var(--color-text-secondary)", fontSize: "0.8125rem", margin: 0 }}>
            Check any conditions that apply to family members
          </p>
        </div>
        {checkedCount > 0 && (
          <div style={{
            padding: "6px 12px",
            backgroundColor: "var(--color-bg-tertiary)",
            borderRadius: 20,
            fontSize: "0.75rem",
            fontWeight: 600,
            color: "var(--color-text-secondary)",
          }}>
            {checkedCount} selected
          </div>
        )}
      </div>

      <div
        style={{
          backgroundColor: "#f9fafb",
          borderRadius: 12,
          padding: 20,
          border: "1px solid var(--color-border-light)",
        }}
      >
        <div style={{ overflowX: "auto" }}>
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              fontSize: "0.8125rem",
              minWidth: 700,
            }}
          >
            <thead>
              <tr>
                <th
                  style={{
                    textAlign: "left",
                    padding: "12px 14px",
                    backgroundColor: "#fff",
                    fontWeight: 600,
                    borderBottom: "2px solid var(--color-border)",
                    borderRadius: "8px 0 0 0",
                  }}
                >
                  Condition/Disorder
                </th>
                {FAMILY_MEMBERS.map((member, idx) => (
                  <th
                    key={member.key}
                    style={{
                      textAlign: "center",
                      padding: "12px 10px",
                      backgroundColor: "#fff",
                      fontWeight: 600,
                      borderBottom: "2px solid var(--color-border)",
                      whiteSpace: "nowrap",
                      fontSize: "0.75rem",
                      borderRadius: idx === FAMILY_MEMBERS.length - 1 ? "0 8px 0 0" : undefined,
                    }}
                  >
                    {member.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {CONDITIONS.map((condition, idx) => {
                return (
                  <tr
                    key={condition.key}
                    style={{
                      backgroundColor: idx % 2 === 0 ? "#fff" : "#fafafa",
                      transition: "background-color 0.15s ease",
                    }}
                  >
                    <td
                      style={{
                        padding: "12px 14px",
                        borderBottom: idx === CONDITIONS.length - 1 ? "none" : "1px solid #e5e7eb",
                        fontWeight: 500,
                        color: "var(--color-text)",
                      }}
                    >
                      {condition.label}
                    </td>
                    {FAMILY_MEMBERS.map((member) => {
                      const checked = isChecked(condition.key, member.key);
                      return (
                        <td
                          key={member.key}
                          style={{
                            textAlign: "center",
                            padding: "10px",
                            borderBottom: idx === CONDITIONS.length - 1 ? "none" : "1px solid #e5e7eb",
                          }}
                        >
                          <label style={{
                            display: "inline-flex",
                            alignItems: "center",
                            justifyContent: "center",
                            width: 24,
                            height: 24,
                            borderRadius: 6,
                            border: checked ? "2px solid var(--color-text)" : "1px solid #d1d5db",
                            backgroundColor: checked ? "var(--color-bg-tertiary)" : "#fff",
                            cursor: "pointer",
                            transition: "all 0.15s ease",
                          }}>
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => toggle(condition.key, member.key)}
                              style={{ display: "none" }}
                            />
                            {checked && (
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--color-text)" strokeWidth="3">
                                <polyline points="20 6 9 17 4 12" />
                              </svg>
                            )}
                          </label>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
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
              disabled={status === "saving"}
              className="primary"
              style={{ padding: "10px 20px" }}
            >
              {status === "saving" ? "Saving..." : "Save Family History"}
            </button>
          )}

          {status === "error" && (
            <span style={{ color: "var(--color-error)", fontSize: "0.875rem" }}>{err}</span>
          )}
        </div>

        <span style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)" }}>
          {checkedCount === 0 ? "No conditions selected" : `${checkedCount} condition${checkedCount !== 1 ? 's' : ''} marked`}
        </span>
      </div>
    </form>
  );
}
