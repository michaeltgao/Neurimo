import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { createVisit, listVisits, deleteVisit, updateVisit, type Visit } from "../api/visits";
import { getChild, updateChild, type Child } from "../api/children";
import { useAuth } from "../context/AuthContext";
import logo from "../assets/logo.png";

// Icons
const CalendarPlusIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
    <line x1="16" y1="2" x2="16" y2="6" />
    <line x1="8" y1="2" x2="8" y2="6" />
    <line x1="3" y1="10" x2="21" y2="10" />
    <line x1="12" y1="14" x2="12" y2="18" />
    <line x1="10" y1="16" x2="14" y2="16" />
  </svg>
);

const ClipboardListIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" />
    <rect x="8" y="2" width="8" height="4" rx="1" ry="1" />
    <path d="M9 12h6" />
    <path d="M9 16h6" />
  </svg>
);

const CalendarIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
    <line x1="16" y1="2" x2="16" y2="6" />
    <line x1="8" y1="2" x2="8" y2="6" />
    <line x1="3" y1="10" x2="21" y2="10" />
  </svg>
);

const ClockIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <polyline points="12 6 12 12 16 14" />
  </svg>
);

const ArrowLeftIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="19" y1="12" x2="5" y2="12" />
    <polyline points="12 19 5 12 12 5" />
  </svg>
);

const UserIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
    <circle cx="12" cy="7" r="4" />
  </svg>
);

const FileTextIcon = () => (
  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
  </svg>
);

const ChevronRightIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="9 18 15 12 9 6" />
  </svg>
);

const PencilIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z" />
  </svg>
);

// Card wrapper style
const cardStyle: React.CSSProperties = {
  backgroundColor: "#fff",
  borderRadius: 12,
  border: "1px solid var(--color-border)",
  boxShadow: "0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.06)",
  padding: 24,
};

// Section header with icon
const SectionHeader = ({ icon, title, subtitle }: { icon: React.ReactNode; title: string; subtitle?: string }) => (
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
      {icon}
    </div>
    <div>
      <h2 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: 2 }}>{title}</h2>
      {subtitle && (
        <p style={{ color: "var(--color-text-secondary)", fontSize: "0.8125rem", margin: 0 }}>
          {subtitle}
        </p>
      )}
    </div>
  </div>
);

// Form field wrapper with icon
const FormField = ({
  label,
  required,
  icon,
  children
}: {
  label: string;
  required?: boolean;
  icon: React.ReactNode;
  children: React.ReactNode;
}) => (
  <div style={{ display: "grid", gap: 6 }}>
    <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span style={{ color: "var(--color-text-secondary)" }}>{icon}</span>
      {label}
      {required && <span style={{ color: "var(--color-error)", marginLeft: 2 }}>*</span>}
    </label>
    {children}
  </div>
);

// Visit number badge
const VisitBadge = ({ number }: { number: number }) => (
  <div style={{
    width: 40,
    height: 40,
    borderRadius: 10,
    backgroundColor: "var(--color-bg-tertiary)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: 700,
    fontSize: "1rem",
    color: "var(--color-text)",
    flexShrink: 0,
  }}>
    {number}
  </div>
);

export default function ChildDetailPage() {
  const { childId } = useParams();
  const childIdNum = Number(childId);
  const nav = useNavigate();
  const { logout } = useAuth();

  function onSignOut() {
    logout();
    nav("/signin");
  }

  const [child, setChild] = useState<Child | null>(null);
  const [visits, setVisits] = useState<Visit[]>([]);
  const [err, setErr] = useState<string | null>(null);

  // edit child form
  const [editingChild, setEditingChild] = useState(false);
  const [editPseudoId, setEditPseudoId] = useState("");
  const [editBirthdate, setEditBirthdate] = useState("");

  // edit visit
  const [editingVisitId, setEditingVisitId] = useState<string | null>(null);
  const [editVisitDate, setEditVisitDate] = useState("");
  const [editAgeMonths, setEditAgeMonths] = useState<number>(12);

  // create visit form
  const [visitDate, setVisitDate] = useState("");
  const [ageMonths, setAgeMonths] = useState<number>(12);

  const validChildId = useMemo(() => Number.isFinite(childIdNum) && childIdNum > 0, [childIdNum]);

  useEffect(() => {
    if (!validChildId) return;

    let cancelled = false;

    Promise.all([getChild(childIdNum), listVisits(childIdNum)])
      .then(([found, v]) => {
        if (cancelled) return;
        setErr(null);
        setChild(found);
        setVisits(v);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        const err = e as { response?: { data?: { detail?: string } }; message?: string };
        setErr(err?.response?.data?.detail ?? err?.message ?? "Failed to load child");
      });

    return () => {
      cancelled = true;
    };
  }, [childIdNum, validChildId]);

  async function onCreateVisit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);

    if (!visitDate) {
      setErr("visit_date is required");
      return;
    }

    try {
      const created = await createVisit(childIdNum, { visit_date: visitDate, age_months: ageMonths });
      setVisitDate("");
      setAgeMonths(12);
      // go to visit page
      nav(`/visits/${created.child_id}-${created.visit_number}`);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } }; message?: string };
      setErr(err?.response?.data?.detail ?? err?.message ?? "Create visit failed");
    }
  }

  async function onDeleteVisit(visit: Visit) {
    if (!confirm(`Are you sure you want to delete Visit ${visit.visit_number}?`)) {
      return;
    }

    setErr(null);
    try {
      await deleteVisit(`${visit.child_id}-${visit.visit_number}`);
      setVisits((prev) => prev.filter((v) => v.id !== visit.id));
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } }; message?: string };
      setErr(err?.response?.data?.detail ?? err?.message ?? "Delete visit failed");
    }
  }

  function startEditingChild() {
    if (!child) return;
    setEditPseudoId(child.pseudo_id);
    setEditBirthdate(child.birthdate);
    setEditingChild(true);
  }

  async function onSaveChild() {
    setErr(null);
    try {
      const updated = await updateChild(childIdNum, {
        pseudo_id: editPseudoId,
        birthdate: editBirthdate,
      });
      setChild(updated);
      setEditingChild(false);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } }; message?: string };
      setErr(err?.response?.data?.detail ?? err?.message ?? "Update child failed");
    }
  }

  function startEditingVisit(visit: Visit) {
    setEditingVisitId(`${visit.child_id}-${visit.visit_number}`);
    setEditVisitDate(visit.visit_date);
    setEditAgeMonths(visit.age_months);
  }

  async function onSaveVisit(visitId: string) {
    setErr(null);
    try {
      const updated = await updateVisit(visitId, {
        visit_date: editVisitDate,
        age_months: editAgeMonths,
      });
      setVisits((prev) => prev.map((v) =>
        `${v.child_id}-${v.visit_number}` === visitId ? updated : v
      ));
      setEditingVisitId(null);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } }; message?: string };
      setErr(err?.response?.data?.detail ?? err?.message ?? "Update visit failed");
    }
  }

  if (!validChildId) {
    return (
      <div style={{ padding: 24, color: "var(--color-text-secondary)", fontSize: "0.875rem" }}>
        Invalid patient ID
      </div>
    );
  }

  return (
    <div style={{ minHeight: "100vh", backgroundColor: "var(--color-bg-secondary)" }}>
      {/* Header */}
      <header
        style={{
          borderBottom: "1px solid var(--color-border)",
          padding: "12px 24px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          backgroundColor: "#fff",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <button
            onClick={() => nav("/children")}
            style={{
              fontSize: "0.8125rem",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <ArrowLeftIcon />
            Back
          </button>
          <img src={logo} alt="Neurimo" style={{ height: 40 }} />
        </div>
        <button onClick={onSignOut} style={{ fontSize: "0.8125rem" }}>
          Sign Out
        </button>
      </header>

      {/* Main content */}
      <main style={{ maxWidth: 1000, margin: "0 auto", padding: "32px 24px" }}>
        {/* Patient info card */}
        {child ? (
          <div style={{
            ...cardStyle,
            marginBottom: 24,
          }}>
            {editingChild ? (
              <div style={{ display: "grid", gap: 16 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div style={{
                    padding: 12,
                    backgroundColor: "var(--color-bg-tertiary)",
                    borderRadius: 12,
                    color: "var(--color-text-secondary)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}>
                    <UserIcon />
                  </div>
                  <h2 style={{ fontSize: "1rem", fontWeight: 600 }}>Edit Patient</h2>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                  <div style={{ display: "grid", gap: 6 }}>
                    <label style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)" }}>Patient ID</label>
                    <input
                      type="text"
                      value={editPseudoId}
                      onChange={(e) => setEditPseudoId(e.target.value)}
                    />
                  </div>
                  <div style={{ display: "grid", gap: 6 }}>
                    <label style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)" }}>Birthdate</label>
                    <input
                      type="date"
                      value={editBirthdate}
                      onChange={(e) => setEditBirthdate(e.target.value)}
                    />
                  </div>
                </div>
                <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
                  <button
                    onClick={() => setEditingChild(false)}
                    style={{ padding: "8px 16px", fontSize: "0.8125rem" }}
                  >
                    Cancel
                  </button>
                  <button
                    onClick={onSaveChild}
                    className="primary"
                    style={{ padding: "8px 16px", fontSize: "0.8125rem" }}
                  >
                    Save
                  </button>
                </div>
              </div>
            ) : (
              <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                <div style={{
                  padding: 12,
                  backgroundColor: "var(--color-bg-tertiary)",
                  borderRadius: 12,
                  color: "var(--color-text-secondary)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}>
                  <UserIcon />
                </div>
                <div style={{ flex: 1 }}>
                  <h1 style={{ fontSize: "1.5rem", marginBottom: 4 }}>{child.pseudo_id}</h1>
                  <div style={{
                    color: "var(--color-text-secondary)",
                    fontSize: "0.875rem",
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                  }}>
                    <span>Born {child.birthdate}</span>
                    <span style={{
                      padding: "2px 10px",
                      backgroundColor: "var(--color-bg-tertiary)",
                      borderRadius: 4,
                      fontSize: "0.75rem",
                      fontWeight: 500,
                    }}>
                      {child.sex === "M" ? "Male" : child.sex === "F" ? "Female" : child.sex}
                    </span>
                  </div>
                </div>
                <button
                  onClick={startEditingChild}
                  style={{
                    padding: 6,
                    color: "var(--color-text-secondary)",
                    backgroundColor: "transparent",
                    border: "1px solid var(--color-border)",
                    borderRadius: 6,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                  title="Edit patient"
                >
                  <PencilIcon />
                </button>
                <div style={{
                  textAlign: "right",
                  color: "var(--color-text-secondary)",
                  fontSize: "0.8125rem",
                }}>
                  <div style={{ fontWeight: 500, color: "var(--color-text)" }}>{visits.length}</div>
                  <div>visit{visits.length !== 1 ? 's' : ''}</div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div style={{
            ...cardStyle,
            marginBottom: 24,
            animation: "pulse 1.5s ease-in-out infinite",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <div style={{ width: 44, height: 44, borderRadius: 12, backgroundColor: "var(--color-bg-tertiary)" }} />
              <div style={{ flex: 1 }}>
                <div style={{ height: 24, width: "40%", backgroundColor: "var(--color-bg-tertiary)", borderRadius: 4, marginBottom: 8 }} />
                <div style={{ height: 14, width: "30%", backgroundColor: "var(--color-bg-tertiary)", borderRadius: 4 }} />
              </div>
            </div>
          </div>
        )}

        {err && (
          <div
            style={{
              padding: "12px 16px",
              marginBottom: 24,
              backgroundColor: "#fef2f2",
              border: "1px solid #fecaca",
              borderRadius: 8,
              color: "var(--color-error)",
              fontSize: "0.875rem",
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            {err}
          </div>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
          {/* Create visit card */}
          <div style={cardStyle}>
            <SectionHeader
              icon={<CalendarPlusIcon />}
              title="Schedule New Visit"
              subtitle="Record a new clinical visit for this patient"
            />

            <form onSubmit={onCreateVisit} style={{ display: "grid", gap: 20 }}>
              <FormField label="Visit Date" required icon={<CalendarIcon />}>
                <input
                  id="visitDate"
                  type="date"
                  value={visitDate}
                  onChange={(e) => setVisitDate(e.target.value)}
                />
              </FormField>

              <FormField label="Age at Visit (months)" required icon={<ClockIcon />}>
                <input
                  id="ageMonths"
                  type="number"
                  value={ageMonths}
                  min={6}
                  max={60}
                  onChange={(e) => setAgeMonths(Number(e.target.value))}
                />
              </FormField>

              <button type="submit" className="primary" style={{ marginTop: 4, padding: "10px 16px" }}>
                Create Visit
              </button>
            </form>
          </div>

          {/* Visits list card */}
          <div style={cardStyle}>
            <SectionHeader
              icon={<ClipboardListIcon />}
              title="Visit History"
              subtitle={`${visits.length} visit${visits.length !== 1 ? 's' : ''} recorded`}
            />

            {!child ? (
              <div style={{ display: "grid", gap: 12 }}>
                {[1, 2].map((i) => (
                  <div
                    key={i}
                    style={{
                      padding: 16,
                      borderRadius: 10,
                      backgroundColor: "var(--color-bg-secondary)",
                      animation: "pulse 1.5s ease-in-out infinite",
                    }}
                  >
                    <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                      <div style={{ width: 40, height: 40, borderRadius: 10, backgroundColor: "var(--color-bg-tertiary)" }} />
                      <div style={{ flex: 1 }}>
                        <div style={{ height: 14, width: "50%", backgroundColor: "var(--color-bg-tertiary)", borderRadius: 4, marginBottom: 8 }} />
                        <div style={{ height: 10, width: "35%", backgroundColor: "var(--color-bg-tertiary)", borderRadius: 4 }} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : visits.length === 0 ? (
              <div style={{
                textAlign: "center",
                padding: "40px 20px",
                color: "var(--color-text-tertiary)",
              }}>
                <div style={{ marginBottom: 16, opacity: 0.5 }}>
                  <FileTextIcon />
                </div>
                <p style={{ fontSize: "0.9375rem", fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 4 }}>
                  No visits recorded
                </p>
                <p style={{ fontSize: "0.8125rem", margin: 0 }}>
                  Schedule the first visit using the form
                </p>
              </div>
            ) : (
              <div style={{ display: "grid", gap: 10 }}>
                {visits.map((v) => {
                  const visitId = `${v.child_id}-${v.visit_number}`;
                  const isEditing = editingVisitId === visitId;

                  if (isEditing) {
                    return (
                      <div
                        key={v.id}
                        style={{
                          padding: 14,
                          border: "1px solid var(--color-border)",
                          borderRadius: 10,
                          backgroundColor: "#fff",
                        }}
                      >
                        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
                          <VisitBadge number={v.visit_number} />
                          <div style={{ fontWeight: 600, fontSize: "0.875rem" }}>
                            Edit Visit {v.visit_number}
                          </div>
                        </div>
                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
                          <div style={{ display: "grid", gap: 4 }}>
                            <label style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)" }}>Visit Date</label>
                            <input
                              type="date"
                              value={editVisitDate}
                              onChange={(e) => setEditVisitDate(e.target.value)}
                              style={{ padding: "6px 10px", fontSize: "0.8125rem" }}
                            />
                          </div>
                          <div style={{ display: "grid", gap: 4 }}>
                            <label style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)" }}>Age (months)</label>
                            <input
                              type="number"
                              value={editAgeMonths}
                              onChange={(e) => setEditAgeMonths(Number(e.target.value))}
                              min={6}
                              max={60}
                              style={{ padding: "6px 10px", fontSize: "0.8125rem" }}
                            />
                          </div>
                        </div>
                        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
                          <button
                            onClick={() => setEditingVisitId(null)}
                            style={{ padding: "6px 12px", fontSize: "0.75rem" }}
                          >
                            Cancel
                          </button>
                          <button
                            onClick={() => onSaveVisit(visitId)}
                            className="primary"
                            style={{ padding: "6px 12px", fontSize: "0.75rem" }}
                          >
                            Save
                          </button>
                        </div>
                      </div>
                    );
                  }

                  return (
                    <div
                      key={v.id}
                      onClick={() => nav(`/visits/${v.child_id}-${v.visit_number}/report`)}
                      style={{
                        padding: 14,
                        border: "1px solid var(--color-border)",
                        borderRadius: 10,
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        gap: 12,
                        transition: "all 0.15s ease",
                        backgroundColor: "#fff",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = "var(--color-bg-secondary)";
                        e.currentTarget.style.borderColor = "var(--color-text-tertiary)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = "#fff";
                        e.currentTarget.style.borderColor = "var(--color-border)";
                      }}
                    >
                      <VisitBadge number={v.visit_number} />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontWeight: 600, fontSize: "0.875rem", marginBottom: 2 }}>
                          Visit {v.visit_number}
                        </div>
                        <div style={{
                          fontSize: "0.75rem",
                          color: "var(--color-text-secondary)",
                          display: "flex",
                          alignItems: "center",
                          gap: 8,
                        }}>
                          <span>Date: {v.visit_date}</span>
                          <span style={{
                            padding: "2px 8px",
                            backgroundColor: "var(--color-bg-tertiary)",
                            borderRadius: 4,
                            fontSize: "0.6875rem",
                            fontWeight: 500,
                          }}>
                            {v.age_months} months
                          </span>
                        </div>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            startEditingVisit(v);
                          }}
                          style={{
                            padding: 6,
                            color: "var(--color-text-secondary)",
                            backgroundColor: "transparent",
                            border: "1px solid var(--color-border)",
                            borderRadius: 6,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                          }}
                          title="Edit visit"
                        >
                          <PencilIcon />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteVisit(v);
                          }}
                          style={{
                            padding: "6px 10px",
                            fontSize: "0.75rem",
                            color: "var(--color-text-secondary)",
                            backgroundColor: "transparent",
                            border: "1px solid var(--color-border)",
                          }}
                        >
                          Delete
                        </button>
                        <ChevronRightIcon />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
