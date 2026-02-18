import { useEffect, useState } from "react";
import { createChild, listChildren, deleteChild, type Child } from "../api/children";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import logo from "../assets/logo.png";

// Icons as inline SVG components for professional clinical look
const UserPlusIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <line x1="19" y1="8" x2="19" y2="14" />
    <line x1="22" y1="11" x2="16" y2="11" />
  </svg>
);

const UsersIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
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

const UserIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
    <circle cx="12" cy="7" r="4" />
  </svg>
);

const BuildingIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="4" y="2" width="16" height="20" rx="2" ry="2" />
    <path d="M9 22v-4h6v4" />
    <path d="M8 6h.01" />
    <path d="M16 6h.01" />
    <path d="M12 6h.01" />
    <path d="M12 10h.01" />
    <path d="M12 14h.01" />
    <path d="M16 10h.01" />
    <path d="M16 14h.01" />
    <path d="M8 10h.01" />
    <path d="M8 14h.01" />
  </svg>
);

const ClipboardIcon = () => (
  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" />
    <rect x="8" y="2" width="8" height="4" rx="1" ry="1" />
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

// Patient avatar with initials
const PatientAvatar = ({ name }: { name: string }) => {
  const initials = name.slice(0, 2).toUpperCase();
  return (
    <div style={{
      width: 40,
      height: 40,
      borderRadius: 10,
      backgroundColor: "var(--color-bg-tertiary)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontWeight: 600,
      fontSize: "0.875rem",
      color: "var(--color-text-secondary)",
      flexShrink: 0,
    }}>
      {initials}
    </div>
  );
};

export default function ChildrenListPage() {
  const [children, setChildren] = useState<Child[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  // form state
  const [pseudoId, setPseudoId] = useState("");
  const [birthdate, setBirthdate] = useState("");
  const [sex, setSex] = useState("M");
  const [clinicId, setClinicId] = useState("clinic-001");

  const nav = useNavigate();
  const { logout } = useAuth();

  function onSignOut() {
    logout();
    nav("/signin");
  }

  async function refresh() {
    setLoading(true);
    setErr(null);
    try {
      const data = await listChildren();
      setChildren(data);
    } catch (e: unknown) {
      const err = e as { message?: string };
      setErr(err?.message ?? "Failed to load children");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function onCreateChild(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);

    if (!pseudoId || !birthdate) {
      setErr("pseudo_id and birthdate are required");
      return;
    }

    try {
      const newChild = await createChild({
        pseudo_id: pseudoId,
        birthdate,
        sex,
        clinic_id: clinicId,
      });
      setPseudoId("");
      setBirthdate("");
      setSex("M");
      setClinicId("clinic-001");
      // go to child detail
      nav(`/children/${newChild.id}`);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } }; message?: string };
      setErr(err?.response?.data?.detail ?? err?.message ?? "Create failed");
    }
  }

  async function onDeleteChild(childId: number, pseudoId: string) {
    if (!confirm(`Are you sure you want to delete child "${pseudoId}"?`)) {
      return;
    }

    setErr(null);
    try {
      await deleteChild(childId);
      await refresh();
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } }; message?: string };
      setErr(err?.response?.data?.detail ?? err?.message ?? "Delete failed");
    }
  }

  return (
    <div style={{ minHeight: "100vh", backgroundColor: "#fff" }}>
      {/* Header */}
      <header
        style={{
          borderBottom: "1px solid var(--color-border)",
          padding: "12px 24px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <img src={logo} alt="Neurimo" style={{ height: 40 }} />
        <button onClick={onSignOut} style={{ fontSize: "0.8125rem" }}>
          Sign Out
        </button>
      </header>

      {/* Main content */}
      <main style={{ maxWidth: 1000, margin: "0 auto", padding: "32px 24px" }}>
        {/* Page header */}
        <div style={{ marginBottom: 32 }}>
          <h1 style={{ fontSize: "1.75rem", marginBottom: 4 }}>Patient Management</h1>
          <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem", margin: 0 }}>
            Create and manage patient records for developmental assessments
          </p>
        </div>

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

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>
          {/* Create form card */}
          <div style={cardStyle}>
            <SectionHeader
              icon={<UserPlusIcon />}
              title="New Patient"
              subtitle="Enter patient details to create a new record"
            />

            <form onSubmit={onCreateChild} style={{ display: "grid", gap: 20 }}>
              <FormField label="Patient ID" required icon={<UserIcon />}>
                <input
                  id="pseudoId"
                  value={pseudoId}
                  onChange={(e) => setPseudoId(e.target.value)}
                  placeholder="e.g., CH-0001"
                />
              </FormField>

              <FormField label="Date of Birth" required icon={<CalendarIcon />}>
                <input
                  id="birthdate"
                  type="date"
                  value={birthdate}
                  onChange={(e) => setBirthdate(e.target.value)}
                />
              </FormField>

              <FormField label="Sex" icon={<UserIcon />}>
                <select id="sex" value={sex} onChange={(e) => setSex(e.target.value)}>
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                  <option value="Other">Other</option>
                </select>
              </FormField>

              <FormField label="Clinic ID" icon={<BuildingIcon />}>
                <input
                  id="clinicId"
                  value={clinicId}
                  onChange={(e) => setClinicId(e.target.value)}
                  placeholder="Enter clinic identifier"
                />
              </FormField>

              <button type="submit" className="primary" style={{ marginTop: 4, padding: "10px 16px" }}>
                Create Patient
              </button>
            </form>
          </div>

          {/* Patient list card */}
          <div style={cardStyle}>
            <SectionHeader
              icon={<UsersIcon />}
              title="Patient Records"
              subtitle={loading ? "Loading..." : `${children.length} patient${children.length !== 1 ? 's' : ''} registered`}
            />

            {loading ? (
              <div style={{ display: "grid", gap: 12 }}>
                {[1, 2, 3].map((i) => (
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
                        <div style={{ height: 14, width: "60%", backgroundColor: "var(--color-bg-tertiary)", borderRadius: 4, marginBottom: 8 }} />
                        <div style={{ height: 10, width: "40%", backgroundColor: "var(--color-bg-tertiary)", borderRadius: 4 }} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : children.length === 0 ? (
              <div style={{
                textAlign: "center",
                padding: "40px 20px",
                color: "var(--color-text-tertiary)",
              }}>
                <div style={{ marginBottom: 16, opacity: 0.5 }}>
                  <ClipboardIcon />
                </div>
                <p style={{ fontSize: "0.9375rem", fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 4 }}>
                  No patients yet
                </p>
                <p style={{ fontSize: "0.8125rem", margin: 0 }}>
                  Create your first patient using the form
                </p>
              </div>
            ) : (
              <div style={{ display: "grid", gap: 10 }}>
                {children.map((c) => (
                  <div
                    key={c.id}
                    onClick={() => nav(`/children/${c.id}`)}
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
                    <PatientAvatar name={c.pseudo_id} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontWeight: 600, fontSize: "0.875rem", marginBottom: 2 }}>
                        {c.pseudo_id}
                      </div>
                      <div style={{
                        fontSize: "0.75rem",
                        color: "var(--color-text-secondary)",
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                      }}>
                        <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                          <CalendarIcon />
                          {c.birthdate}
                        </span>
                        <span style={{
                          padding: "2px 8px",
                          backgroundColor: "var(--color-bg-tertiary)",
                          borderRadius: 4,
                          fontSize: "0.6875rem",
                          fontWeight: 500,
                        }}>
                          {c.sex === 'M' ? 'Male' : c.sex === 'F' ? 'Female' : 'Other'}
                        </span>
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteChild(c.id, c.pseudo_id);
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
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
