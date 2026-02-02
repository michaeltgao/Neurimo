import { useEffect, useState } from "react";
import { createChild, listChildren, deleteChild, type Child } from "../api/children";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function ChildrenListPage() {
  const [children, setChildren] = useState<Child[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  // form state
  const [pseudoId, setPseudoId] = useState("");
  const [birthdate, setBirthdate] = useState("");
  const [sex, setSex] = useState("M");
  const [clinicId, setClinicId] = useState("default");

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
      setClinicId("default");
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
        <h1 style={{ fontSize: "1rem", fontWeight: 600 }}>Neurimo</h1>
        <button onClick={onSignOut} style={{ fontSize: "0.8125rem" }}>
          Sign Out
        </button>
      </header>

      {/* Main content */}
      <main style={{ maxWidth: 960, margin: "0 auto", padding: "32px 24px" }}>
        <div style={{ marginBottom: 32 }}>
          <h1>Children</h1>
          <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem", marginTop: 4 }}>
            Manage patient records and create new visits
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
            }}
          >
            {err}
          </div>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>
          {/* Create form */}
          <div>
            <h2 style={{ marginBottom: 16 }}>New patient</h2>
            <form onSubmit={onCreateChild} style={{ display: "grid", gap: 16 }}>
              <div style={{ display: "grid", gap: 6 }}>
                <label htmlFor="pseudoId">Patient ID</label>
                <input
                  id="pseudoId"
                  value={pseudoId}
                  onChange={(e) => setPseudoId(e.target.value)}
                  placeholder="e.g., CH-0001"
                />
              </div>

              <div style={{ display: "grid", gap: 6 }}>
                <label htmlFor="birthdate">Date of birth</label>
                <input
                  id="birthdate"
                  type="date"
                  value={birthdate}
                  onChange={(e) => setBirthdate(e.target.value)}
                />
              </div>

              <div style={{ display: "grid", gap: 6 }}>
                <label htmlFor="sex">Sex</label>
                <select id="sex" value={sex} onChange={(e) => setSex(e.target.value)}>
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>

              <div style={{ display: "grid", gap: 6 }}>
                <label htmlFor="clinicId">Clinic ID</label>
                <input
                  id="clinicId"
                  value={clinicId}
                  onChange={(e) => setClinicId(e.target.value)}
                />
              </div>

              <button type="submit" className="primary" style={{ marginTop: 8 }}>
                Create patient
              </button>
            </form>
          </div>

          {/* List */}
          <div>
            <h2 style={{ marginBottom: 16 }}>Patients</h2>
            {loading ? (
              <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem" }}>Loading...</p>
            ) : children.length === 0 ? (
              <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem" }}>No patients yet.</p>
            ) : (
              <div style={{ display: "grid", gap: 8 }}>
                {children.map((c) => (
                  <div
                    key={c.id}
                    onClick={() => nav(`/children/${c.id}`)}
                    style={{
                      padding: "12px 16px",
                      border: "1px solid var(--color-border)",
                      borderRadius: 8,
                      cursor: "pointer",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      transition: "background-color 0.15s ease",
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "var(--color-bg-secondary)")}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                  >
                    <div>
                      <div style={{ fontWeight: 500, fontSize: "0.875rem" }}>{c.pseudo_id}</div>
                      <div style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)", marginTop: 2 }}>
                        {c.birthdate} · {c.sex}
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteChild(c.id, c.pseudo_id);
                      }}
                      style={{
                        padding: "4px 8px",
                        fontSize: "0.75rem",
                        color: "var(--color-text-secondary)",
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
