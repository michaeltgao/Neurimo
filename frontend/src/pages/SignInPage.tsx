import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function SignInPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);

  const { login } = useAuth();
  const nav = useNavigate();

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);

    if (!email || !password) {
      setErr("Email and password are required");
      return;
    }

    const success = login(password);
    if (success) {
      nav("/children");
    } else {
      setErr("Invalid credentials");
    }
  }

  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#fff",
      }}
    >
      <div style={{ width: "100%", maxWidth: 320, padding: 24 }}>
        <div style={{ marginBottom: 40, textAlign: "center" }}>
          <h1 style={{ marginBottom: 4, fontSize: "1.75rem", fontWeight: 600, letterSpacing: "-0.02em" }}>
            Neurimo
          </h1>
          <p style={{ margin: 0, color: "var(--color-text-secondary)", fontSize: "0.8125rem", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Clinician Web Portal
          </p>
          <div style={{ marginTop: 24, paddingTop: 24, borderTop: "1px solid var(--color-border)" }}>
            <p style={{ margin: 0, color: "var(--color-text-secondary)", fontSize: "0.875rem" }}>
              Sign in to access your account
            </p>
          </div>
        </div>

        <form onSubmit={onSubmit} style={{ display: "grid", gap: 16 }}>
          <div style={{ display: "grid", gap: 6 }}>
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
            />
          </div>

          <div style={{ display: "grid", gap: 6 }}>
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
            />
          </div>

          {err && (
            <div
              style={{
                padding: "10px 12px",
                backgroundColor: "#fef2f2",
                border: "1px solid #fecaca",
                borderRadius: 6,
                color: "var(--color-error)",
                fontSize: "0.8125rem",
              }}
            >
              {err}
            </div>
          )}

          <button
            type="submit"
            className="primary"
            style={{ marginTop: 8, fontWeight: 500 }}
          >
            Sign In
          </button>
        </form>
      </div>
    </div>
  );
}
