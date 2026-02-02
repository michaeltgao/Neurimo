import FileUpload from "./FileUpload";

type Props = {
  title: string;
  instructions: string;
  status: "missing" | "uploaded";
  fileName?: string;
  onUpload: (file: File) => Promise<void>;
};

export default function VisitTaskCard({ title, instructions, status, fileName, onUpload }: Props) {
  return (
    <div
      style={{
        border: "1px solid var(--color-border)",
        borderRadius: 8,
        padding: 16,
        backgroundColor: status === "uploaded" ? "var(--color-bg-secondary)" : "#fff",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16 }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 500, fontSize: "0.875rem" }}>{title}</div>
          <div style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)", marginTop: 4 }}>
            {instructions}
          </div>
        </div>
        <div
          style={{
            fontSize: "0.75rem",
            fontWeight: 500,
            padding: "4px 8px",
            borderRadius: 4,
            backgroundColor: status === "uploaded" ? "#dcfce7" : "var(--color-bg-tertiary)",
            color: status === "uploaded" ? "#166534" : "var(--color-text-secondary)",
          }}
        >
          {status === "uploaded" ? "Uploaded" : "Pending"}
        </div>
      </div>

      <div style={{ marginTop: 12 }}>
        <FileUpload onUpload={onUpload} uploadedFileName={fileName} />
      </div>
    </div>
  );
}
