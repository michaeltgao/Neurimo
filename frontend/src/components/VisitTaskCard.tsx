import FileUpload from "./FileUpload";

type Props = {
  title: string;
  instructions: string;
  status: "missing" | "uploaded";
  fileName?: string;
  onUpload: (file: File) => Promise<void>;
};

export default function VisitTaskCard({ title, instructions, status, fileName, onUpload }: Props) {
  const isUploaded = status === "uploaded";

  return (
    <div
      style={{
        border: "1px solid var(--color-border)",
        borderRadius: 10,
        padding: 16,
        backgroundColor: "#fff",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
        <div>
          <div style={{ fontWeight: 600, fontSize: "0.875rem", marginBottom: 2 }}>{title}</div>
          <div style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)", lineHeight: 1.4 }}>
            {instructions}
          </div>
        </div>
        <div
          style={{
            fontSize: "0.6875rem",
            fontWeight: 600,
            padding: "4px 10px",
            borderRadius: 20,
            backgroundColor: "var(--color-bg-tertiary)",
            color: "var(--color-text-secondary)",
            textTransform: "uppercase",
            letterSpacing: "0.025em",
            flexShrink: 0,
          }}
        >
          {isUploaded ? "Complete" : "Pending"}
        </div>
      </div>

      {/* Upload area */}
      <div style={{ marginTop: 12 }}>
        <FileUpload onUpload={onUpload} uploadedFileName={fileName} />
      </div>
    </div>
  );
}