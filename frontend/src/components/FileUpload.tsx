import { useRef, useState, useEffect } from "react";

type Props = {
  accept?: string;
  disabled?: boolean;
  onUpload: (file: File) => Promise<void>;
  uploadedFileName?: string;
};

const UploadIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" />
    <line x1="12" y1="3" x2="12" y2="15" />
  </svg>
);

const FileVideoIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <polygon points="10 11 10 17 15 14 10 11" />
  </svg>
);

const RefreshIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="23 4 23 10 17 10" />
    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
  </svg>
);

export default function FileUpload({ accept = "video/*", disabled, onUpload, uploadedFileName }: Props) {
  const [status, setStatus] = useState<"idle" | "uploading" | "done" | "error">(uploadedFileName ? "done" : "idle");
  const [message, setMessage] = useState<string>("");
  const [fileName, setFileName] = useState<string>(uploadedFileName ?? "");
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Sync state when uploadedFileName prop changes (e.g., when navigating back)
  useEffect(() => {
    if (uploadedFileName) {
      setStatus("done");
      setFileName(uploadedFileName);
    }
  }, [uploadedFileName]);

  async function handleFile(file: File) {
    setStatus("uploading");
    setMessage("");
    setFileName(file.name);

    try {
      await onUpload(file);
      setStatus("done");
      setMessage("Uploaded");
    } catch (e: unknown) {
      setStatus("error");
      setFileName("");
      const err = e as { response?: { data?: { detail?: string } }; message?: string };
      setMessage(err?.response?.data?.detail ?? err?.message ?? "Upload failed");
    }
  }

  async function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    await handleFile(file);
    // allow uploading the same file again if needed
    e.target.value = "";
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled || status === "uploading") return;
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    if (disabled || status === "uploading") return;
    setIsDragOver(true);
  }

  function handleDragLeave() {
    setIsDragOver(false);
  }

  function handleButtonClick() {
    inputRef.current?.click();
  }

  const isDisabled = disabled || status === "uploading";
  const isDone = status === "done";

  // Show compact view when uploaded
  if (isDone) {
    return (
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "10px 12px",
        backgroundColor: "#fff",
        borderRadius: 8,
        border: "1px solid var(--color-border)",
        boxShadow: "0 1px 3px rgba(0, 0, 0, 0.08)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
          <FileVideoIcon />
          <span style={{
            fontSize: "0.8125rem",
            color: "var(--color-text-secondary)",
            fontWeight: 500,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}>
            {fileName || "Uploaded"}
          </span>
        </div>
        <button
          type="button"
          onClick={handleButtonClick}
          disabled={isDisabled}
          style={{
            padding: "4px 10px",
            borderRadius: 6,
            border: "1px solid var(--color-border)",
            background: "var(--color-bg-secondary)",
            cursor: isDisabled ? "not-allowed" : "pointer",
            fontSize: "0.75rem",
            fontWeight: 500,
            color: "var(--color-text-secondary)",
            display: "flex",
            alignItems: "center",
            gap: 4,
          }}
        >
          <RefreshIcon />
          Replace
        </button>
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          disabled={isDisabled}
          onChange={handleChange}
          style={{ display: "none" }}
        />
      </div>
    );
  }

  return (
    <div
      onClick={isDisabled ? undefined : handleButtonClick}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      style={{
        padding: "20px 16px",
        borderRadius: 8,
        border: `2px dashed ${isDragOver ? "var(--color-accent)" : status === "error" ? "#fca5a5" : "var(--color-border)"}`,
        backgroundColor: isDragOver ? "#eff6ff" : status === "error" ? "#fef2f2" : "var(--color-bg-secondary)",
        cursor: isDisabled ? "not-allowed" : "pointer",
        transition: "all 0.15s ease",
        textAlign: "center",
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        disabled={isDisabled}
        onChange={handleChange}
        style={{ display: "none" }}
      />

      {status === "uploading" ? (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
          <div style={{
            width: 24,
            height: 24,
            border: "2px solid var(--color-bg-tertiary)",
            borderTopColor: "var(--color-accent)",
            borderRadius: "50%",
            animation: "spin 0.8s linear infinite",
          }} />
          <span style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)" }}>
            Uploading {fileName}...
          </span>
        </div>
      ) : status === "error" ? (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: "0.875rem", color: "var(--color-error)" }}>
            {message}
          </span>
          <span style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)" }}>
            Click to try again
          </span>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
          <div style={{ color: "var(--color-text-tertiary)" }}>
            <UploadIcon />
          </div>
          <div>
            <span style={{ fontSize: "0.8125rem", color: "var(--color-text-secondary)" }}>
              Drag & drop or{" "}
            </span>
            <span style={{ fontSize: "0.8125rem", color: "var(--color-accent)", fontWeight: 500 }}>
              browse
            </span>
          </div>
          <span style={{ fontSize: "0.75rem", color: "var(--color-text-tertiary)" }}>
            MP4, MOV, or WebM (max 100MB)
          </span>
        </div>
      )}
    </div>
  );
}
