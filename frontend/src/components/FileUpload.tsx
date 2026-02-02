import { useRef, useState, useEffect } from "react";

type Props = {
  accept?: string;
  disabled?: boolean;
  onUpload: (file: File) => Promise<void>;
  uploadedFileName?: string;
};

export default function FileUpload({ accept = "video/*", disabled, onUpload, uploadedFileName }: Props) {
  const [status, setStatus] = useState<"idle" | "uploading" | "done" | "error">(uploadedFileName ? "done" : "idle");
  const [message, setMessage] = useState<string>("");
  const [fileName, setFileName] = useState<string>(uploadedFileName ?? "");
  const inputRef = useRef<HTMLInputElement>(null);

  // Sync state when uploadedFileName prop changes (e.g., when navigating back)
  useEffect(() => {
    if (uploadedFileName) {
      setStatus("done");
      setFileName(uploadedFileName);
    }
  }, [uploadedFileName]);

  async function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

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
    } finally {
      // allow uploading the same file again if needed
      e.target.value = "";
    }
  }

  function handleButtonClick() {
    inputRef.current?.click();
  }

  const isDisabled = disabled || status === "uploading";

  return (
    <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        disabled={isDisabled}
        onChange={handleChange}
        style={{ display: "none" }}
      />
      <button
        type="button"
        onClick={handleButtonClick}
        disabled={isDisabled}
        style={{
          padding: "6px 12px",
          borderRadius: 6,
          border: "1px solid #ccc",
          background: isDisabled ? "#f5f5f5" : "#fff",
          cursor: isDisabled ? "not-allowed" : "pointer",
        }}
      >
        {status === "done" ? "Replace file" : "Choose file"}
      </button>
      <span style={{ fontSize: 13, opacity: 0.85 }}>
        {status === "idle" && "No file selected"}
        {status === "uploading" && "Uploading..."}
        {status === "done" && `✅ ${fileName || "Uploaded"}`}
        {status === "error" && `❌ ${message}`}
      </span>
    </div>
  );
}
