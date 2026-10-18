import { useRef, useState } from "react";

export default function DocumentManager({ documents, onUpload, engineReady }) {
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const fileRef = useRef();

  const handleFiles = async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    setUploading(true);
    setUploadMsg("");
    try {
      const result = await onUpload(files);
      setUploadMsg(`✅ Ingested ${result.ingested?.length || files.length} file(s)`);
    } catch (err) {
      setUploadMsg(`⚠️ ${err.message}`);
    } finally {
      setUploading(false);
      fileRef.current.value = "";
    }
  };

  return (
    <div className="doc-manager">
      <input
        ref={fileRef}
        type="file"
        accept=".pdf,.txt,.md"
        multiple
        onChange={handleFiles}
        style={{ display: "none" }}
      />
      <button
        className="btn-upload"
        onClick={() => fileRef.current.click()}
        disabled={!engineReady || uploading}
      >
        {uploading ? "Ingesting…" : "📂 Upload Documents"}
      </button>
      {!engineReady && (
        <p className="upload-helper-msg">
          ⚠️ Please initialize the engine with an OpenAI API Key on the top left to enable document uploads.
        </p>
      )}
      {uploadMsg && <p className="upload-msg">{uploadMsg}</p>}

      {documents.length > 0 && (
        <div className="doc-list">
          {documents.map((d) => (
            <div key={d.doc_name} className="doc-item">
              <span className="doc-icon">📄</span>
              <div className="doc-info">
                <span className="doc-name">{d.doc_name}</span>
                <span className="doc-meta">
                  {d.chunks} chunks · {d.pages} pages · {d.size_kb}KB
                  {d.embed_time_ms && ` · ${d.embed_time_ms}ms`}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {documents.length === 0 && engineReady && (
        <p className="no-docs">No documents indexed yet.</p>
      )}
    </div>
  );
}
