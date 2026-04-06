import { useState } from "react";
import axios from "axios";
import { X } from "lucide-react";

const API = axios.create({ baseURL: "http://localhost:8080" });

export default function TestPredictModal({ open, onClose, port }) {
  const [featuresStr, setFeaturesStr] = useState("");
  const [out, setOut] = useState(null);
  const [busy, setBusy] = useState(false);

  if (!open) return null;

  const run = async () => {
    const parts = featuresStr.split(",").map((s) => s.trim()).filter(Boolean);
    const features = parts.map((x) => parseFloat(x));
    setBusy(true);
    setOut(null);
    try {
      const { data } = await API.post(`/api/test-predict/${port}`, { features });
      setOut({ ok: data });
    } catch (e) {
      setOut({ err: e?.response?.data?.detail || e.message });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-white/10 p-4 backdrop-blur-sm">
      <div className="glass-panel-strong animate-modal-in max-h-[90vh] w-full max-w-md overflow-hidden rounded-3xl p-0 shadow-[0_8px_32px_0_rgba(31,38,135,0.15)]">
        <div className="flex items-center justify-between border-b border-white/50 px-5 py-4">
          <h4 className="font-bold text-slate-800">Test prediction</h4>
          <button
            type="button"
            onClick={onClose}
            className="cursor-pointer rounded-xl p-1.5 text-slate-600 transition-colors hover:bg-white/50"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="p-5">
          <p className="text-sm text-slate-600">Port {port}</p>
          <input
            className="mt-2 w-full cursor-text rounded-xl border border-white/60 bg-white/45 px-3 py-2 text-sm text-slate-800 backdrop-blur-sm outline-none focus:ring-2 focus:ring-sky-200"
            placeholder="Comma-separated features"
            value={featuresStr}
            onChange={(e) => setFeaturesStr(e.target.value)}
          />
          <button
            type="button"
            disabled={busy}
            onClick={run}
            className="btn-primary mt-3 w-full cursor-pointer disabled:opacity-50"
          >
            {busy ? "Running…" : "Run"}
          </button>
          {out?.ok != null && (
            <pre className="code-block custom-scrollbar mt-3 max-h-40 overflow-auto text-xs">
              {JSON.stringify(out.ok, null, 2)}
            </pre>
          )}
          {out?.err && <p className="mt-3 text-sm text-red-600">{out.err}</p>}
        </div>
      </div>
    </div>
  );
}
