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
          {out?.ok != null && out.ok.predicted_class !== undefined ? (
            <div className="mt-3 space-y-3">
              <div className="rounded-2xl border border-emerald-200/70 bg-gradient-to-br from-emerald-50/80 to-teal-50/60 p-4 backdrop-blur-sm">
                <p className="text-xs font-semibold uppercase tracking-wide text-emerald-700">
                  Predicted class
                </p>
                <p className="mt-1 text-2xl font-bold text-slate-800">
                  Class {out.ok.predicted_class}
                </p>
                {typeof out.ok.confidence === "number" && (
                  <p className="mt-0.5 text-sm font-medium text-emerald-700">
                    Confidence: {(out.ok.confidence * 100).toFixed(2)}%
                  </p>
                )}
              </div>
              {Array.isArray(out.ok.all_probabilities) && (
                <details className="rounded-xl border border-white/60 bg-white/40 px-3 py-2 text-xs text-slate-700 backdrop-blur-sm">
                  <summary className="cursor-pointer font-semibold text-slate-800">
                    All probabilities ({out.ok.all_probabilities.length})
                  </summary>
                  <ul className="mt-2 space-y-1">
                    {out.ok.all_probabilities.map((p, i) => (
                      <li
                        key={i}
                        className={`flex justify-between font-mono ${
                          i === out.ok.predicted_class
                            ? "font-bold text-emerald-700"
                            : ""
                        }`}
                      >
                        <span>Class {i}</span>
                        <span>
                          {typeof p === "number" ? (p * 100).toFixed(2) : p}%
                        </span>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          ) : out?.ok != null ? (
            <pre className="code-block custom-scrollbar mt-3 max-h-40 overflow-auto text-xs">
              {JSON.stringify(out.ok, null, 2)}
            </pre>
          ) : null}
          {out?.err && <p className="mt-3 text-sm text-red-600">{out.err}</p>}
        </div>
      </div>
    </div>
  );
}
