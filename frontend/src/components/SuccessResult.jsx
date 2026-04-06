import { useState } from "react";
import axios from "axios";
import { BookOpen, Copy, ExternalLink, Sparkles } from "lucide-react";

const API = axios.create({ baseURL: "http://localhost:8080" });

function portFromUrl(url) {
  try {
    const u = new URL(url);
    return u.port ? parseInt(u.port, 10) : 80;
  } catch {
    return null;
  }
}

export default function SuccessResult({ job, onDeployAnother }) {
  const [featuresStr, setFeaturesStr] = useState("");
  const [predResult, setPredResult] = useState(null);
  const [busy, setBusy] = useState(false);

  if (!job?.api_url) return null;

  const port = portFromUrl(job.api_url);

  const runPredict = async () => {
    if (!port) return;
    const parts = featuresStr.split(",").map((s) => s.trim()).filter(Boolean);
    const features = parts.map((x) => parseFloat(x));
    if (features.some((n) => Number.isNaN(n))) {
      setPredResult({ error: "Enter comma-separated numbers only." });
      return;
    }
    setBusy(true);
    setPredResult(null);
    try {
      const { data } = await API.post(`/api/test-predict/${port}`, { features });
      setPredResult({ ok: data });
    } catch (e) {
      setPredResult({ error: e?.response?.data?.detail || e.message });
    } finally {
      setBusy(false);
    }
  };

  const [copied, setCopied] = useState(false);

  const copy = () => {
    navigator.clipboard.writeText(job.api_url);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section className="mx-auto max-w-3xl px-4 pb-12">
      <div className="glass-panel-strong success-card-glow hover:scale-[1.01] cursor-default rounded-3xl border border-emerald-200/50 p-8 transition-transform duration-200">
        <div className="flex items-center gap-2 text-emerald-600">
          <Sparkles className="h-6 w-6" />
          <h3 className="text-xl font-bold text-slate-800">Deployment Successful!</h3>
        </div>

        <div className="mt-6 flex flex-wrap items-center gap-2">
          <span className="rounded-full border border-white/60 bg-white/45 px-3 py-1 text-xs font-bold text-indigo-800 backdrop-blur-sm">
            {job.framework || "framework"}
          </span>
          <span className="text-sm font-semibold text-slate-800">{job.model_name}</span>
        </div>

        <div className="mt-4 flex flex-col gap-2 rounded-2xl border border-white/60 bg-white/40 p-4 backdrop-blur-xl sm:flex-row sm:items-center sm:justify-between">
          <code className="break-all text-sm text-slate-800">{job.api_url}</code>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={copy} className="btn-ghost cursor-pointer py-2 text-xs">
              <Copy className="h-3.5 w-3.5" /> {copied ? "Copied!" : "Copy URL"}
            </button>
            <a
              href={job.api_url}
              target="_blank"
              rel="noreferrer"
              className="btn-primary cursor-pointer py-2 text-xs"
            >
              <ExternalLink className="h-3.5 w-3.5" /> Open
            </a>
            <a
              href={`${job.api_url}/docs`}
              target="_blank"
              rel="noreferrer"
              className="btn-ghost cursor-pointer py-2 text-xs"
            >
              <BookOpen className="h-3.5 w-3.5" /> API Docs
            </a>
          </div>
        </div>

        <div className="mt-8">
          <h4 className="font-bold text-slate-800">Quick test</h4>
          <p className="text-sm text-slate-600">Send a 1D feature vector (comma-separated).</p>
          <input
            className="mt-3 w-full cursor-text rounded-xl border border-white/60 bg-white/45 px-3 py-2 text-sm text-slate-800 shadow-inner backdrop-blur-sm outline-none transition-all duration-200 focus:border-sky-400 focus:ring-2 focus:ring-sky-200"
            placeholder="5.1, 3.5, 1.4, 0.2"
            value={featuresStr}
            onChange={(e) => setFeaturesStr(e.target.value)}
          />
          <button
            type="button"
            disabled={busy || !port}
            onClick={runPredict}
            className="btn-primary mt-3 cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy ? "Running…" : "Run Prediction"}
          </button>
          {predResult?.ok != null && (
            <pre className="code-block custom-scrollbar mt-4 max-h-48 overflow-auto text-xs">
              {JSON.stringify(predResult.ok, null, 2)}
            </pre>
          )}
          {predResult?.error && <p className="mt-4 text-sm text-red-600">{predResult.error}</p>}
        </div>

        <button
          type="button"
          onClick={onDeployAnother}
          className="btn-ghost mt-8 w-full cursor-pointer py-3"
        >
          Deploy Another Model
        </button>
      </div>
    </section>
  );
}
