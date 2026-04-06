import { useState } from "react";
import axios from "axios";
import { BookOpen, FlaskConical, Square } from "lucide-react";
import TestPredictModal from "./TestPredictModal.jsx";

const API = axios.create({ baseURL: "http://localhost:8080" });

const badgeColors = {
  sklearn: "border-blue-300/60 bg-white/50 text-blue-800 shadow-sm",
  xgboost: "border-emerald-300/60 bg-white/50 text-emerald-800 shadow-sm",
  pytorch: "border-orange-300/60 bg-white/50 text-orange-800 shadow-sm",
  tensorflow: "border-green-300/60 bg-white/50 text-green-800 shadow-sm",
  onnx: "border-purple-300/60 bg-white/50 text-purple-800 shadow-sm",
  lightgbm: "border-teal-300/60 bg-white/50 text-teal-800 shadow-sm",
  catboost: "border-cyan-300/60 bg-white/50 text-cyan-800 shadow-sm",
};

export default function DeploymentCard({ dep, onStopped }) {
  const [modal, setModal] = useState(false);
  const cls =
    badgeColors[(dep.framework || "").toLowerCase()] ||
    "border-white/60 bg-white/45 text-slate-800 shadow-sm";

  const stop = async () => {
    if (!confirm(`Stop container ${dep.container_name}?`)) return;
    try {
      await API.delete(`/api/deployments/${encodeURIComponent(dep.container_name)}`);
      onStopped?.();
    } catch (e) {
      alert(e?.response?.data?.detail || e.message);
    }
  };

  return (
    <div className="glass-panel hover:scale-[1.01] hover:shadow-xl relative cursor-default overflow-hidden rounded-2xl p-5 transition-all duration-200">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[3px] bg-gradient-to-r from-blue-400 to-teal-400" />
      <div className="flex flex-wrap items-start justify-between gap-2">
        <span
          className={`rounded-full border px-2.5 py-0.5 text-xs font-bold backdrop-blur-sm ${cls}`}
        >
          {dep.framework}
        </span>
        <span className="text-sm font-bold text-slate-800">{dep.model_name}</span>
      </div>
      <div className="mt-4 space-y-1 text-sm text-slate-600">
        <p className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
          {dep.status === "running" ? "Running" : dep.status}
        </p>
        <p>
          Port: <span className="font-semibold text-slate-800">{dep.port}</span>
        </p>
        <p className="truncate text-slate-700">{dep.api_url}</p>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <a
          href={dep.docs_url}
          target="_blank"
          rel="noreferrer"
          className="btn-ghost inline-flex min-w-[6rem] flex-1 cursor-pointer justify-center py-2 text-xs"
        >
          <BookOpen className="h-3.5 w-3.5" /> API Docs
        </a>
        <button
          type="button"
          onClick={() => setModal(true)}
          className="btn-ghost inline-flex min-w-[6rem] flex-1 cursor-pointer justify-center py-2 text-xs"
        >
          <FlaskConical className="h-3.5 w-3.5" /> Test
        </button>
        <button
          type="button"
          onClick={stop}
          className="btn-danger inline-flex min-w-[6rem] flex-1 cursor-pointer justify-center py-2 text-xs"
        >
          <Square className="h-3.5 w-3.5" /> Stop
        </button>
      </div>
      <TestPredictModal open={modal} onClose={() => setModal(false)} port={dep.port} />
    </div>
  );
}
