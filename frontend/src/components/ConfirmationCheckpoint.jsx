import { useState } from "react";
import { AlertTriangle, ChevronDown, ChevronRight } from "lucide-react";

export default function ConfirmationCheckpoint({ job, onConfirm }) {
  const [open, setOpen] = useState(false);
  if (!job?.confirmation_data) return null;

  const c = job.confirmation_data;

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-white/10 p-4 backdrop-blur-sm">
      <div
        className="glass-panel-strong animate-modal-in w-full max-w-lg rounded-3xl border border-white/80 shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-deploy-title"
      >
        <div className="rounded-t-3xl bg-gradient-to-r from-amber-300 to-orange-300 px-6 py-4">
          <div className="flex items-start gap-3">
            <div>
              <h3 id="confirm-deploy-title" className="text-lg font-bold text-slate-900">
                Confirm Deployment
              </h3>
              <p className="mt-1 text-sm font-medium text-slate-800/90">
                Docker image built successfully. Review before going live.
              </p>
            </div>
            <AlertTriangle className="mt-0.5 h-6 w-6 shrink-0 text-amber-900/70" aria-hidden />
          </div>
        </div>

        <div className="p-6 md:p-8">
          <dl className="grid grid-cols-1 gap-3 rounded-2xl border border-white/60 bg-white/40 p-4 text-sm backdrop-blur-xl sm:grid-cols-2">
            <div>
              <dt className="text-slate-500">Framework</dt>
              <dd className="font-semibold text-slate-800">{c.framework ?? "—"}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Model</dt>
              <dd className="font-semibold text-slate-800">{c.model_name ?? job.model_name}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Build time</dt>
              <dd className="font-semibold text-slate-800">{c.build_duration_seconds ?? "—"} seconds</dd>
            </div>
            <div>
              <dt className="text-slate-500">Deploy port</dt>
              <dd className="font-semibold text-slate-800">
                {c.port != null ? c.port : "Auto-assigned after confirm"}
              </dd>
            </div>
            <div className="sm:col-span-2">
              <dt className="text-slate-500">Memory limit</dt>
              <dd className="font-semibold text-slate-800">{c.memory_limit ?? "—"}</dd>
            </div>
            <div className="sm:col-span-2">
              <dt className="text-slate-500">Image</dt>
              <dd className="break-all font-mono text-xs text-slate-800">{c.image_tag}</dd>
            </div>
          </dl>

          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="btn-ghost mt-4 w-full justify-start sm:w-auto"
          >
            {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            Dockerfile preview (first 10 lines)
          </button>
          {open && (
            <pre className="custom-scrollbar mt-2 max-h-48 overflow-auto rounded-xl border border-white/50 bg-black/10 p-4 font-mono text-xs text-slate-700 backdrop-blur-sm">
              {c.dockerfile_preview || "—"}
            </pre>
          )}

          <div className="mt-6 flex flex-col gap-3 sm:flex-row">
            <button type="button" onClick={() => onConfirm(true)} className="btn-primary flex-1 py-3">
              Confirm &amp; Deploy
            </button>
            <button type="button" onClick={() => onConfirm(false)} className="btn-danger flex-1 py-3">
              Cancel Deployment
            </button>
          </div>
          <p className="mt-4 text-center text-xs leading-relaxed text-slate-500">
            Confirm starts the container and exposes your live API.
          </p>
        </div>
      </div>
    </div>
  );
}
