import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import {
  Check,
  CloudUpload,
  Copy,
  ExternalLink,
  Loader2,
  Pencil,
} from "lucide-react";
import { useCloudDeploy } from "../hooks/useCloudDeploy.js";

const FORMAT_PILLS = [".pkl", ".pt", ".h5", ".onnx", ".joblib"];

const CARD =
  "rounded-2xl border border-white/60 bg-white/50 p-6 shadow-[0_8px_32px_0_rgba(31,38,135,0.07)] backdrop-blur-xl";

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DeployCloudPage() {
  const location = useLocation();
  const {
    step,
    jobId,
    confirmedName,
    setConfirmedName,
    hfToken,
    setHfToken,
    framework,
    modelFile,
    setModelFile,
    statusMessage,
    result,
    error,
    setError,
    stepLabels,
    activePillIndex,
    submitUpload,
    confirmDeploy,
    pollStatus,
    hydrateFromRedirect,
    reset,
  } = useCloudDeploy();

  const [drag, setDrag] = useState(false);
  const [copied, setCopied] = useState(false);
  const hydratedRef = useRef(false);

  useEffect(() => {
    hydratedRef.current = false;
  }, [location.pathname]);

  useEffect(() => {
    if (hydratedRef.current) return;
    const st = location.state;
    if (!st) return;
    if (st.modelFile instanceof File) {
      hydrateFromRedirect(st.modelFile, st.suggestedSpaceName || st.suggested_space_name || "");
      hydratedRef.current = true;
    }
  }, [location.state, hydrateFromRedirect]);

  const previewSlug = (confirmedName || "").trim() || "your-space-name";
  const previewLine = `huggingface.co/spaces/<your-username>/${previewSlug.replace(/^\/+|\/+$/g, "")}`;

  const spaceUrl = result?.space_url || "";
  const apiUrl = result?.api_url || "";

  const copyPrimary = () => {
    const u = spaceUrl || apiUrl;
    if (!u) return;
    navigator.clipboard.writeText(u);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="mx-auto max-w-2xl px-4 py-10">
      <section className="mb-8">
        <h1 className="text-3xl font-bold text-slate-800">Deploy to Cloud</h1>
        <p className="mt-2 text-slate-600">
          Hugging Face Spaces deployment with a quick name confirmation step.
        </p>
      </section>

      {framework && (step === "checkpoint" || step === "deploying") && (
        <p className="mb-4 text-sm text-slate-600">
          Framework: <span className="font-semibold text-slate-800">{framework}</span>
          {jobId ? (
            <span className="ml-2 text-slate-400">· Job {jobId}</span>
          ) : null}
        </p>
      )}

      {step === "upload" && (
        <div className={`${CARD} space-y-6`}>
          <h2 className="text-lg font-bold text-slate-800">Step 1 — Upload &amp; token</h2>

          <div
            role="button"
            tabIndex={0}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              const f = e.dataTransfer.files?.[0];
              if (f) setModelFile(f);
            }}
            onClick={() => document.getElementById("cloud-model-input")?.click()}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") document.getElementById("cloud-model-input")?.click();
            }}
            className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-12 transition-all ${
              drag
                ? "border-blue-400 bg-blue-50/30"
                : "border-blue-300/60 bg-white/20 hover:border-blue-400 hover:bg-blue-50/30"
            }`}
          >
            <CloudUpload className="h-10 w-10 text-sky-500" />
            <p className="mt-3 font-semibold text-slate-800">Drop your model file here</p>
            <p className="text-sm text-slate-600">or click to browse</p>
            <div className="mt-4 flex flex-wrap justify-center gap-2">
              {FORMAT_PILLS.map((ext) => (
                <span
                  key={ext}
                  className="rounded-full border border-white/70 bg-white/50 px-3 py-1 text-xs text-slate-600 backdrop-blur-sm"
                >
                  {ext}
                </span>
              ))}
            </div>
            <input
              id="cloud-model-input"
              type="file"
              className="hidden"
              accept=".joblib,.pkl,.pickle,.pt,.pth,.onnx,.h5,.keras"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) setModelFile(f);
              }}
            />
          </div>
          {modelFile && (
            <p className="text-sm text-slate-700">
              <span className="font-semibold">{modelFile.name}</span> ({formatBytes(modelFile.size)})
            </p>
          )}

          <div>
            <label className="text-sm font-semibold text-slate-800">Hugging Face token</label>
            <p className="mt-1 text-sm text-slate-500">Get token at huggingface.co/settings/tokens</p>
            <input
              type="password"
              value={hfToken}
              onChange={(e) => setHfToken(e.target.value)}
              placeholder="hf_..."
              className="mt-2 w-full rounded-xl border border-white/60 bg-white/50 px-3 py-2 text-sm text-slate-800 backdrop-blur-sm outline-none focus:ring-2 focus:ring-sky-200"
            />
          </div>

          {error && <p className="text-sm text-red-600">{error}</p>}

          <button
            type="button"
            onClick={() => {
              setError(null);
              submitUpload();
            }}
            className="btn-primary w-full sm:w-auto"
          >
            Next →
          </button>
        </div>
      )}

      {step === "checkpoint" && (
        <div className={`${CARD} space-y-5`}>
          <h2 className="text-lg font-bold text-slate-800">☁️ Confirm Your Space Name</h2>

          <div>
            <label className="text-sm font-semibold text-slate-800">Space Name</label>
            <div className="relative mt-1">
              <input
                type="text"
                value={confirmedName}
                onChange={(e) => setConfirmedName(e.target.value)}
                className="w-full rounded-xl border border-white/60 bg-white/50 py-2.5 pl-3 pr-10 text-sm text-slate-800 backdrop-blur-sm outline-none focus:ring-2 focus:ring-sky-200"
                placeholder="mymodel-deploy"
                autoComplete="off"
              />
              <Pencil
                className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
                aria-hidden
              />
            </div>
          </div>

          <div className="rounded-xl border border-white/50 bg-black/5 px-4 py-3 text-sm text-slate-700 backdrop-blur-sm">
            <p className="font-medium text-slate-800">Your model will be live at:</p>
            <p className="mt-1 break-all font-mono text-slate-600">{previewLine}</p>
          </div>

          {error && <p className="text-sm text-red-600">{error}</p>}

          <button type="button" onClick={() => confirmDeploy()} className="btn-primary w-full">
            ✅ Confirm &amp; Deploy
          </button>
        </div>
      )}

      {step === "deploying" && (
        <div className={`${CARD} space-y-6`}>
          <h2 className="text-lg font-bold text-slate-800">Deployment in progress</h2>

          <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
            {stepLabels.map((label, idx) => {
              const done = idx < activePillIndex;
              const active = idx === activePillIndex;
              const cls = done
                ? "border-emerald-300/60 bg-gradient-to-r from-emerald-300 to-teal-300 text-white"
                : active
                  ? "border-blue-300/60 bg-gradient-to-r from-blue-400 to-teal-400 text-white shadow-lg shadow-blue-300/40"
                  : "border-white/60 bg-white/30 text-slate-500";
              return (
                <div
                  key={label}
                  className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-xs font-semibold ${cls}`}
                >
                  {done && <Check className="h-3.5 w-3.5" strokeWidth={3} />}
                  {active && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                  {label}
                </div>
              );
            })}
          </div>

          <p className="text-sm text-slate-600">
            {statusMessage || "Starting…"}
          </p>

          <button type="button" onClick={() => pollStatus()} className="btn-ghost text-xs">
            Refresh status
          </button>

          {error && <p className="text-sm text-red-600">{error}</p>}
        </div>
      )}

      {step === "failed" && (
        <div className={`${CARD} space-y-4`}>
          <h2 className="text-lg font-bold text-slate-800">Deployment failed</h2>
          <p className="text-sm text-slate-600">{error || "Something went wrong."}</p>
          <button type="button" onClick={() => reset()} className="btn-primary">
            Start over
          </button>
        </div>
      )}

      {step === "success" && (
        <div className={`${CARD} space-y-6`}>
          <h2 className="text-2xl font-bold text-slate-800">🎉 Your Model is Live!</h2>
          {result?.space_name && (
            <p className="text-sm text-slate-600">
              Space: <span className="font-semibold text-slate-800">{result.space_name}</span>
            </p>
          )}
          <div className="space-y-2 text-sm">
            {spaceUrl && (
              <p className="break-all text-slate-700">
                <span className="font-medium text-slate-800">Space: </span>
                {spaceUrl}
              </p>
            )}
            {apiUrl && (
              <p className="break-all text-slate-700">
                <span className="font-medium text-slate-800">API: </span>
                {apiUrl}
              </p>
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            {spaceUrl && (
              <a
                href={spaceUrl}
                target="_blank"
                rel="noreferrer"
                className="btn-primary"
              >
                <ExternalLink className="mr-1 h-4 w-4" /> Open Space
              </a>
            )}
            {apiUrl && (
              <a
                href={apiUrl}
                target="_blank"
                rel="noreferrer"
                className="btn-ghost"
              >
                <ExternalLink className="mr-1 h-4 w-4" /> View API
              </a>
            )}
            <button type="button" onClick={copyPrimary} className="btn-ghost">
              <Copy className="mr-1 h-4 w-4" />
              {copied ? "Copied!" : "Copy URL"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
