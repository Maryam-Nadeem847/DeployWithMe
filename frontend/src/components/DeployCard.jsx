import { useCallback, useState } from "react";
import { CheckCircle2, CloudUpload, FileCode2, HelpCircle, Loader2, UploadCloud } from "lucide-react";

const FORMAT_PILLS = [".pkl", ".pt", ".h5", ".onnx", ".joblib"];

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DeployCard({ onDeploy, disabled }) {
  const [modelFile, setModelFile] = useState(null);
  const [reqFile, setReqFile] = useState(null);
  const [drag, setDrag] = useState(false);

  const onDropModel = useCallback((e) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setModelFile(f);
  }, []);

  const canSubmit = modelFile && !disabled;

  return (
    <section id="deploy" className="mx-auto max-w-xl px-4 pb-12">
      <div className="glass-panel-strong hover:scale-[1.01] cursor-default rounded-3xl p-8 transition-transform duration-200">
        <h2 className="bg-gradient-to-r from-slate-700 to-indigo-500 bg-clip-text text-xl font-bold text-transparent">
          Deploy Your Model
        </h2>

        <div
          role="button"
          tabIndex={0}
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDropModel}
          className={`mt-6 flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-12 transition-all duration-200 ${
            drag
              ? "border-blue-400 bg-blue-50/30"
              : "border-blue-300/60 bg-white/20 hover:border-blue-400 hover:bg-blue-50/30"
          }`}
          onClick={() => document.getElementById("model-input")?.click()}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") document.getElementById("model-input")?.click();
          }}
        >
          <UploadCloud className="h-10 w-10 text-sky-500" strokeWidth={1.5} aria-hidden />
          <p className="mt-3 font-semibold text-slate-800">Drop your model file here</p>
          <p className="text-sm text-slate-600">or click to browse</p>
          <p className="mt-2 text-xs text-slate-500">Also: .joblib \u00b7 .pth \u00b7 .keras</p>
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
            id="model-input"
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
          <div className="mt-3 flex items-center gap-2 text-sm text-emerald-600">
            <CheckCircle2 className="h-4 w-4 shrink-0" />
            <span className="font-medium text-slate-800">{modelFile.name}</span>
            <span className="text-slate-600">({formatBytes(modelFile.size)})</span>
          </div>
        )}

        <div className="mt-8">
          <div className="mb-2 flex items-center gap-2">
            <label className="text-sm font-semibold text-slate-800">requirements.txt (Optional)</label>
            <span className="group relative">
              <HelpCircle className="h-4 w-4 cursor-help text-slate-500" />
              <span className="pointer-events-none absolute left-0 top-6 z-20 hidden w-64 rounded-2xl border border-white/60 bg-white/60 p-2 text-xs text-slate-700 shadow-[0_8px_32px_0_rgba(31,38,135,0.12)] backdrop-blur-xl group-hover:block">
                Recommended for PyTorch and TensorFlow.{" "}
                <code className="rounded bg-black/10 px-1 font-mono text-slate-800 backdrop-blur-sm">
                  pip freeze &gt; requirements.txt
                </code>
              </span>
            </span>
          </div>
          <div
            className="flex cursor-pointer items-center gap-3 rounded-2xl border border-white/60 bg-white/40 px-4 py-3 backdrop-blur-sm transition-all duration-200 hover:bg-white/55"
            onClick={() => document.getElementById("req-input")?.click()}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") document.getElementById("req-input")?.click();
            }}
            role="button"
            tabIndex={0}
          >
            <FileCode2 className="h-5 w-5 text-cyan-500" />
            <span className="text-sm text-slate-600">
              {reqFile ? reqFile.name : "Click to attach requirements.txt"}
            </span>
            <input
              id="req-input"
              type="file"
              className="hidden"
              accept=".txt"
              onChange={(e) => {
                const f = e.target.files?.[0];
                setReqFile(f || null);
              }}
            />
          </div>
          {reqFile && (
            <div className="mt-2 flex items-center gap-2 text-sm text-cyan-600">
              <CheckCircle2 className="h-4 w-4" />
              Attached: {reqFile.name}
            </div>
          )}
        </div>

        <button
          type="button"
          disabled={!canSubmit || disabled}
          onClick={() => canSubmit && !disabled && onDeploy(modelFile, reqFile)}
          className={`mt-8 flex w-full cursor-pointer items-center justify-center gap-2 rounded-xl py-3 text-sm font-semibold transition-all duration-200 ${
            canSubmit && !disabled
              ? "btn-primary"
              : "cursor-not-allowed border border-white/50 bg-white/30 text-slate-500 opacity-80 backdrop-blur-sm"
          }`}
        >
          {disabled ? <Loader2 className="h-5 w-5 animate-spin" /> : null}
          Deploy Model
        </button>
      </div>
    </section>
  );
}
