import { AlertCircle, CheckCircle, Container, Loader2, Search, Settings, Sparkles, Wrench } from "lucide-react";

const NODES = [
  { id: "validate", label: "Validate inputs" },
  { id: "detect", label: "Detect framework" },
  { id: "prepare", label: "Prepare build" },
  { id: "write", label: "Write bundle" },
  { id: "build", label: "Docker build" },
  { id: "heal", label: "Self-heal (LLM)" },
  { id: "run", label: "Run container" },
  { id: "health", label: "Health check" },
  { id: "live", label: "Live API" },
];

function logIcon(line) {
  const s = String(line).toLowerCase();
  if (s.includes("fail") || s.includes("error")) return <AlertCircle className="inline h-3.5 w-3.5 text-red-400" />;
  if (s.includes("docker")) return <Container className="inline h-3.5 w-3.5 text-sky-400" />;
  if (s.includes("build")) return <Wrench className="inline h-3.5 w-3.5 text-amber-400" />;
  if (s.includes("detect")) return <Search className="inline h-3.5 w-3.5 text-violet-400" />;
  if (s.includes("heal")) return <Sparkles className="inline h-3.5 w-3.5 text-pink-400" />;
  if (s.includes("ok") || s.includes("success")) return <CheckCircle className="inline h-3.5 w-3.5 text-emerald-400" />;
  return <Settings className="inline h-3.5 w-3.5 animate-spin text-slate-400" />;
}

function liveStatusBadge(job) {
  const st = job?.status;
  const logs = (job?.decision_log || []).join("\n").toLowerCase();
  if (st === "failed") {
    return { label: "Failed", dotClass: "bg-red-400", pulse: false };
  }
  if (st === "success") {
    return { label: "Live", dotClass: "bg-emerald-400", pulse: false };
  }
  if (st === "awaiting_confirmation") {
    return { label: "Awaiting confirm", dotClass: "bg-amber-400", pulse: true };
  }
  if (logs.includes("health check")) {
    return { label: "Health check\u2026", dotClass: "bg-cyan-400", pulse: true };
  }
  return { label: "Deploying\u2026", dotClass: "bg-emerald-400", pulse: true };
}

function computeTimeline(job) {
  const logs = job.decision_log || [];
  const text = logs.join("\n").toLowerCase();
  const err = (job.error || "").toLowerCase();
  const failed = job.status === "failed";
  const st = job.status;

  const has = (sub) => text.includes(sub);

  const doneValidate = has("validated model path") || has("validated");
  const doneDetect = has("detected framework") || has("llm classification");
  const donePrepare = has("build id") && has("staging directory");
  const doneWrite =
    has("generated fastapi") || has("wrote dockerfile") || has("staged model");
  const doneBuild = has("docker image built");
  const doneRun = has("started container");
  const doneHealth = has("health check ok");
  const doneLive = st === "success";

  /** @type {Array<'pending'|'active'|'done'|'failed'>} */
  const states = NODES.map(() => "pending");

  const markDone = (idx) => {
    for (let i = 0; i <= idx; i += 1) states[i] = "done";
  };

  if (doneLive) {
    markDone(8);
    return { states, failedIndex: -1 };
  }

  if (failed) {
    if (err.includes("validate") || err.includes("not found")) {
      states[0] = "failed";
      return { states, failedIndex: 0 };
    }
    if (err.includes("detect") || err.includes("classification")) {
      markDone(0);
      states[1] = "failed";
      return { states, failedIndex: 1 };
    }
    if (err.includes("docker_build") || err.includes("build")) {
      markDone(3);
      states[4] = "failed";
      return { states, failedIndex: 4 };
    }
    if (err.includes("heal")) {
      markDone(4);
      states[5] = "failed";
      return { states, failedIndex: 5 };
    }
    if (err.includes("docker_run") || err.includes("run")) {
      markDone(5);
      states[6] = "failed";
      return { states, failedIndex: 6 };
    }
    if (err.includes("health")) {
      markDone(6);
      states[7] = "failed";
      return { states, failedIndex: 7 };
    }
    markDone(3);
    states[4] = "failed";
    return { states, failedIndex: 4 };
  }

  if (!doneValidate) {
    states[0] = "active";
    return { states, failedIndex: -1 };
  }
  markDone(0);
  if (!doneDetect) {
    states[1] = "active";
    return { states, failedIndex: -1 };
  }
  markDone(1);
  if (!donePrepare) {
    states[2] = "active";
    return { states, failedIndex: -1 };
  }
  markDone(2);
  if (!doneWrite) {
    states[3] = "active";
    return { states, failedIndex: -1 };
  }
  markDone(3);
  if (!doneBuild) {
    states[4] = "active";
    return { states, failedIndex: -1 };
  }
  markDone(5);

  if (st === "awaiting_confirmation") {
    states[6] = "active";
    return { states, failedIndex: -1 };
  }

  if (!doneRun) {
    states[6] = "active";
    return { states, failedIndex: -1 };
  }
  markDone(6);
  if (!doneHealth) {
    states[7] = "active";
    return { states, failedIndex: -1 };
  }
  markDone(7);
  states[8] = "active";
  return { states, failedIndex: -1 };
}

export default function DeployProgress({ job, visible }) {
  if (!visible || !job) return null;

  const pct = Math.min(100, Math.max(0, job.progress ?? 0));
  const { states } = computeTimeline(job);
  const titleModel = job.model_name || "model";
  const logs = job.decision_log || [];
  const badge = liveStatusBadge(job);

  return (
    <section className="mx-auto max-w-lg px-4 pb-12 md:max-w-2xl">
      <div className="glass-panel hover:scale-[1.01] cursor-default rounded-3xl p-6 transition-transform duration-200 md:p-8">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2 font-bold text-slate-800">
            <Loader2 className="h-5 w-5 shrink-0 animate-spin text-sky-500 drop-shadow-[0_0_8px_rgba(14,165,233,0.45)]" />
            <span className="truncate">Deploying {titleModel}\u2026</span>
          </div>
          <div className="flex shrink-0 items-center gap-2 rounded-full border border-white/60 bg-white/35 px-3 py-1.5 text-xs font-semibold text-slate-700 backdrop-blur-md">
            <span
              className={`h-2 w-2 rounded-full ${badge.dotClass} ${badge.pulse ? "animate-pulse" : ""}`}
            />
            {badge.label}
          </div>
        </div>

        <div className="mt-6">
          <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-white/40 backdrop-blur-sm">
            <div
              className="relative h-full overflow-hidden rounded-full transition-[width] duration-500 ease-out"
              style={{ width: `${pct}%` }}
            >
              <div className="h-full w-full bg-gradient-to-r from-cyan-400 to-pink-400" />
              <div className="progress-shimmer pointer-events-none absolute inset-0 overflow-hidden rounded-full" />
            </div>
          </div>
          <div className="mt-2 flex justify-between text-xs text-slate-600">
            <span>{pct}%</span>
            <span className="font-medium text-slate-700">{job.status}</span>
          </div>
        </div>

        <div className="relative mt-8">
          <ul className="space-y-0">
            {NODES.map((node, idx) => {
              const s = states[idx];
              const isPending = s === "pending";
              const isActive = s === "active";
              const isDone = s === "done";
              const isFailed = s === "failed";

              let pillClass =
                "border border-white/40 bg-white/30 text-slate-400 backdrop-blur-md";
              if (isDone) {
                pillClass =
                  "border border-emerald-200/50 bg-gradient-to-br from-emerald-300 to-teal-300 text-white shadow-sm shadow-emerald-200/40";
              } else if (isActive) {
                pillClass =
                  "border border-blue-300/60 bg-gradient-to-r from-blue-400 to-teal-400 text-white shadow-lg shadow-blue-300/50";
              } else if (isFailed) {
                pillClass =
                  "border border-rose-200/60 bg-gradient-to-br from-red-300 to-rose-300 text-white shadow-sm shadow-rose-200/50";
              }

              const segmentFilled = states[idx] === "done";
              const showConnector = idx < NODES.length - 1;

              return (
                <li key={node.id}>
                  <div className="flex gap-3">
                    <div
                      className={`relative z-[1] flex h-10 min-w-[2.5rem] shrink-0 items-center justify-center rounded-full text-xs font-bold ${pillClass}`}
                    >
                      {isDone && <span className="text-white">\u2713</span>}
                      {isFailed && <span className="text-white">\u2715</span>}
                      {isActive && <Loader2 className="h-4 w-4 animate-spin text-white" />}
                      {isPending && <span className="text-slate-400">{idx + 1}</span>}
                    </div>
                    <div
                      className={`flex min-h-10 flex-1 items-center rounded-2xl border px-4 py-2.5 text-sm font-medium transition-all duration-200 ${pillClass}`}
                    >
                      <span
                        className={
                          isPending ? "text-slate-500" : isActive || isFailed || isDone ? "text-white" : "text-slate-800"
                        }
                      >
                        {node.label}
                      </span>
                    </div>
                  </div>
                  {showConnector && (
                    <div className="ml-5 flex h-6 w-6 justify-center">
                      <div className="relative h-full w-0.5 overflow-hidden rounded-full bg-white/40">
                        <div
                          className={`absolute left-0 top-0 h-full w-full origin-top rounded-full bg-gradient-to-b from-cyan-400 to-pink-400 transition-transform duration-500 ease-out ${
                            segmentFilled ? "scale-y-100" : "scale-y-0"
                          }`}
                        />
                      </div>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </div>

        <div className="mt-8">
          <p className="mb-2 text-sm font-semibold text-slate-800">Reasoning log</p>
          <div className="custom-scrollbar max-h-64 overflow-y-auto rounded-2xl border border-white/50 bg-black/5 p-4 font-mono text-xs leading-relaxed text-slate-800 shadow-inner backdrop-blur-xl">
            {logs.length === 0 ? (
              <span className="text-slate-500">Waiting for log output\u2026</span>
            ) : (
              logs.map((line, i) => (
                <div
                  key={`${i}-${String(line).slice(0, 32)}`}
                  className="log-line-enter border-b border-white/25 py-2 last:border-0"
                  style={{ animationDelay: `${Math.min(i * 40, 600)}ms` }}
                >
                  <span className="mr-2 text-slate-500">{logIcon(line)}</span>
                  <span className="text-slate-700">{line}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
