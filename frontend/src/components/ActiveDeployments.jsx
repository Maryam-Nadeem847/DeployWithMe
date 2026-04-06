import { Activity, Box, Cpu, Layers, RefreshCw } from "lucide-react";
import DeploymentCard from "./DeploymentCard.jsx";

function portRangeLabel(items) {
  if (!items.length) return "\u2014";
  const ports = items.map((d) => d.port).filter((p) => p != null);
  if (!ports.length) return "\u2014";
  const min = Math.min(...ports);
  const max = Math.max(...ports);
  return min === max ? String(min) : `${min}\u2013${max}`;
}

export default function ActiveDeployments({ items, loading, onRefresh }) {
  const frameworks = new Set(items.map((d) => (d.framework || "").toLowerCase()).filter(Boolean));

  const stats = [
    {
      label: "Models running",
      value: String(items.length),
      sub: "Active containers",
      icon: Activity,
      accent: "from-blue-400 to-cyan-400",
    },
    {
      label: "Frameworks",
      value: String(frameworks.size || 0),
      sub: "Unique stacks",
      icon: Layers,
      accent: "from-emerald-400 to-teal-400",
    },
    {
      label: "Ports in use",
      value: portRangeLabel(items),
      sub: "Host bindings",
      icon: Cpu,
      accent: "from-fuchsia-400 to-pink-400",
    },
  ];

  return (
    <section id="active" className="mx-auto max-w-6xl px-4 pb-20">
      <div className="flex items-center gap-3">
        <h2 className="bg-gradient-to-r from-slate-700 to-indigo-500 bg-clip-text text-2xl font-bold text-transparent">
          Deployed Models
        </h2>
        <button
          type="button"
          onClick={onRefresh}
          className="btn-ghost inline-flex cursor-pointer items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-medium"
          title="Refresh deployments"
        >
          <RefreshCw className="h-3.5 w-3.5" /> Refresh
        </button>
      </div>
      <p className="mt-1 leading-relaxed text-slate-600">
        Running services. Click Refresh to update.
      </p>

      <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-3">
        {loading
          ? [0, 1, 2].map((k) => (
              <div
                key={k}
                className="glass-panel skeleton-shimmer relative h-28 overflow-hidden rounded-2xl p-5"
              >
                <div className="h-4 w-24 rounded-lg bg-white/40" />
                <div className="mt-4 h-8 w-16 rounded-lg bg-white/50" />
              </div>
            ))
          : stats.map((s) => {
              const Icon = s.icon;
              return (
                <div
                  key={s.label}
                  className="glass-panel hover:scale-[1.01] relative cursor-default overflow-hidden rounded-2xl p-5 transition-transform duration-200"
                >
                  <div
                    className={`absolute right-3 top-3 flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br ${s.accent} p-2 text-white opacity-90 shadow-md`}
                  >
                    <Icon className="h-5 w-5" />
                  </div>
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{s.label}</p>
                  <p className="mt-2 text-3xl font-bold text-slate-800">{s.value}</p>
                  <p className="mt-1 text-sm text-slate-500">{s.sub}</p>
                </div>
              );
            })}
      </div>

      {loading ? (
        <div className="mt-10 grid grid-cols-1 gap-4 md:grid-cols-2">
          {[0, 1].map((k) => (
            <div
              key={k}
              className="glass-panel skeleton-shimmer h-48 overflow-hidden rounded-2xl p-5"
            />
          ))}
        </div>
      ) : items.length === 0 ? (
        <div className="glass-panel mt-10 flex flex-col items-center rounded-2xl py-16 text-center">
          <Box className="h-12 w-12 text-slate-400" />
          <p className="mt-4 font-bold text-slate-800">No models currently deployed</p>
          <p className="mt-1 text-sm text-slate-600">Deploy your first model above</p>
        </div>
      ) : (
        <div className="mt-8 grid grid-cols-1 gap-4 md:grid-cols-2">
          {items.map((dep) => (
            <DeploymentCard key={dep.container_name} dep={dep} onStopped={onRefresh} />
          ))}
        </div>
      )}
    </section>
  );
}
