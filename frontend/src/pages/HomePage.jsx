import { Bot, CloudUpload, Rocket, ShieldCheck } from "lucide-react";
import { Link } from "react-router-dom";
import AgentMascot from "../components/AgentMascot.jsx";

const frameworks = [
  { name: "sklearn", icon: "🧠", cls: "border-blue-300/60 text-blue-900" },
  { name: "XGBoost", icon: "⚡", cls: "border-emerald-300/60 text-emerald-900" },
  { name: "PyTorch", icon: "🔥", cls: "border-orange-300/60 text-orange-900" },
  { name: "TensorFlow", icon: "🟢", cls: "border-green-300/60 text-green-900" },
  { name: "ONNX", icon: "🧩", cls: "border-purple-300/60 text-purple-900" },
];

export default function HomePage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-10 md:py-14">
      <section className="grid min-h-[80vh] items-center gap-10 md:grid-cols-2">
        <div>
          <h1 className="text-4xl font-bold tracking-tight text-slate-800 md:text-6xl">
            Deploy ML/DL Models with an
            <span className="mt-2 block bg-gradient-to-r from-blue-500 to-teal-400 bg-clip-text text-transparent">
              Autonomous Agent
            </span>
          </h1>
          <p className="mt-5 max-w-xl text-lg text-slate-600">
            From model file to live API. Local Docker deployment and cloud deployment workflows
            in one polished control center.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link to="/deploy" className="btn-primary">
              Deploy Locally
            </Link>
            <Link to="/deploy/cloud" className="btn-ghost">
              Deploy to Cloud
            </Link>
          </div>
        </div>
        <AgentMascot />
      </section>

      <section className="mt-10 grid gap-4 sm:grid-cols-3">
        {[
          { icon: Bot, title: "5+ Frameworks", text: "sklearn to ONNX supported" },
          { icon: Rocket, title: "Zero DevOps", text: "No Dockerfiles to hand-write" },
          { icon: ShieldCheck, title: "Self-Healing Agent", text: "LLM-assisted build recovery" },
        ].map((s) => (
          <div key={s.title} className="glass-panel hover:scale-[1.01] p-5">
            <s.icon className="h-6 w-6 text-blue-500" />
            <p className="mt-3 text-xl font-bold text-slate-800">{s.title}</p>
            <p className="mt-1 text-sm text-slate-600">{s.text}</p>
          </div>
        ))}
      </section>

      <section className="mt-14">
        <h2 className="text-center text-2xl font-bold text-slate-800">How It Works</h2>
        <div className="mt-6 grid gap-4 md:grid-cols-3">
          {[
            ["01", "Upload Model", "Drop your serialized model and optional requirements file."],
            ["02", "Agent Deploys", "Build, heal, run, and health-check the deployment workflow."],
            ["03", "Get Live API", "Receive /predict and /health endpoints instantly."],
          ].map(([n, t, d]) => (
            <div key={t} className="glass-panel hover:scale-[1.01] p-6">
              <p className="text-sm font-bold text-blue-500">{n}</p>
              <h3 className="mt-2 text-lg font-bold text-slate-800">{t}</h3>
              <p className="mt-2 text-sm text-slate-600">{d}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mt-14 pb-8">
        <h2 className="text-center text-2xl font-bold text-slate-800">Supported Frameworks</h2>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
          {frameworks.map((f) => (
            <span
              key={f.name}
              className={`glass-panel inline-flex items-center gap-1.5 rounded-full border px-4 py-1.5 text-sm ${f.cls}`}
            >
              <span>{f.icon}</span> {f.name}
            </span>
          ))}
        </div>
      </section>
    </div>
  );
}
