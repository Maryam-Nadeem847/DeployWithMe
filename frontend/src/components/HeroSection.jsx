import { useEffect, useState } from "react";
import { Cpu, Container, Zap } from "lucide-react";

const PHRASES = [
  "Powered by LangGraph + Gemini AI",
  "Zero DevOps Knowledge Required",
  "sklearn · PyTorch · TensorFlow · ONNX · XGBoost",
];

const MOCK_LINES = [
  "[12:04:01] Detecting framework… sklearn",
  "[12:04:02] Writing FastAPI bundle + Dockerfile",
  "[12:04:18] docker build — image deploy_agent_iris:latest",
  "[12:04:42] Health check OK → http://localhost:8001/predict",
  "[12:05:01] POST /predict — 200 OK (12ms)",
];

export default function HeroSection() {
  const [i, setI] = useState(0);
  const [text, setText] = useState("");
  const phrase = PHRASES[i % PHRASES.length];

  useEffect(() => {
    let c = 0;
    setText("");
    const id = setInterval(() => {
      c += 1;
      setText(phrase.slice(0, c));
      if (c >= phrase.length) clearInterval(id);
    }, 42);
    return () => clearInterval(id);
  }, [phrase]);

  useEffect(() => {
    const id = setInterval(() => setI((x) => x + 1), 4500);
    return () => clearInterval(id);
  }, []);

  const loopLines = [...MOCK_LINES, ...MOCK_LINES];

  return (
    <section className="mx-auto max-w-6xl px-4 pb-10 pt-12 text-center md:pt-16">
      <h1 className="text-4xl font-bold tracking-tight text-slate-800 md:text-6xl">
        Deploy Any ML/DL Model
        <span className="mt-2 block bg-gradient-to-r from-blue-500 to-teal-400 bg-clip-text text-transparent">
          Instantly. Automatically.
        </span>
      </h1>
      <p className="mx-auto mt-5 min-h-[2rem] max-w-2xl text-lg leading-relaxed text-slate-600">
        <span className="font-medium text-slate-800">{text}</span>
        <span className="ml-1 inline-block h-5 w-0.5 animate-pulse bg-sky-400" />
      </p>

      <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
        {[
          { icon: Cpu, text: "LangGraph Powered" },
          { icon: Zap, text: "Gemini AI" },
          { icon: Container, text: "Docker Native" },
        ].map((t) => (
          <span
            key={t.text}
            className="glass-panel inline-flex cursor-default items-center gap-1.5 rounded-full px-4 py-1.5 text-sm text-slate-700 transition-all duration-200 hover:scale-[1.02]"
          >
            <t.icon className="h-3.5 w-3.5 text-slate-500" /> {t.text}
          </span>
        ))}
      </div>

      <div className="relative mx-auto mt-12 max-w-xl">
        <div className="absolute -inset-1 rounded-2xl bg-gradient-to-r from-blue-400/30 to-emerald-400/30 opacity-40 blur-md" aria-hidden />
        <a
          href="#deploy"
          className="relative inline-flex cursor-pointer items-center justify-center rounded-xl bg-gradient-to-r from-blue-400 to-emerald-400 px-8 py-3.5 font-semibold text-white shadow-lg transition-all duration-200 hover:scale-[1.02] hover:shadow-xl hover:shadow-blue-200/80 active:scale-[0.98]"
        >
          Start deploying
        </a>
      </div>

      <div className="glass-panel-strong mx-auto mt-14 max-w-2xl overflow-hidden rounded-3xl p-1 text-left">
        <div className="flex items-center gap-2 border-b border-white/40 px-4 py-2">
          <span className="h-2.5 w-2.5 rounded-full bg-rose-300/90" />
          <span className="h-2.5 w-2.5 rounded-full bg-amber-200/90" />
          <span className="h-2.5 w-2.5 rounded-full bg-emerald-300/80" />
          <span className="ml-2 text-xs font-medium text-slate-500">deploy-agent.log</span>
        </div>
        <div className="terminal-scroll-mask max-h-36 overflow-hidden px-4 py-3">
          <div className="terminal-scroll-inner font-mono text-xs leading-relaxed">
            {loopLines.map((line, idx) => (
              <p key={`${idx}-${line.slice(0, 24)}`} className="border-b border-white/20 py-1.5 text-slate-700 last:border-0">
                {line}
              </p>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
