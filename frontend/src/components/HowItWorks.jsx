import { ArrowRight, CheckCircle, Cpu, Rocket, Search, UploadCloud } from "lucide-react";

const steps = [
  {
    icon: UploadCloud,
    accent: "from-sky-400 to-blue-400",
    title: "Upload Model",
    text: "Upload your trained model file. sklearn, PyTorch, TensorFlow, ONNX, and more.",
  },
  {
    icon: Search,
    accent: "from-violet-400 to-purple-400",
    title: "Auto Detection",
    text: "The agent detects framework and picks a compatible inference stack.",
  },
  {
    icon: Cpu,
    accent: "from-amber-400 to-orange-400",
    title: "Smart Build",
    text: "Generates Dockerfile and FastAPI app, then builds an isolated container.",
  },
  {
    icon: CheckCircle,
    accent: "from-emerald-400 to-teal-400",
    title: "Your Approval",
    text: "Review build details and confirm before the container goes live.",
  },
  {
    icon: Rocket,
    accent: "from-pink-400 to-rose-400",
    title: "Live API",
    text: "Health and predict endpoints go live. Test predictions from this UI.",
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="mx-auto max-w-6xl px-4 py-16">
      <h2 className="bg-gradient-to-r from-slate-700 to-indigo-500 bg-clip-text text-center text-2xl font-bold text-transparent">
        How It Works
      </h2>
      <div className="mt-10 flex flex-col gap-6 md:flex-row md:flex-wrap md:justify-center">
        {steps.map((s, i) => {
          const Icon = s.icon;
          return (
            <div
              key={s.title}
              className="glass-panel hover:scale-[1.01] flex min-w-[200px] flex-1 cursor-default flex-col items-center p-6 text-center transition-transform duration-200 md:max-w-[220px]"
            >
              <div
                className={`flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br ${s.accent} text-white shadow-md`}
              >
                <Icon className="h-6 w-6" strokeWidth={1.5} />
              </div>
              <h3 className="mt-3 font-bold text-slate-800">{s.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">{s.text}</p>
              {i < steps.length - 1 && (
                <ArrowRight className="mx-auto mt-4 hidden h-5 w-5 text-slate-400 md:block" />
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
