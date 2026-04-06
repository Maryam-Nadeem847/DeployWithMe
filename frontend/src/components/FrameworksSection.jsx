const items = [
  {
    name: "sklearn",
    label: "Classification & Regression",
    cls: "border-blue-300/60 bg-white/45 text-blue-900",
  },
  {
    name: "XGBoost",
    label: "Gradient Boosting",
    cls: "border-emerald-300/60 bg-white/45 text-emerald-900",
  },
  {
    name: "PyTorch",
    label: "Deep Learning",
    cls: "border-orange-300/60 bg-white/45 text-orange-900",
  },
  {
    name: "TensorFlow",
    label: "Neural Networks",
    cls: "border-green-300/60 bg-white/45 text-green-900",
  },
  {
    name: "ONNX",
    label: "Optimized Inference",
    cls: "border-purple-300/60 bg-white/45 text-purple-900",
  },
];

export default function FrameworksSection() {
  return (
    <section className="mx-auto max-w-6xl px-4 pb-20">
      <h2 className="bg-gradient-to-r from-slate-700 to-indigo-500 bg-clip-text text-center text-2xl font-bold text-transparent">Supported Frameworks</h2>
      <div className="mt-10 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
        {items.map((f) => (
          <div
            key={f.name}
            className="glass-panel hover:scale-[1.01] cursor-default p-5 text-center transition-transform duration-200"
          >
            <span
              className={`inline-block rounded-full border px-3 py-1 text-xs font-bold backdrop-blur-sm ${f.cls}`}
            >
              {f.name}
            </span>
            <p className="mt-3 text-sm leading-relaxed text-slate-600">{f.label}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
