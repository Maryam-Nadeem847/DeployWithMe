import { Code2, Container, FileText, ShieldCheck } from "lucide-react";

export default function DocsPage() {
  const cards = [
    {
      icon: FileText,
      title: "Supported Formats",
      body: ".pkl, .pickle, .joblib, .sav, .pt, .pth, .onnx, .h5, .keras",
    },
    {
      icon: Container,
      title: "Local Deployment",
      body: "Agent generates FastAPI + Docker bundle, builds image, runs container, checks health.",
    },
    {
      icon: Code2,
      title: "Endpoints",
      body: "Deployed model APIs expose /health and /predict. Dashboard includes live test requests.",
    },
    {
      icon: ShieldCheck,
      title: "Human-in-the-Loop",
      body: "Before final run, you can review details and confirm or cancel deployment.",
    },
  ];

  return (
    <div className="mx-auto max-w-6xl px-4 py-10">
      <section className="mb-8">
        <h1 className="text-3xl font-bold text-slate-800">Documentation</h1>
        <p className="mt-2 text-slate-600">
          Quick reference for deployment flows and model serving behavior.
        </p>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        {cards.map((c) => (
          <article key={c.title} className="glass-panel hover:scale-[1.01] p-6">
            <c.icon className="h-6 w-6 text-blue-500" />
            <h2 className="mt-3 text-lg font-bold text-slate-800">{c.title}</h2>
            <p className="mt-2 text-sm leading-relaxed text-slate-600">{c.body}</p>
          </article>
        ))}
      </section>
    </div>
  );
}
