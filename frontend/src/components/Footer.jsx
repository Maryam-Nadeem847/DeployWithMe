import { Cloud, Github } from "lucide-react";

export default function Footer() {
  return (
    <footer className="border-t border-white/50 bg-white/30 py-8 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-4 text-sm text-slate-600 md:flex-row">
        <div className="flex items-center gap-2 font-bold text-slate-800">
          <Cloud className="h-5 w-5 text-sky-500" strokeWidth={1.5} aria-hidden />
          <span className="bg-gradient-to-r from-blue-500 to-teal-400 bg-clip-text text-transparent">
            DeployWithMe
          </span>
        </div>
        <p className="text-center leading-relaxed">Autonomous model deployment agent</p>
        <a
          href="#"
          className="inline-flex cursor-pointer items-center gap-1 text-slate-700 transition-colors hover:text-blue-500"
        >
          <Github className="h-4 w-4" /> GitHub
        </a>
      </div>
    </footer>
  );
}
