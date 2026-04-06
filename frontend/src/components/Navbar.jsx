import { Cloud, Github, BookOpen } from "lucide-react";

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 border-b border-white/50 bg-white/30 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <a href="#" className="flex cursor-pointer items-center gap-2 transition-opacity hover:opacity-90">
          <Cloud className="h-6 w-6 text-sky-500" strokeWidth={1.5} aria-hidden />
          <span className="bg-gradient-to-r from-blue-500 to-teal-400 bg-clip-text text-xl font-bold text-transparent">
            DeployWithMe
          </span>
        </a>
        <nav className="flex items-center gap-4 text-sm font-medium">
          <a
            href="#"
            className="inline-flex cursor-pointer items-center gap-1 text-slate-700 transition-colors hover:text-blue-500"
          >
            GitHub <Github className="h-4 w-4" />
          </a>
          <a
            href="#how-it-works"
            className="inline-flex cursor-pointer items-center gap-1 text-slate-700 transition-colors hover:text-blue-500"
          >
            Documentation <BookOpen className="h-4 w-4" />
          </a>
          <a
            href="#deploy"
            className="cursor-pointer rounded-full bg-gradient-to-r from-blue-400 to-emerald-400 px-5 py-2 font-semibold text-white shadow-sm transition-all duration-200 hover:shadow-lg hover:shadow-blue-200"
          >
            Deploy Model
          </a>
        </nav>
      </div>
    </header>
  );
}
