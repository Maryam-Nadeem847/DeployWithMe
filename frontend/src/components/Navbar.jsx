import { useState } from "react";
import { Cloud, Menu, X } from "lucide-react";
import { NavLink } from "react-router-dom";

const navItems = [
  { to: "/", label: "Home" },
  { to: "/deploy", label: "Deploy Local" },
  { to: "/deploy/cloud", label: "Deploy Cloud" },
  { to: "/dashboard", label: "Dashboard" },
  { to: "/docs", label: "Docs" },
];

function navClass(isActive) {
  return `text-sm font-medium transition-colors ${
    isActive ? "text-blue-500 font-semibold" : "text-slate-700 hover:text-blue-500"
  }`;
}

export default function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 border-b border-white/50 bg-white/30 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <NavLink
          to="/"
          className="flex items-center gap-2 transition-opacity hover:opacity-90"
          onClick={() => setOpen(false)}
        >
          <Cloud className="h-6 w-6 text-sky-500" strokeWidth={1.5} aria-hidden />
          <span className="bg-gradient-to-r from-blue-500 to-teal-400 bg-clip-text text-xl font-bold text-transparent">
            DeployWithMe
          </span>
        </NavLink>

        <nav className="hidden items-center gap-5 md:flex">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => navClass(isActive)}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <button
          type="button"
          className="btn-ghost !px-3 !py-2 md:hidden"
          onClick={() => setOpen((v) => !v)}
          aria-label="Toggle menu"
          aria-expanded={open}
        >
          {open ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
        </button>
      </div>

      {open && (
        <div className="mx-4 mb-3 rounded-2xl border border-white/60 bg-white/50 p-3 shadow-[0_8px_32px_0_rgba(31,38,135,0.07)] backdrop-blur-xl md:hidden">
          <div className="flex flex-col gap-2">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                onClick={() => setOpen(false)}
                className={({ isActive }) =>
                  `rounded-xl px-3 py-2 ${navClass(isActive)} ${
                    isActive ? "bg-white/60" : "hover:bg-white/40"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </div>
        </div>
      )}
    </header>
  );
}
