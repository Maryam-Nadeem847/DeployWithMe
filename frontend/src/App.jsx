import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar.jsx";
import Footer from "./components/Footer.jsx";
import HomePage from "./pages/HomePage.jsx";
import DeployLocalPage from "./pages/DeployLocalPage.jsx";
import DeployCloudPage from "./pages/DeployCloudPage.jsx";
import DashboardPage from "./pages/DashboardPage.jsx";
import DocsPage from "./pages/DocsPage.jsx";

const FLOAT_PARTICLES = Array.from({ length: 38 }, (_, i) => {
  const s = Math.sin(i * 12.9898) * 43758.5453;
  const t = s - Math.floor(s);
  const u = Math.sin(i * 78.233) * 43758.5453;
  const v = u - Math.floor(u);
  return {
    id: i,
    left: `${4 + t * 92}%`,
    top: `${6 + v * 88}%`,
    size: i % 4 === 0 ? "w-1.5 h-1.5" : "w-1 h-1",
    duration: `${6 + (i % 5) * 0.85}s`,
    delay: `${(i % 9) * 0.55}s`,
    opacity: 0.25 + (i % 4) * 0.12,
  };
});

export default function App() {
  return (
    <Router>
      <div className="relative min-h-screen">
        <div
          className="fixed inset-0 z-0 bg-gradient-to-br from-[#e0f2fe] via-[#fce7f3] to-[#e0e7ff]"
          aria-hidden
        />
        <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden" aria-hidden>
          <div className="absolute -left-20 -top-24 h-[22rem] w-[22rem] rounded-full bg-blue-200 opacity-40 blur-3xl animate-blob-pulse" />
          <div
            className="absolute -right-16 -top-20 h-[20rem] w-[20rem] rounded-full bg-pink-200 opacity-40 blur-3xl animate-blob-pulse"
            style={{ animationDelay: "2s" }}
          />
          <div
            className="absolute -bottom-28 -left-12 h-[24rem] w-[24rem] rounded-full bg-emerald-200 opacity-30 blur-3xl animate-blob-pulse"
            style={{ animationDelay: "4s" }}
          />
          <div
            className="absolute -bottom-20 -right-16 h-[21rem] w-[21rem] rounded-full bg-indigo-200 opacity-40 blur-3xl animate-blob-pulse"
            style={{ animationDelay: "6s" }}
          />
        </div>
        <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden" aria-hidden>
          {FLOAT_PARTICLES.map((p) => (
            <span
              key={p.id}
              className={`absolute rounded-full bg-white/80 shadow-[0_0_6px_rgba(147,197,253,0.5)] animate-float-particle ${p.size}`}
              style={{
                left: p.left,
                top: p.top,
                animationDuration: p.duration,
                animationDelay: p.delay,
                opacity: p.opacity,
              }}
            />
          ))}
        </div>

        <div className="relative z-10 animate-page-in">
          <Navbar />
          <main className="min-h-[calc(100vh-10rem)]">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/deploy" element={<DeployLocalPage />} />
              <Route path="/deploy/cloud" element={<DeployCloudPage />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/docs" element={<DocsPage />} />
            </Routes>
          </main>
          <Footer />
        </div>
      </div>
    </Router>
  );
}
