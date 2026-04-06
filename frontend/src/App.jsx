import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import Navbar from "./components/Navbar.jsx";
import HeroSection from "./components/HeroSection.jsx";
import DeployCard from "./components/DeployCard.jsx";
import DeployProgress from "./components/DeployProgress.jsx";
import ConfirmationCheckpoint from "./components/ConfirmationCheckpoint.jsx";
import SuccessResult from "./components/SuccessResult.jsx";
import ActiveDeployments from "./components/ActiveDeployments.jsx";
import HowItWorks from "./components/HowItWorks.jsx";
import FrameworksSection from "./components/FrameworksSection.jsx";
import Footer from "./components/Footer.jsx";
import { useDeployment } from "./hooks/useDeployment.js";
import { useDeployments } from "./hooks/useDeployments.js";

const healthClient = axios.create({ baseURL: "http://localhost:8080", timeout: 5000 });

/** Deterministic particle layout (stable across renders, ethereal dust) */
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
  const { jobId, status, job, error, startDeploy, confirmDeploy, reset } = useDeployment();
  const { items, refresh, loading: deploymentsLoading } = useDeployments();
  const [dockerOk, setDockerOk] = useState(null);

  useEffect(() => {
    healthClient
      .get("/api/health")
      .then((r) => setDockerOk(r.data.docker === "running"))
      .catch(() => setDockerOk(false));
  }, []);

  const onDeploy = useCallback(
    async (modelFile, reqFile) => {
      await startDeploy(modelFile, reqFile);
    },
    [startDeploy]
  );

  const busy =
    !!jobId &&
    status !== "idle" &&
    status !== "success" &&
    status !== "failed";

  const showProgress =
    !!jobId && status !== "idle" && status !== "success" && status !== "failed";
  const showConfirm = status === "awaiting_confirmation";
  const showSuccess = status === "success" && job?.api_url;

  const errorShake = useMemo(() => Boolean(error), [error]);

  return (
    <div className="relative min-h-screen">
      {/* Fixed full-viewport gradient mesh */}
      <div
        className="fixed inset-0 z-0 bg-gradient-to-br from-[#e0f2fe] via-[#fce7f3] to-[#e0e7ff]"
        aria-hidden
      />

      {/* Four animated blobs */}
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

      {/* Floating particles */}
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
        {dockerOk === false && (
          <div className="mx-auto max-w-4xl px-4 pt-3">
            <div className="glass-panel rounded-2xl px-4 py-3 text-center text-sm text-slate-800">
              Docker does not appear to be running. Start Docker Desktop, then refresh this page.
            </div>
          </div>
        )}
        <HeroSection />

        {error && (
          <div className="mx-auto mb-6 max-w-xl px-4">
            <div
              className={`glass-panel rounded-2xl px-4 py-3 text-sm text-slate-800 ${
                errorShake ? "animate-shake-once border border-red-300/80" : "border border-red-200/60"
              }`}
            >
              {error}
            </div>
          </div>
        )}

        <DeployCard onDeploy={onDeploy} disabled={busy} />

        <DeployProgress job={job} visible={showProgress} />

        {showConfirm && job && (
          <ConfirmationCheckpoint job={job} onConfirm={(ok) => confirmDeploy(ok)} />
        )}

        {showSuccess && job && (
          <SuccessResult
            job={job}
            onDeployAnother={() => {
              reset();
              refresh();
            }}
          />
        )}

        {status === "failed" && job && !showConfirm && (
          <div className="mx-auto max-w-3xl px-4 pb-12">
            <div className="glass-card hover:scale-[1.01] cursor-default p-6 text-slate-800 transition-transform duration-200">
              <p className="font-bold text-slate-800">Deployment failed</p>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">
                {job.error || "Unknown error"}
              </p>
              <button type="button" onClick={() => reset()} className="btn-primary mt-4">
                Try Again
              </button>
            </div>
          </div>
        )}

        <ActiveDeployments
          items={items}
          loading={deploymentsLoading}
          onRefresh={refresh}
        />
        <HowItWorks />
        <FrameworksSection />
        <Footer />
      </div>
    </div>
  );
}
