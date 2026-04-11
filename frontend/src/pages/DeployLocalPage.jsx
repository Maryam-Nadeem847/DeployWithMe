import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import DeployCard from "../components/DeployCard.jsx";
import DeployProgress from "../components/DeployProgress.jsx";
import ConfirmationCheckpoint from "../components/ConfirmationCheckpoint.jsx";
import SuccessResult from "../components/SuccessResult.jsx";
import { useDeployment } from "../hooks/useDeployment.js";

const healthClient = axios.create({ baseURL: "http://localhost:8080", timeout: 5000 });

export default function DeployLocalPage() {
  const navigate = useNavigate();
  const {
    jobId,
    status,
    job,
    error,
    cloudRoute,
    pendingCloudFiles,
    clearCloudRoute,
    startDeploy,
    confirmDeploy,
    reset,
  } = useDeployment();
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

  const busy = !!jobId && status !== "idle" && status !== "success" && status !== "failed";
  const showProgress = !!jobId && status !== "idle" && status !== "success" && status !== "failed";
  const showConfirm = status === "awaiting_confirmation";
  const showSuccess = status === "success" && job?.api_url;
  const errorShake = useMemo(() => Boolean(error), [error]);

  return (
    <div className="mx-auto max-w-6xl px-4 py-10">
      <section className="mb-8">
        <h1 className="text-3xl font-bold text-slate-800">Deploy Locally</h1>
        <p className="mt-2 text-slate-600">
          Deploy your model as a Dockerized FastAPI inference service on this machine.
        </p>
      </section>

      {dockerOk === false && (
        <div className="mx-auto mb-6 max-w-4xl">
          <div className="glass-panel rounded-2xl px-4 py-3 text-center text-sm text-slate-800">
            Docker does not appear to be running. Start Docker Desktop, then refresh this page.
          </div>
        </div>
      )}

      {error && (
        <div className="mx-auto mb-6 max-w-xl">
          <div
            className={`glass-panel rounded-2xl px-4 py-3 text-sm text-slate-800 ${
              errorShake ? "animate-shake-once border border-red-300/80" : "border border-red-200/60"
            }`}
          >
            {error}
          </div>
        </div>
      )}

      {cloudRoute && pendingCloudFiles?.modelFile && (
        <div className="mx-auto mb-6 max-w-xl rounded-2xl border border-white/60 bg-white/50 p-6 shadow-[0_8px_32px_0_rgba(31,38,135,0.07)] backdrop-blur-xl">
          <p className="text-sm font-medium text-slate-800">
            Your model is{" "}
            <span className="font-bold text-slate-900">
              {cloudRoute.file_size_mb != null ? `${cloudRoute.file_size_mb}` : "—"}MB
            </span>{" "}
            — automatically routing to HuggingFace Spaces for better performance.
          </p>
          {cloudRoute.reason ? (
            <p className="mt-2 text-sm text-slate-600">{cloudRoute.reason}</p>
          ) : null}
          <button
            type="button"
            onClick={() => {
              navigate("/deploy/cloud", {
                state: {
                  modelFile: pendingCloudFiles.modelFile,
                  reqFile: pendingCloudFiles.requirementsFile,
                  suggestedSpaceName: cloudRoute.suggested_space_name,
                  fileSizeMb: cloudRoute.file_size_mb,
                  reason: cloudRoute.reason,
                },
              });
              clearCloudRoute();
            }}
            className="btn-primary mt-4"
          >
            Continue to Cloud Deploy →
          </button>
        </div>
      )}

      <DeployCard onDeploy={onDeploy} disabled={busy} />
      <DeployProgress job={job} visible={showProgress} />

      {showConfirm && job && (
        <ConfirmationCheckpoint job={job} onConfirm={(ok) => confirmDeploy(ok)} />
      )}

      {showSuccess && job && <SuccessResult job={job} onDeployAnother={reset} />}

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
    </div>
  );
}
