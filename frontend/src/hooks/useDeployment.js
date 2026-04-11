import { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8080",
  timeout: 120000,
});

const POLL_MS = 2000;

export function useDeployment() {
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState("idle");
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  /** When POST /api/deploy returns route: "cloud", show redirect UI */
  const [cloudRoute, setCloudRoute] = useState(null);
  const [pendingCloudFiles, setPendingCloudFiles] = useState(null);
  const pollRef = useRef(null);

  const stopPoll = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const fetchStatus = useCallback(async (id) => {
    try {
      const { data } = await API.get(`/api/status/${id}`);
      setJob(data);
      setStatus(data.status || "unknown");

      if (data.status === "success" || data.status === "failed") {
        stopPoll();
      }
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
      stopPoll();
    }
  }, [stopPoll]);

  useEffect(() => {
    if (!jobId) return;
    fetchStatus(jobId);
    stopPoll();
    pollRef.current = setInterval(() => fetchStatus(jobId), POLL_MS);
    return () => stopPoll();
  }, [jobId, fetchStatus, stopPoll]);

  const startDeploy = useCallback(
    async (modelFile, requirementsFile) => {
      setError(null);
      setJob(null);
      setCloudRoute(null);
      setPendingCloudFiles(null);
      setStatus("starting");
      const form = new FormData();
      form.append("model_file", modelFile);
      if (requirementsFile) form.append("requirements_file", requirementsFile);

      try {
        const { data } = await API.post("/api/deploy", form, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        if (data?.route === "cloud") {
          setCloudRoute({
            reason: data.reason,
            suggested_space_name: data.suggested_space_name,
            file_size_mb: data.file_size_mb,
          });
          setPendingCloudFiles({
            modelFile,
            requirementsFile: requirementsFile || null,
          });
          setJobId(null);
          setStatus("idle");
          return;
        }
        setJobId(data.job_id);
        setStatus("started");
      } catch (e) {
        const msg = e?.response?.data?.detail || e.message;
        setError(typeof msg === "string" ? msg : JSON.stringify(msg));
        setStatus("failed");
      }
    },
    []
  );

  const confirmDeploy = useCallback(
    async (confirmed) => {
      if (!jobId) return;
      try {
        await API.post(`/api/confirm/${jobId}`, { confirmed });
        if (!confirmed) {
          setStatus("failed");
          stopPoll();
        }
      } catch (e) {
        setError(e?.response?.data?.detail || e.message);
      }
    },
    [jobId, stopPoll]
  );

  const clearCloudRoute = useCallback(() => {
    setCloudRoute(null);
    setPendingCloudFiles(null);
  }, []);

  const reset = useCallback(() => {
    stopPoll();
    setJobId(null);
    setJob(null);
    setStatus("idle");
    setError(null);
    setCloudRoute(null);
    setPendingCloudFiles(null);
  }, [stopPoll]);

  return {
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
    stopPoll,
  };
}
