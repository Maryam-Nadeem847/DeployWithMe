import { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8080",
  timeout: 120000,
});

const POLL_MS = 5000;

const STEP_LABELS = [
  "Validating",
  "Creating Space",
  "Uploading Files",
  "Building",
  "Live",
];

/** Map backend `step` string to pill index 0..4 */
function pillIndexFromApiStep(stepRaw, statusRaw) {
  const st = String(statusRaw || "").toLowerCase();
  if (st === "success") return STEP_LABELS.length - 1;
  const s = String(stepRaw || "").toLowerCase();
  if (s.includes("valid")) return 0;
  if (s.includes("creat") || (s.includes("space") && !s.includes("upload"))) return 1;
  if (s.includes("upload")) return 2;
  if (s.includes("build")) return 3;
  if (s.includes("live")) return 4;
  return 0;
}

export function useCloudDeploy() {
  /** @type {'upload' | 'checkpoint' | 'deploying' | 'success' | 'failed'} */
  const [step, setStep] = useState("upload");
  const [jobId, setJobId] = useState(null);
  const [suggestedName, setSuggestedName] = useState("");
  const [confirmedName, setConfirmedName] = useState("");
  const [hfToken, setHfToken] = useState("");
  const [framework, setFramework] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [apiStatus, setApiStatus] = useState("");
  const [apiStep, setApiStep] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const [modelFile, setModelFile] = useState(null);
  const [modelType, setModelType] = useState("");
  const [modelTypeDesc, setModelTypeDesc] = useState("");
  const [inputSpecAuto, setInputSpecAuto] = useState(null);
  const [inputSpec, setInputSpec] = useState(null);
  const pollRef = useRef(null);

  const stopPoll = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const pollStatus = useCallback(async () => {
    const id = jobId;
    if (!id) return;
    try {
      const { data } = await API.get(`/api/status/cloud/${id}`);
      const st = String(data?.status || "").toLowerCase();
      setApiStatus(data?.status || "");
      setApiStep(data?.step || "");
      setStatusMessage(data?.message || "");

      const res = data?.result;
      if (res && typeof res === "object") {
        setResult({
          space_url: res.space_url || res.spaceUrl || data.space_url,
          api_url: res.api_url || res.apiUrl || data.api_url,
          space_name: res.space_name || res.spaceName || data.space_name,
        });
      } else if (data.space_url || data.api_url) {
        setResult({
          space_url: data.space_url,
          api_url: data.api_url,
          space_name: data.space_name,
        });
      }

      if (st === "success") {
        stopPoll();
        setStep("success");
      } else if (st === "failed") {
        stopPoll();
        setStep("failed");
        setError(data?.message || "Cloud deployment failed");
      }
    } catch (e) {
      stopPoll();
      setStep("failed");
      setError(e?.response?.data?.detail || e.message);
    }
  }, [jobId, stopPoll]);

  useEffect(() => {
    if (step !== "deploying" || !jobId) {
      stopPoll();
      return;
    }
    pollStatus();
    stopPoll();
    pollRef.current = setInterval(pollStatus, POLL_MS);
    return () => stopPoll();
  }, [step, jobId, pollStatus, stopPoll]);

  const hydrateFromRedirect = useCallback((file, suggestedSpaceName) => {
    if (file instanceof File) setModelFile(file);
    if (typeof suggestedSpaceName === "string" && suggestedSpaceName) {
      setSuggestedName(suggestedSpaceName);
      setConfirmedName(suggestedSpaceName);
    }
  }, []);

  const submitUpload = useCallback(async () => {
    if (!modelFile) {
      setError("Select a model file.");
      return;
    }
    const token = hfToken.trim();
    if (!token) {
      setError("Enter your Hugging Face token.");
      return;
    }
    setError(null);
    try {
      const form = new FormData();
      form.append("model_file", modelFile);
      form.append("hf_token", token);
      const { data } = await API.post("/api/deploy/cloud", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const id = data?.job_id;
      if (!id) {
        setError("No job_id returned from server.");
        return;
      }
      setJobId(id);
      const sug = data?.suggested_space_name ?? data?.suggestedSpaceName ?? "";
      setSuggestedName(sug);
      setConfirmedName((prev) => (sug ? sug : prev));
      if (data?.framework) setFramework(data.framework);
      const auto = data?.input_spec_auto ?? data?.inputSpecAuto ?? null;
      setInputSpecAuto(auto);
      setInputSpec(auto);
      setStep("checkpoint");
    } catch (e) {
      const msg = e?.response?.data?.detail || e.message;
      setError(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
  }, [modelFile, hfToken]);

  const confirmDeploy = useCallback(async () => {
    const id = jobId;
    const name = (confirmedName || suggestedName || "").trim();
    const token = hfToken.trim();
    if (!id || !name || !token) {
      setError("Space name and token are required.");
      return;
    }
    if (!modelType) {
      setError("Please select a model type.");
      return;
    }
    if (modelType === "Other" && !modelTypeDesc.trim()) {
      setError("Please describe your model.");
      return;
    }
    setError(null);
    try {
      await API.post(`/api/deploy/cloud/confirm/${id}`, {
        confirmed_space_name: name,
        hf_token: token,
        model_type: modelType,
        model_type_description: modelType === "Other" ? modelTypeDesc.trim() : null,
        input_spec: inputSpec,
      });
      setStep("deploying");
      setStatusMessage("");
      setApiStatus("");
      setApiStep("");
      setResult(null);
    } catch (e) {
      const msg = e?.response?.data?.detail || e.message;
      setError(typeof msg === "string" ? msg : JSON.stringify(msg));
      setStep("failed");
    }
  }, [jobId, confirmedName, suggestedName, hfToken, modelType, modelTypeDesc, inputSpec]);

  const reset = useCallback(() => {
    stopPoll();
    setStep("upload");
    setJobId(null);
    setSuggestedName("");
    setConfirmedName("");
    setHfToken("");
    setFramework("");
    setStatusMessage("");
    setApiStatus("");
    setApiStep("");
    setResult(null);
    setError(null);
    setModelFile(null);
    setModelType("");
    setModelTypeDesc("");
    setInputSpecAuto(null);
    setInputSpec(null);
  }, [stopPoll]);

  const activePillIndex =
    step === "deploying"
      ? pillIndexFromApiStep(apiStep, apiStatus)
      : step === "success"
        ? STEP_LABELS.length - 1
        : -1;

  return {
    step,
    jobId,
    suggestedName,
    confirmedName,
    setConfirmedName,
    hfToken,
    setHfToken,
    framework,
    modelFile,
    setModelFile,
    modelType,
    setModelType,
    modelTypeDesc,
    setModelTypeDesc,
    inputSpec,
    setInputSpec,
    inputSpecAuto,
    statusMessage,
    apiStatus,
    apiStep,
    result,
    error,
    setError,
    stepLabels: STEP_LABELS,
    activePillIndex,
    submitUpload,
    confirmDeploy,
    pollStatus,
    hydrateFromRedirect,
    reset,
  };
}
