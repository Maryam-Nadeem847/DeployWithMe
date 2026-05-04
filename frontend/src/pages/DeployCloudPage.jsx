import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  CloudUpload,
  Copy,
  ExternalLink,
  Loader2,
  Pencil,
} from "lucide-react";
import { useCloudDeploy } from "../hooks/useCloudDeploy.js";

const FORMAT_PILLS = [".pkl", ".pt", ".h5", ".onnx", ".joblib"];

const MODEL_TYPE_OPTIONS = [
  "Image Classification",
  "Image Segmentation",
  "Tabular/Regression",
  "Text Classification",
  "Object Detection",
  "Time Series",
  "Other",
];

const CARD =
  "rounded-2xl border border-white/60 bg-white/50 p-6 shadow-[0_8px_32px_0_rgba(31,38,135,0.07)] backdrop-blur-xl";

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

const IMAGE_TYPES = new Set([
  "Image Classification",
  "Image Segmentation",
  "Object Detection",
]);

// Per-model-type schema descriptors. Each entry must match exactly what the
// generated app.py in hf_deployer.py emits — keep these in sync.
//   - request: literal JSON body shape the test UI must POST
//   - response: literal JSON shape(s) the API may return
//   - inputUI:  what input control(s) the test UI must render
//   - outputUI: how to render the response, including dual-shape feature detection
const MODEL_TYPE_SCHEMAS = {
  "Image Classification": {
    request:
      '{"image": "<base64-encoded image bytes, NO data: prefix>", "size": <int, optional>}',
    response:
      'TWO POSSIBLE SHAPES — handle BOTH at runtime via feature detection:\n' +
      '  RICH (when the model output looks like a probability distribution):\n' +
      '    {"predicted_class": <int>, "confidence": <float in 0..1>, "all_probabilities": [<float>, ...], "input_size": [H, W]}\n' +
      '  RAW (otherwise — e.g. logits or class-label outputs):\n' +
      '    {"prediction": <number array>, "input_size": [H, W]}',
    inputUI:
      'Render: a file picker (accept image/*) that converts the chosen file to base64 and STRIPS the "data:image/...;base64," prefix before sending. Plus a numeric "Resize" input defaulting to 224 with a visible warning underneath that reads exactly: "Value must be divisible by 32."',
    outputUI:
      'In JS, ALWAYS branch with: if ("predicted_class" in data) { /* rich */ } else if ("prediction" in data) { /* raw */ } else if ("error" in data) { /* error */ }.\n' +
      'RICH: show "Class: <predicted_class>" as a large headline, confidence as a percentage with 1 decimal (multiply by 100, e.g. 0.874 -> "87.4%"), and a collapsible <details> labeled "All probabilities" listing each index and probability.\n' +
      'RAW: render the prediction array readably (small <pre> is fine) and show input_size below.\n' +
      'NEVER assume both shapes coexist in the same response.',
  },
  "Object Detection": {
    request:
      '{"image": "<base64-encoded image bytes, NO data: prefix>"}  (NO size key — object detection models do their own preprocessing)',
    response:
      '{"detections": [{"class": <string>, "confidence": <float in 0..1>, "bbox": [x1, y1, x2, y2]}, ...]}\n' +
      'Do NOT expect "prediction", "input_size", "predicted_class", or "all_probabilities" keys — they are NOT present for Object Detection.',
    inputUI:
      'Render: a file picker (accept image/*) that converts the chosen file to base64 and STRIPS the "data:image/...;base64," prefix before sending. Do NOT include any resize input.',
    outputUI:
      'For each detection in data.detections, render a card showing: class name (bold), confidence as a percentage with 1 decimal (multiply by 100), and bbox as "[x1, y1, x2, y2]" with all four numbers rounded to integers.\n' +
      'If data.detections is an empty array, render the literal text "No objects detected." instead of an empty list.',
  },
  "Image Segmentation": {
    request:
      '{"image": "<base64-encoded image bytes, NO data: prefix>", "size": <int, optional>}',
    response:
      '{"mask_png_base64": "<base64-encoded PNG>", "shape": [<int>, ...], "input_size": [H, W]}',
    inputUI:
      'Render: a file picker (accept image/*) that converts the chosen file to base64 and STRIPS the "data:image/...;base64," prefix before sending. Plus a numeric "Resize" input defaulting to 256 with a visible warning underneath that reads exactly: "Value must be divisible by 32."',
    outputUI:
      'Render the mask as <img src={"data:image/png;base64," + data.mask_png_base64} style="max-width:100%"> so the user sees the predicted segmentation mask. Below it, show "shape: <data.shape>" and "input_size: <data.input_size>" as plain text.',
  },
  "Text Classification": {
    request: '{"text": "<string>"}',
    response:
      'TWO POSSIBLE SHAPES — handle BOTH at runtime:\n' +
      '  RICH: {"predicted_class": <int>, "confidence": <float in 0..1>, "all_probabilities": [<float>, ...]}\n' +
      '  RAW:  {"prediction": <value (string|number|array)>}',
    inputUI:
      'Render: a <textarea> labeled "Text" + a Submit button. Send {"text": <textarea value>}.',
    outputUI:
      'Branch on `if ("predicted_class" in data)`. RICH: show predicted_class + confidence% prominently with collapsible all_probabilities. RAW: show data.prediction (stringify if array).',
  },
  "Tabular/Regression": {
    request: '{"features": [<number>, <number>, ...]}',
    response:
      'TWO POSSIBLE SHAPES — handle BOTH at runtime:\n' +
      '  RICH (sklearn classifier with predict_proba, or DL model with softmax output):\n' +
      '    {"predicted_class": <int>, "confidence": <float>, "all_probabilities": [<float>, ...]}\n' +
      '  RAW (sklearn regressor, or sklearn classifier without predict_proba):\n' +
      '    {"prediction": <value (number|array)>}',
    inputUI:
      'Render: a text input for comma-separated floats (placeholder "1.2, 3.4, 5.6") + Submit. Parse to a number array and send {"features": [...]}.',
    outputUI:
      'Branch on `if ("predicted_class" in data)`. RICH: show predicted_class + confidence% + collapsible all_probabilities. RAW: show data.prediction (regressor returns scalar/array; classifier-without-proba returns class label).',
  },
  "Time Series": {
    request: '{"sequence": [<number>, <number>, ...]}',
    response:
      'TWO POSSIBLE SHAPES — handle BOTH at runtime:\n' +
      '  RICH: {"predicted_class": <int>, "confidence": <float>, "all_probabilities": [<float>, ...]}\n' +
      '  RAW:  {"prediction": <value>}',
    inputUI:
      'Render: a text input for comma-separated sequence values + Submit. Parse to a number array and send {"sequence": [...]}.',
    outputUI:
      'Branch on `if ("predicted_class" in data)`. RICH: predicted_class + confidence% + collapsible all_probabilities. RAW: show data.prediction.',
  },
  Other: {
    request: '{"data": "<string>"}',
    response:
      'Schema is unspecified. Treat the response as arbitrary JSON.',
    inputUI:
      'Render: a single <textarea> labeled "Data" + Submit button. Send {"data": <textarea value>}.',
    outputUI:
      'JSON.stringify the response and display it inside a <pre> block. Make no assumptions about keys.',
  },
};

// Same-origin proxy that forwards the test UI's request to the deployed HF
// Space. The generated UI lives in a sandboxed iframe with a null origin and
// cannot reliably hit hf.space directly across CORS preflight, so it always
// goes through this local endpoint instead.
const TEST_PROXY_URL = "http://localhost:8080/api/test-cloud-predict";

function buildGeminiPrompt({ apiUrl, modelType, modelTypeDesc }) {
  const schema = MODEL_TYPE_SCHEMAS[modelType] || MODEL_TYPE_SCHEMAS["Tabular/Regression"];
  const typeDescription =
    modelType === "Other"
      ? `Other — ${(modelTypeDesc || "").trim()}`
      : modelType;

  return `Generate a complete, self-contained, single-file HTML document (with ALL CSS and JavaScript inline) that serves as a test interface for an ML model deployed on Hugging Face Spaces.

Model type: ${typeDescription}

REQUEST FLOW — IMPORTANT, READ CAREFULLY:
This page runs inside a sandboxed iframe with a null origin. It MUST NOT call the Space directly — that fails with "Failed to fetch" because of cross-origin / iframe sandbox restrictions. Instead, EVERY prediction request goes to a same-origin local proxy that forwards the call server-side:

  POST ${TEST_PROXY_URL}
  Content-Type: application/json
  Body: {"api_url": "${apiUrl}", "payload": <MODEL_PAYLOAD>}

…where <MODEL_PAYLOAD> is EXACTLY this JSON shape (this is what the model itself expects):
${schema.request}

The proxy returns the model's /predict response verbatim with the same status code. On transport failure it returns a JSON envelope: {"error": "<string>", "type": "<string>"} with status 502, or {"error": "...", "status_code": <int>, "body": "..."} when the upstream returned non-JSON.

RESPONSE SCHEMA — what the model returns inside the proxied response body:
${schema.response}

INPUT UI:
${schema.inputUI}

OUTPUT UI:
${schema.outputUI}

Error handling:
- If the response JSON has an "error" key (string), render it prominently in red with the "type" field if present. Do NOT crash.
- On a non-2xx HTTP status, parse the body as JSON if possible and show the "error" / "type" / "traceback" / "body" fields; otherwise show the raw text.
- Use feature detection (if ("key" in data)) before reading any field. NEVER assume keys that are not in the RESPONSE SCHEMA above are present.
- Wrap fetch() in try/catch and show the message text on TypeError (network failure) — do not let an exception bubble up uncaught.

Hard requirements:
- Fully self-contained: no external CSS, no external JS, no CDNs, no <link>, no <script src=>. All styles and scripts inline. Only built-in browser APIs (fetch, FileReader, etc.).
- Call ${TEST_PROXY_URL} via fetch() POST with header Content-Type: application/json and body {"api_url": "${apiUrl}", "payload": {...}}. Do NOT call ${apiUrl} or ${apiUrl}/predict directly under any circumstance.
- Mobile-friendly responsive layout.
- Output ONLY the raw HTML starting with <!DOCTYPE html>. No markdown fences, no commentary.`;
}

const GEMINI_PRIMARY_MODEL = "gemini-2.5-flash";
const GEMINI_FALLBACK_MODEL = "gemini-2.5-flash-lite";

async function callGeminiOnce(prompt, apiKey, model) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
    model
  )}:generateContent?key=${encodeURIComponent(apiKey)}`;
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] }),
  });
}

async function extractHtmlFromGeminiResponse(res, modelLabel) {
  if (!res.ok) {
    const t = await res.text();
    throw new Error(
      `Gemini (${modelLabel}) error ${res.status}: ${t.slice(0, 300)}`
    );
  }
  const data = await res.json();
  let html = data?.candidates?.[0]?.content?.parts?.[0]?.text || "";
  html = html
    .replace(/^\s*```html\s*/i, "")
    .replace(/^\s*```\s*/i, "")
    .replace(/\s*```\s*$/i, "")
    .trim();
  if (!html) throw new Error(`Empty response from Gemini (${modelLabel}).`);
  return html;
}

const NORMALIZATION_OPTIONS = [
  { value: "div255", label: "/255 → [0,1]" },
  { value: "imagenet", label: "ImageNet stats" },
  { value: "minus1to1", label: "[-1, 1]" },
  { value: "raw", label: "Raw [0, 255]" },
];

const COLOR_ORDER_OPTIONS = [
  { value: "RGB", label: "RGB" },
  { value: "BGR", label: "BGR" },
];

const INTERPOLATION_OPTIONS = [
  { value: "BILINEAR", label: "Bilinear" },
  { value: "BICUBIC", label: "Bicubic" },
  { value: "NEAREST", label: "Nearest" },
  { value: "LANCZOS", label: "Lanczos" },
];

function normLabel(value) {
  return NORMALIZATION_OPTIONS.find((o) => o.value === value)?.label || value;
}

function channelsLabel(c) {
  if (c === 1) return "grayscale";
  if (c === 3) return "RGB";
  return `${c}-channel`;
}

function specWithDefaults(s) {
  return {
    height: Number(s?.height) || 224,
    width: Number(s?.width) || 224,
    channels: Number(s?.channels) || 3,
    channel_order: s?.channel_order || "NHWC",
    normalization: s?.normalization || "div255",
    channel_color_order: s?.channel_color_order || "RGB",
    interpolation: s?.interpolation || "BILINEAR",
  };
}

function InputSpecPanel({ spec, auto, onChange }) {
  const autoOk = !!auto && auto.auto_detected && !auto.dynamic;
  const [expanded, setExpanded] = useState(!autoOk);
  const current = specWithDefaults(spec || auto);
  const set = (key, value) => onChange({ ...current, [key]: value });
  const grayscale = current.channels === 1;

  return (
    <div className="rounded-xl border border-white/60 bg-white/40 p-4 backdrop-blur-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-slate-800">Input Spec</p>
          {autoOk ? (
            <p className="mt-0.5 flex items-center gap-1 text-xs font-medium text-emerald-600">
              <CheckCircle2 className="h-3.5 w-3.5" />
              Auto-detected from model
            </p>
          ) : (
            <p className="mt-0.5 flex items-center gap-1 text-xs font-medium text-amber-600">
              <AlertTriangle className="h-3.5 w-3.5" />
              Auto-detect couldn&apos;t determine these — please confirm
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={() => setExpanded((x) => !x)}
          className="btn-ghost text-xs"
        >
          {expanded ? "Collapse" : "Edit"}
        </button>
      </div>

      <div className="mt-3 grid grid-cols-1 gap-x-6 gap-y-1 text-sm text-slate-700 sm:grid-cols-2">
        <div>
          <span className="text-slate-500">Size:</span>{" "}
          <span className="font-medium">
            {current.width} × {current.height}
          </span>
        </div>
        <div>
          <span className="text-slate-500">Channels:</span>{" "}
          <span className="font-medium">
            {current.channels} ({channelsLabel(current.channels)})
          </span>
        </div>
        <div>
          <span className="text-slate-500">Normalization:</span>{" "}
          <span className="font-medium">{normLabel(current.normalization)}</span>
        </div>
        <div>
          <span className="text-slate-500">Channel order:</span>{" "}
          <span className="font-medium">{current.channel_color_order}</span>
        </div>
      </div>

      {expanded && (
        <div className="mt-4 space-y-4 border-t border-white/60 pt-4">
          <div className="grid grid-cols-2 gap-3">
            <label className="text-xs">
              <span className="block font-semibold text-slate-700">Width</span>
              <input
                type="number"
                min="1"
                value={current.width}
                onChange={(e) => set("width", Number(e.target.value) || 0)}
                className="mt-1 w-full rounded-lg border border-white/60 bg-white/50 px-2 py-1.5 text-sm text-slate-800 backdrop-blur-sm outline-none focus:ring-2 focus:ring-sky-200"
              />
            </label>
            <label className="text-xs">
              <span className="block font-semibold text-slate-700">Height</span>
              <input
                type="number"
                min="1"
                value={current.height}
                onChange={(e) => set("height", Number(e.target.value) || 0)}
                className="mt-1 w-full rounded-lg border border-white/60 bg-white/50 px-2 py-1.5 text-sm text-slate-800 backdrop-blur-sm outline-none focus:ring-2 focus:ring-sky-200"
              />
            </label>
            <label className="text-xs">
              <span className="block font-semibold text-slate-700">Channels</span>
              <select
                value={current.channels}
                onChange={(e) => {
                  const c = Number(e.target.value);
                  const next = { ...current, channels: c };
                  if (c === 1 && next.normalization === "imagenet") {
                    next.normalization = "div255";
                  }
                  onChange(next);
                }}
                className="mt-1 w-full rounded-lg border border-white/60 bg-white/50 px-2 py-1.5 text-sm text-slate-800 backdrop-blur-sm outline-none focus:ring-2 focus:ring-sky-200"
              >
                <option value={1}>1 (grayscale)</option>
                <option value={3}>3 (RGB)</option>
              </select>
            </label>
            <label className="text-xs">
              <span className="block font-semibold text-slate-700">
                Layout
              </span>
              <select
                value={current.channel_order}
                onChange={(e) => set("channel_order", e.target.value)}
                className="mt-1 w-full rounded-lg border border-white/60 bg-white/50 px-2 py-1.5 text-sm text-slate-800 backdrop-blur-sm outline-none focus:ring-2 focus:ring-sky-200"
              >
                <option value="NHWC">NHWC</option>
                <option value="NCHW">NCHW</option>
              </select>
            </label>
          </div>

          <div>
            <p className="text-xs font-semibold text-slate-700">Normalization</p>
            <div className="mt-1 grid grid-cols-1 gap-1.5 sm:grid-cols-2">
              {NORMALIZATION_OPTIONS.map((o) => {
                const disabled = grayscale && o.value === "imagenet";
                const tooltip = disabled
                  ? "ImageNet mean/std are defined for 3-channel RGB images and don't apply to single-channel inputs."
                  : undefined;
                return (
                  <label
                    key={o.value}
                    title={tooltip}
                    className={`flex items-center gap-2 rounded-lg border border-white/60 bg-white/40 px-2.5 py-1.5 text-xs ${
                      disabled
                        ? "cursor-not-allowed opacity-50"
                        : "cursor-pointer hover:bg-white/60"
                    }`}
                  >
                    <input
                      type="radio"
                      name="normalization"
                      disabled={disabled}
                      value={o.value}
                      checked={current.normalization === o.value}
                      onChange={(e) => set("normalization", e.target.value)}
                    />
                    {o.label}
                    {disabled && (
                      <span className="ml-auto text-[10px] text-slate-500">
                        3-channel only
                      </span>
                    )}
                  </label>
                );
              })}
            </div>
          </div>

          <div>
            <p className="text-xs font-semibold text-slate-700">Channel order</p>
            <div className="mt-1 flex flex-wrap gap-2">
              {COLOR_ORDER_OPTIONS.map((o) => (
                <label
                  key={o.value}
                  className="flex cursor-pointer items-center gap-2 rounded-lg border border-white/60 bg-white/40 px-2.5 py-1.5 text-xs hover:bg-white/60"
                >
                  <input
                    type="radio"
                    name="channelColorOrder"
                    value={o.value}
                    checked={current.channel_color_order === o.value}
                    onChange={(e) => set("channel_color_order", e.target.value)}
                  />
                  {o.label}
                </label>
              ))}
            </div>
          </div>

          <div>
            <p className="text-xs font-semibold text-slate-700">Interpolation</p>
            <select
              value={current.interpolation}
              onChange={(e) => set("interpolation", e.target.value)}
              className="mt-1 w-full rounded-lg border border-white/60 bg-white/50 px-2 py-1.5 text-sm text-slate-800 backdrop-blur-sm outline-none focus:ring-2 focus:ring-sky-200"
            >
              {INTERPOLATION_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
    </div>
  );
}

async function callGeminiWithFallback(prompt, apiKey) {
  const primary = await callGeminiOnce(prompt, apiKey, GEMINI_PRIMARY_MODEL);
  if (primary.status !== 503) {
    return extractHtmlFromGeminiResponse(primary, GEMINI_PRIMARY_MODEL);
  }
  // Primary model is overloaded — fall back to the lite model immediately.
  const fallback = await callGeminiOnce(prompt, apiKey, GEMINI_FALLBACK_MODEL);
  if (fallback.status === 503) {
    throw new Error("Gemini is currently busy. Please try again in a moment.");
  }
  return extractHtmlFromGeminiResponse(fallback, GEMINI_FALLBACK_MODEL);
}

export default function DeployCloudPage() {
  const location = useLocation();
  const {
    step,
    jobId,
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
    result,
    error,
    setError,
    stepLabels,
    activePillIndex,
    submitUpload,
    confirmDeploy,
    pollStatus,
    hydrateFromRedirect,
    reset,
  } = useCloudDeploy();

  const [drag, setDrag] = useState(false);
  const [copied, setCopied] = useState(false);
  const hydratedRef = useRef(false);

  const [generatingTestUI, setGeneratingTestUI] = useState(false);
  const [generatedTestHTML, setGeneratedTestHTML] = useState("");
  const [testGenError, setTestGenError] = useState("");

  useEffect(() => {
    hydratedRef.current = false;
  }, [location.pathname]);

  useEffect(() => {
    if (hydratedRef.current) return;
    const st = location.state;
    if (!st) return;
    if (st.modelFile instanceof File) {
      hydrateFromRedirect(st.modelFile, st.suggestedSpaceName || st.suggested_space_name || "");
      hydratedRef.current = true;
    }
  }, [location.state, hydrateFromRedirect]);

  const previewSlug = (confirmedName || "").trim() || "your-space-name";
  const previewLine = `huggingface.co/spaces/<your-username>/${previewSlug.replace(/^\/+|\/+$/g, "")}`;

  const spaceUrl = result?.space_url || "";
  const apiUrl = result?.api_url || "";

  const copyPrimary = () => {
    const u = spaceUrl || apiUrl;
    if (!u) return;
    navigator.clipboard.writeText(u);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const triggerTestUI = async () => {
    setTestGenError("");
    setGeneratedTestHTML("");
    if (!apiUrl) {
      setTestGenError("No deployed API URL available.");
      return;
    }
    if (!modelType) {
      setTestGenError("Model type was not captured at deployment time.");
      return;
    }
    const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
    if (!apiKey) {
      setTestGenError(
        "VITE_GEMINI_API_KEY is not set. Add it to frontend/.env and restart the dev server."
      );
      return;
    }

    setGeneratingTestUI(true);
    try {
      const prompt = buildGeminiPrompt({ apiUrl, modelType, modelTypeDesc });
      const html = await callGeminiWithFallback(prompt, apiKey);
      setGeneratedTestHTML(html);
    } catch (e) {
      setTestGenError(e?.message || String(e));
    } finally {
      setGeneratingTestUI(false);
    }
  };

  return (
    <div className="mx-auto max-w-2xl px-4 py-10">
      <section className="mb-8">
        <h1 className="text-3xl font-bold text-slate-800">Deploy to Cloud</h1>
        <p className="mt-2 text-slate-600">
          Hugging Face Spaces deployment with a quick name confirmation step.
        </p>
      </section>

      {framework && (step === "checkpoint" || step === "deploying") && (
        <p className="mb-4 text-sm text-slate-600">
          Framework: <span className="font-semibold text-slate-800">{framework}</span>
          {jobId ? (
            <span className="ml-2 text-slate-400">· Job {jobId}</span>
          ) : null}
        </p>
      )}

      {step === "upload" && (
        <div className={`${CARD} space-y-6`}>
          <h2 className="text-lg font-bold text-slate-800">Step 1 — Upload &amp; token</h2>

          <div
            role="button"
            tabIndex={0}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              const f = e.dataTransfer.files?.[0];
              if (f) setModelFile(f);
            }}
            onClick={() => document.getElementById("cloud-model-input")?.click()}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") document.getElementById("cloud-model-input")?.click();
            }}
            className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-12 transition-all ${
              drag
                ? "border-blue-400 bg-blue-50/30"
                : "border-blue-300/60 bg-white/20 hover:border-blue-400 hover:bg-blue-50/30"
            }`}
          >
            <CloudUpload className="h-10 w-10 text-sky-500" />
            <p className="mt-3 font-semibold text-slate-800">Drop your model file here</p>
            <p className="text-sm text-slate-600">or click to browse</p>
            <div className="mt-4 flex flex-wrap justify-center gap-2">
              {FORMAT_PILLS.map((ext) => (
                <span
                  key={ext}
                  className="rounded-full border border-white/70 bg-white/50 px-3 py-1 text-xs text-slate-600 backdrop-blur-sm"
                >
                  {ext}
                </span>
              ))}
            </div>
            <input
              id="cloud-model-input"
              type="file"
              className="hidden"
              accept=".joblib,.pkl,.pickle,.pt,.pth,.onnx,.h5,.keras"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) setModelFile(f);
              }}
            />
          </div>
          {modelFile && (
            <p className="text-sm text-slate-700">
              <span className="font-semibold">{modelFile.name}</span> ({formatBytes(modelFile.size)})
            </p>
          )}

          <div>
            <label className="text-sm font-semibold text-slate-800">Hugging Face token</label>
            <p className="mt-1 text-sm text-slate-500">Get token at huggingface.co/settings/tokens</p>
            <input
              type="password"
              value={hfToken}
              onChange={(e) => setHfToken(e.target.value)}
              placeholder="hf_..."
              className="mt-2 w-full rounded-xl border border-white/60 bg-white/50 px-3 py-2 text-sm text-slate-800 backdrop-blur-sm outline-none focus:ring-2 focus:ring-sky-200"
            />
          </div>

          {error && <p className="text-sm text-red-600">{error}</p>}

          <button
            type="button"
            onClick={() => {
              setError(null);
              submitUpload();
            }}
            className="btn-primary w-full sm:w-auto"
          >
            Next →
          </button>
        </div>
      )}

      {step === "checkpoint" && (
        <div className={`${CARD} space-y-5`}>
          <h2 className="text-lg font-bold text-slate-800">☁️ Confirm Your Space Name</h2>

          <div>
            <label className="text-sm font-semibold text-slate-800">Space Name</label>
            <div className="relative mt-1">
              <input
                type="text"
                value={confirmedName}
                onChange={(e) => setConfirmedName(e.target.value)}
                className="w-full rounded-xl border border-white/60 bg-white/50 py-2.5 pl-3 pr-10 text-sm text-slate-800 backdrop-blur-sm outline-none focus:ring-2 focus:ring-sky-200"
                placeholder="mymodel-deploy"
                autoComplete="off"
              />
              <Pencil
                className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
                aria-hidden
              />
            </div>
          </div>

          <div className="rounded-xl border border-white/50 bg-black/5 px-4 py-3 text-sm text-slate-700 backdrop-blur-sm">
            <p className="font-medium text-slate-800">Your model will be live at:</p>
            <p className="mt-1 break-all font-mono text-slate-600">{previewLine}</p>
          </div>

          <div>
            <label className="text-sm font-semibold text-slate-800">Model type</label>
            <p className="mt-1 text-xs text-slate-500">
              Required — drives the generated /predict schema and Gradio input widget.
            </p>
            <div className="mt-2 grid max-h-64 grid-cols-1 gap-2 overflow-y-auto pr-1 sm:grid-cols-2">
              {MODEL_TYPE_OPTIONS.map((t) => (
                <label
                  key={t}
                  className={`flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-2 text-sm backdrop-blur-sm transition-colors ${
                    modelType === t
                      ? "border-sky-300 bg-sky-50/70 text-slate-800"
                      : "border-white/60 bg-white/40 text-slate-700 hover:bg-white/60"
                  }`}
                >
                  <input
                    type="radio"
                    name="deployModelType"
                    value={t}
                    checked={modelType === t}
                    onChange={(e) => setModelType(e.target.value)}
                  />
                  {t}
                </label>
              ))}
            </div>
            {modelType === "Other" && (
              <textarea
                value={modelTypeDesc}
                onChange={(e) => setModelTypeDesc(e.target.value)}
                placeholder="Describe your model — inputs, outputs, expected request shape…"
                rows={3}
                className="mt-3 w-full rounded-xl border border-white/60 bg-white/50 p-3 text-sm text-slate-800 backdrop-blur-sm outline-none focus:ring-2 focus:ring-sky-200"
              />
            )}
          </div>

          {IMAGE_TYPES.has(modelType) && (
            <InputSpecPanel
              spec={inputSpec}
              auto={inputSpecAuto}
              onChange={setInputSpec}
            />
          )}

          {error && <p className="text-sm text-red-600">{error}</p>}

          <button
            type="button"
            onClick={() => confirmDeploy()}
            disabled={
              !modelType || (modelType === "Other" && !modelTypeDesc.trim())
            }
            className="btn-primary w-full disabled:cursor-not-allowed disabled:opacity-50"
          >
            ✅ Confirm &amp; Deploy
          </button>
        </div>
      )}

      {step === "deploying" && (
        <div className={`${CARD} space-y-6`}>
          <h2 className="text-lg font-bold text-slate-800">Deployment in progress</h2>

          <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
            {stepLabels.map((label, idx) => {
              const done = idx < activePillIndex;
              const active = idx === activePillIndex;
              const cls = done
                ? "border-emerald-300/60 bg-gradient-to-r from-emerald-300 to-teal-300 text-white"
                : active
                  ? "border-blue-300/60 bg-gradient-to-r from-blue-400 to-teal-400 text-white shadow-lg shadow-blue-300/40"
                  : "border-white/60 bg-white/30 text-slate-500";
              return (
                <div
                  key={label}
                  className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-xs font-semibold ${cls}`}
                >
                  {done && <Check className="h-3.5 w-3.5" strokeWidth={3} />}
                  {active && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                  {label}
                </div>
              );
            })}
          </div>

          <p className="text-sm text-slate-600">
            {statusMessage || "Starting…"}
          </p>

          <button type="button" onClick={() => pollStatus()} className="btn-ghost text-xs">
            Refresh status
          </button>

          {error && <p className="text-sm text-red-600">{error}</p>}
        </div>
      )}

      {step === "failed" && (
        <div className={`${CARD} space-y-4`}>
          <h2 className="text-lg font-bold text-slate-800">Deployment failed</h2>
          <p className="text-sm text-slate-600">{error || "Something went wrong."}</p>
          <button type="button" onClick={() => reset()} className="btn-primary">
            Start over
          </button>
        </div>
      )}

      {step === "success" && (
        <div className={`${CARD} space-y-6`}>
          <h2 className="text-2xl font-bold text-slate-800">🎉 Your Model is Live!</h2>
          {result?.space_name && (
            <p className="text-sm text-slate-600">
              Space: <span className="font-semibold text-slate-800">{result.space_name}</span>
            </p>
          )}
          <div className="space-y-2 text-sm">
            {spaceUrl && (
              <p className="break-all text-slate-700">
                <span className="font-medium text-slate-800">Space: </span>
                {spaceUrl}
              </p>
            )}
            {apiUrl && (
              <p className="break-all text-slate-700">
                <span className="font-medium text-slate-800">API: </span>
                {apiUrl}
              </p>
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            {spaceUrl && (
              <a
                href={spaceUrl}
                target="_blank"
                rel="noreferrer"
                className="btn-primary"
              >
                <ExternalLink className="mr-1 h-4 w-4" /> Open Space
              </a>
            )}
            {apiUrl && (
              <a
                href={apiUrl}
                target="_blank"
                rel="noreferrer"
                className="btn-ghost"
              >
                <ExternalLink className="mr-1 h-4 w-4" /> View API
              </a>
            )}
            <button type="button" onClick={copyPrimary} className="btn-ghost">
              <Copy className="mr-1 h-4 w-4" />
              {copied ? "Copied!" : "Copy URL"}
            </button>
            <button
              type="button"
              onClick={triggerTestUI}
              disabled={generatingTestUI}
              className="btn-ghost disabled:opacity-50"
            >
              {generatingTestUI ? (
                <>
                  <Loader2 className="mr-1 inline h-4 w-4 animate-spin" />
                  Generating…
                </>
              ) : (
                "Test Deployment"
              )}
            </button>
            <button type="button" onClick={() => reset()} className="btn-primary">
              Deploy Another Model
            </button>
          </div>

          {modelType && (
            <p className="text-xs text-slate-500">
              Model type:{" "}
              <span className="font-semibold text-slate-700">{modelType}</span>
              {modelType === "Other" && modelTypeDesc
                ? ` — ${modelTypeDesc}`
                : null}
            </p>
          )}
          {testGenError && (
            <p className="text-sm text-red-600">{testGenError}</p>
          )}

          {generatedTestHTML && (
            <div className="space-y-3 pt-2">
              <div className="flex items-center justify-between">
                <h3 className="text-base font-bold text-slate-800">Generated Test Interface</h3>
                <button
                  type="button"
                  onClick={() => setGeneratedTestHTML("")}
                  className="btn-ghost text-xs"
                >
                  Close
                </button>
              </div>
              <iframe
                title="Generated Test Interface"
                srcDoc={generatedTestHTML}
                sandbox="allow-scripts allow-forms"
                className="h-[600px] w-full rounded-xl border border-white/60 bg-white"
              />
            </div>
          )}
        </div>
      )}

    </div>
  );
}
