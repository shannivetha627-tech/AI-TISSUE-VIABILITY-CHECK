const camera = document.getElementById("camera");
const status = document.getElementById("status");
const verifyButton = document.getElementById("verify");
const challenge = document.body.dataset.challenge;
let stream;

function setStatus(message, state = "") {
  status.textContent = message;
  status.className = `status ${state}`;
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus(
      "Camera access is unavailable. Please use a supported browser.",
      "error",
    );
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 640, max: 640 },
        height: { ideal: 480, max: 480 },
      },
      audio: false,
    });
    camera.srcObject = stream;
    await camera.play();
    setStatus(
      `Camera ready. Look at the camera, then turn your head slightly ${challenge}.`,
    );
    verifyButton.disabled = false;
  } catch (error) {
    setStatus(
      "Camera access is unavailable. Please check browser permissions and try again.",
      "error",
    );
  }
}

verifyButton.addEventListener("click", async () => {
  verifyButton.disabled = true;
  setStatus("Get ready...");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    await new Promise((resolve) => setTimeout(resolve, 1200));
    const canvas = document.createElement("canvas");
    const captureWidth = Math.min(camera.videoWidth || 640, 640);
    const captureHeight = Math.min(camera.videoHeight || 480, 480);
    canvas.width = captureWidth;
    canvas.height = captureHeight;
    const context = canvas.getContext("2d");
    const frames = [];
    setStatus(`Turn your head slightly ${challenge}`);
    for (let index = 0; index < 9; index += 1) {
      context.drawImage(camera, 0, 0, captureWidth, captureHeight);
      frames.push(canvas.toDataURL("image/jpeg", 0.7));
      await new Promise((resolve) => setTimeout(resolve, 400));
    }
    setStatus("Checking liveness...");
    const formData = new URLSearchParams();
    frames.forEach((frame) => formData.append("image_data", frame));
    const response = await fetch("/doctor/verify-face", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: formData,
      signal: controller.signal,
    });
    const result = await response.json();
    if (!response.ok || !result.verified) {
      throw new Error(result.error || "Verification failed. Please retry.");
    }
    setStatus("Identity verified", "ok");
    stream?.getTracks().forEach((track) => track.stop());
    window.location.assign(result.redirect);
  } catch (error) {
    setStatus(
      error.name === "AbortError"
        ? "Verification timed out. Please try again."
        : error.message,
      "error",
    );
  } finally {
    clearTimeout(timeout);
    verifyButton.disabled = false;
  }
});

window.addEventListener("beforeunload", () => {
  stream?.getTracks().forEach((track) => track.stop());
});
startCamera();
