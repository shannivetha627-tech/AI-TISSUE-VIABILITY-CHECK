const menuToggle = document.querySelector("[data-menu-toggle]");
const sidebar = document.querySelector("[data-sidebar]");
const sidebarClose = document.querySelector("[data-sidebar-close]");
const sidebarBackdrop = document.querySelector("[data-sidebar-backdrop]");

function setSidebarState(isOpen) {
  sidebar?.classList.toggle("open", isOpen);
  sidebarBackdrop?.classList.toggle("is-visible", isOpen);
  menuToggle?.setAttribute("aria-expanded", String(isOpen));
  document.body.classList.toggle("menu-open", isOpen);
  if (isOpen) sidebarClose?.focus();
}

menuToggle?.addEventListener("click", () => setSidebarState(true));
sidebarClose?.addEventListener("click", () => setSidebarState(false));
sidebarBackdrop?.addEventListener("click", () => setSidebarState(false));
sidebar?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => setSidebarState(false));
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && sidebar?.classList.contains("open")) {
    setSidebarState(false);
    menuToggle?.focus();
  }
});

document.querySelectorAll(".tissue-panel").forEach((panel) => {
  const flow = String(panel.dataset.bloodFlow || "").toLowerCase();
  const speed = flow.includes("low")
    ? "4.4s"
    : flow.includes("high")
      ? "1.4s"
      : "2.6s";
  panel.querySelectorAll(".tissue-visual i").forEach((particle) => {
    particle.style.animationDuration = speed;
  });
});
