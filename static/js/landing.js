const landingMenuToggle = document.querySelector("[data-landing-menu]");
const landingMenu = document.getElementById("landing-menu");
const landingBackdrop = document.querySelector("[data-landing-backdrop]");
const landingClose = document.querySelector("[data-landing-close]");
const landingNav = document.querySelector(".landing-nav");

const updateNav = () => {
  landingNav?.classList.toggle("is-scrolled", window.scrollY > 18);
};
updateNav();
window.addEventListener("scroll", updateNav, { passive: true });

if (landingMenuToggle && landingMenu) {
  const setMenuState = (isOpen) => {
    landingMenuToggle.setAttribute("aria-expanded", String(isOpen));
    landingMenuToggle.querySelector(".sr-only").textContent = isOpen
      ? "Close menu"
      : "Open menu";
    landingMenu.classList.toggle("is-open", isOpen);
    landingBackdrop?.classList.toggle("is-visible", isOpen);
    document.body.classList.toggle("menu-open", isOpen);
    if (isOpen) landingClose?.focus();
  };

  landingMenuToggle.addEventListener("click", () => {
    setMenuState(landingMenuToggle.getAttribute("aria-expanded") !== "true");
  });

  landingMenu.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => setMenuState(false));
  });
  landingClose?.addEventListener("click", () => setMenuState(false));
  landingBackdrop?.addEventListener("click", () => setMenuState(false));

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && landingMenu.classList.contains("is-open")) {
      setMenuState(false);
      landingMenuToggle.focus();
    }
  });
}

const revealItems = document.querySelectorAll("[data-reveal]");
if ("IntersectionObserver" in window && revealItems.length) {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.style.setProperty(
          "--reveal-delay",
          entry.target.dataset.delay ||
            entry.target.style.getPropertyValue("--delay") ||
            "0ms",
        );
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    { threshold: 0.12 },
  );
  revealItems.forEach((item) => revealObserver.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add("is-visible"));
}

const timeline = document.querySelector("[data-timeline]");
if (timeline && "IntersectionObserver" in window) {
  const timelineObserver = new IntersectionObserver(
    ([entry]) => {
      if (entry.isIntersecting) timeline.classList.add("is-active");
    },
    { threshold: 0.3 },
  );
  timelineObserver.observe(timeline);
}
