/* TL;DR: a three-headed dragon. Two fierce heads (forming trust, spotting the adversary), one derpy head (using trust). */
(function () {
  "use strict";
  const root = document.getElementById("dragon");
  if (!root) return;
  const G = "#3f8f6b", GD = "#2c6e50", GL = "#cfe8d9", HORN = "#efe6cf";

  // a fierce head drawn facing right; the snout points to +x: jaws open, fangs, glowing eye, smoke
  const fierce = `
    <path d="M-50 -14 L-92 -10 L-58 -28 L-96 -44 L-54 -46 L-80 -72 L-36 -56z" fill="${GD}"/>
    <path d="M-26 -54 L-92 -118 L-40 -48z" fill="${HORN}" stroke="${GD}" stroke-width="3"/>
    <path d="M0 -58 L-22 -132 L14 -54z" fill="${HORN}" stroke="${GD}" stroke-width="3"/>
    <path d="M-16 6 L110 -16 L98 40 L-16 26z" fill="#6e1414"/>
    <path d="M-34 16 Q8 20 96 34 Q106 42 96 50 Q40 58 -12 48 Q-40 40 -34 16z" fill="${G}" stroke="${GD}" stroke-width="3"/>
    <path d="M26 24 l4 -13 l5 14z M44 27 l4 -14 l5 15z M62 30 l4 -14 l5 15z M80 33 l4 -15 l5 16z" fill="#fff"/>
    <path d="M-54 -12 Q-50 -62 4 -64 Q48 -64 108 -38 Q124 -30 116 -16 L68 -6 L8 4 Q-22 10 -40 6 Q-58 0 -54 -12z" fill="${G}" stroke="${GD}" stroke-width="3"/>
    <path d="M18 0 l5 14 l5 -15z M36 -3 l5 14 l5 -15z M54 -6 l5 13 l5 -14z M72 -9 l5 12 l5 -13z" fill="#fff"/>
    <path d="M90 -14 l6 26 l7 -28z" fill="#fff" stroke="${GD}" stroke-width="1.5"/>
    <circle cx="40" cy="-28" r="19" fill="#ff5a2a" opacity=".28"/>
    <ellipse cx="40" cy="-28" rx="14" ry="7" fill="#ffd23f" transform="rotate(16 40 -28)"/>
    <ellipse cx="40" cy="-28" rx="14" ry="7" fill="none" stroke="#e8461f" stroke-width="2" transform="rotate(16 40 -28)"/>
    <rect x="38" y="-36" width="4.5" height="16" rx="2" fill="#1c2230" transform="rotate(16 40 -28)"/>
    <path d="M8 -50 Q36 -46 66 -30" stroke="#1d4a36" stroke-width="10" stroke-linecap="round" fill="none"/>
    <path d="M14 -54 L22 -66 L30 -52" fill="#1d4a36"/>
    <circle cx="104" cy="-28" r="3.4" fill="#1d4a36"/>
    <g class="dr-smoke"><circle cx="124" cy="-40" r="6" fill="#9aa0aa"/><circle cx="136" cy="-52" r="8" fill="#b4b9c1"/><circle cx="152" cy="-62" r="10" fill="#cfd3d9"/></g>
    <path d="M-42 -24 Q-58 -14 -52 2 M-38 4 Q-52 14 -44 28" stroke="${GD}" stroke-width="3" fill="none"/>`;

  // the derpy head, looking at the viewer, tongue out
  const derp = `
    <path d="M-30 -44 Q-62 -66 -58 -94 Q-40 -70 -16 -52z" fill="${HORN}" stroke="${GD}" stroke-width="3"/>
    <path d="M26 -46 Q44 -78 70 -76 Q52 -62 40 -40z" fill="${HORN}" stroke="${GD}" stroke-width="3"/>
    <ellipse cx="0" cy="-6" rx="58" ry="50" fill="${G}" stroke="${GD}" stroke-width="3"/>
    <ellipse cx="6" cy="30" rx="44" ry="26" fill="${GL}" stroke="${GD}" stroke-width="3"/>
    <circle cx="-12" cy="22" r="3.4" fill="${GD}"/><circle cx="22" cy="22" r="3.4" fill="${GD}"/>
    <g class="dr-tongue"><path d="M10 44 Q2 78 18 98 Q34 112 40 92 Q42 72 28 46z" fill="#e8798e" stroke="#b8475e" stroke-width="3"/><path d="M24 56 Q28 76 32 90" stroke="#b8475e" stroke-width="2" fill="none"/></g>
    <path d="M-30 44 Q6 58 40 42" stroke="${GD}" stroke-width="3" fill="none"/>
    <rect x="-6" y="45" width="11" height="13" rx="2" fill="#fff" stroke="${GD}" stroke-width="2"/>
    <g class="dr-eyes">
      <circle cx="-22" cy="-24" r="22" fill="#fff" stroke="${GD}" stroke-width="3"/><circle cx="-30" cy="-18" r="8" fill="#1c2230"/>
      <circle cx="24" cy="-28" r="15" fill="#fff" stroke="${GD}" stroke-width="3"/><circle cx="31" cy="-34" r="5.5" fill="#1c2230"/>
    </g>`;

  root.innerHTML = `
  <svg viewBox="0 -44 1000 514" class="dr-svg" role="img" aria-label="A three-headed dragon: two fierce heads labelled forming trust and spotting the adversary, and a derpy head with its tongue out labelled using trust.">
    <path d="M300 470 Q240 300 120 330 Q210 360 200 470z" fill="${GD}" opacity=".55"/>
    <path d="M700 470 Q770 300 900 320 Q800 360 810 470z" fill="${GD}" opacity=".55"/>
    <g class="dr-neck dr-n1"><path d="M410 400 C 340 330, 260 330, 236 236" stroke="${GD}" stroke-width="66" fill="none" stroke-linecap="round"/><path d="M410 400 C 340 330, 260 330, 236 236" stroke="${G}" stroke-width="58" fill="none" stroke-linecap="round"/></g>
    <g class="dr-neck dr-n2"><path d="M490 380 C 480 300, 470 240, 480 150" stroke="${GD}" stroke-width="66" fill="none" stroke-linecap="round"/><path d="M490 380 C 480 300, 470 240, 480 150" stroke="${G}" stroke-width="58" fill="none" stroke-linecap="round"/></g>
    <g class="dr-neck dr-n3"><path d="M590 400 C 660 340, 740 330, 760 250" stroke="${GD}" stroke-width="66" fill="none" stroke-linecap="round"/><path d="M590 400 C 660 340, 740 330, 760 250" stroke="${G}" stroke-width="58" fill="none" stroke-linecap="round"/></g>
    <path d="M250 470 Q290 360 500 350 Q710 360 750 470z" fill="${G}" stroke="${GD}" stroke-width="3"/>
    <path d="M400 470 Q420 400 500 396 Q580 400 600 470z" fill="${GL}"/>
    <g transform="translate(232 214)"><g class="dr-head dr-h1"><g transform="scale(-1.15 1.15)">${fierce}</g></g></g>
    <g transform="translate(476 128)"><g class="dr-head dr-h2"><g transform="scale(-1.15 1.15) rotate(-6)">${fierce}</g></g></g>
    <g transform="translate(772 206)"><g class="dr-head dr-h3"><g transform="rotate(10) scale(1.3)">${derp}</g></g></g>
    <g class="dr-stars" transform="translate(772 96)">${[0, 1, 2, 3, 4].map(() => `<path class="dr-star" d="M0 -11 L3 -3.5 L11 -3.5 L4.6 1.4 L7 9 L0 4.6 L-7 9 L-4.6 1.4 L-11 -3.5 L-3 -3.5z" fill="#f5c542" stroke="#c9961c" stroke-width="1.5"/>`).join("")}</g>
  </svg>
  <div class="dr-labels">
    <div class="dr-lab pos"><h4>Forming trust</h4><div class="dl">Built up game after game</div></div>
    <div class="dr-lab pos"><h4>Spotting the adversary</h4><div class="dl">Found within a few games</div></div>
    <div class="dr-lab neg"><h4>Using trust</h4><div class="dl">Rarely turns into a deal</div></div>
  </div>
  <div class="dr-punch">Knowing whom to trust is not enough.</div>`;

  // dizzy stars circling the derpy head
  const stars = [...root.querySelectorAll(".dr-star")];
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const orbit = (t) => {
    stars.forEach((st, k) => {
      const a = t / 900 + (k * 2 * Math.PI) / stars.length;
      st.setAttribute("transform", `translate(${Math.cos(a) * 92} ${Math.sin(a) * 24}) scale(${1.45 + 0.4 * Math.sin(a)}) rotate(${(t / 6) % 360})`);
    });
    if (!reduce) requestAnimationFrame(orbit);
  };
  orbit(0);

  if (!("IntersectionObserver" in window)) { root.classList.add("go"); return; }
  new IntersectionObserver((es, ob) => es.forEach((e) => { if (e.isIntersecting) { root.classList.add("go"); ob.disconnect(); } }), { threshold: 0.35 }).observe(root);
})();
