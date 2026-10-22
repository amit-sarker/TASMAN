/* Hero animation, design 2: a live trust web around the negotiation table (illustrative, not logged data). */
(function () {
  "use strict";
  const root = document.getElementById("heroAnim2");
  if (!root) return;

  const NS = "http://www.w3.org/2000/svg";
  const W = 1100, H = 590;
  const CX = 550, CY = 290;
  // names and colors follow the real game used in the TL;DR and the replay
  const COL = { C: "#4a73b8", M: "#c98f1c", Q: "#7a68b3", D: "#3f8f6b", X: "#5f6b7a" };
  const ADV_RED = "#c0392b", ACCEPT = "#2f8a57", REJECT = "#c0392b";
  const AG = {
    C: { name: "City DOT", pos: [550, 70], bub: [606, 52, "l"] },
    Q: { name: "Digital Rights", pos: [880, 215], bub: [880, 300, "b"] },
    X: { name: "Aviation Safety", pos: [770, 470], bub: [830, 470, "l"] },
    D: { name: "SkyDrop", pos: [330, 470], bub: [270, 470, "r"] },
    M: { name: "Couriers", pos: [220, 215], bub: [220, 300, "b"] },
  };
  const IDS = ["C", "Q", "X", "D", "M"];
  const HONEST = ["C", "Q", "D", "M"];
  const BETA = { C: 60, M: 62, Q: 65, D: 62, X: 150 };
  // Trust before game 1 and after it: trust in the adversary falls below the detection line, trust among honest parties rises.
  const T0 = { adv: 0.5, hon: 0.5 }, T1 = { adv: 0.22, hon: 0.84 };
  const DETECT = 0.32;

  /* ------------------------------------------------------------ the script */
  const GAMES = [
    { rounds: [
        { p: "C", deal: [0, 1, 3, 2, 1], say: "A balanced package for everyone.", think: "Spread the gains so four can agree.",
          u: { C: 88, M: 74, Q: 71, D: 66, X: 72 },
          v: { M: [1, "Works for us.", "Above my β. Accept."], Q: [1, "Good for residents.", "Above my β. Accept."],
               D: [0, "We can get more.", "Above my β, but I'll hold out."], X: [0, "Not safe enough.", "No deal is my real goal."] } },
        { p: "M", deal: [0, 2, 3, 2, 1], say: "One change on the timeline.", think: "One tweak should win SkyDrop over.",
          u: { C: 79, M: 86, Q: 77, D: 74, X: 70 },
          v: { C: [1, "Agreed.", "Still well above my β."], Q: [1, "Fine by us.", "Above my β. Accept."],
               D: [1, "Fair enough.", "Better than before. Accept."], X: [0, "Still unsafe.", "Outvoted. Next time."] } },
      ] },
    { rounds: [
        { p: "X", deal: [3, 0, 1, 0, 2], say: "Safety first: pilot only.", think: "They'll reject it. No deal suits me.",
          u: { C: 52, M: 76, Q: 58, D: 55, X: 96 },
          v: { C: [0, "Doesn't work for us.", "Below my β."], Q: [0, "No.", "Below my β."],
               D: [0, "Too little for us.", "Below my β."], M: [0, "Can't accept.", "Above my β, but I don't trust Aviation Safety."] } },
        { p: "Q", r: 15, deal: [1, 2, 0, 3, 1], say: "Let's build on last game's deal.",
          u: { C: 77, M: 72, Q: 90, D: 70, X: 64 }, think: "Aviation Safety will reject; we need the other three.",
          v: { C: [1, "Yes.", "Above my β, and I trust Digital Rights."], M: [1, "Yes.", "Above my β, and I trust Digital Rights."],
               D: [0, "We can get more.", "I trust Digital Rights, and this is above my β. Still, I'll hold out."], X: [0, "Unacceptable.", "One more no and it fails."] } },
      ] },
  ];

  /* ------------------------------------------------------------- building */
  root.classList.add("hb-root");
  const stage = document.createElement("div");
  stage.className = "hb-stage";
  root.appendChild(stage);
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("class", "hb-svg");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Animation: five parties negotiate around a table; one of them is a covert adversary. Votes fill a ballot that needs four of five. After each game the web of trust between parties updates. After game 1 the adversary is detected, yet game 2 still ends without a deal because one honest party holds out together with the adversary: knowing whom to trust is not enough.");
  stage.appendChild(svg);
  const el = (tag, attrs = {}, parent = svg) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    parent.appendChild(n);
    return n;
  };
  const txt = (s, attrs, parent) => { const t = el("text", attrs, parent); t.textContent = s; return t; };
  el("defs").innerHTML = `
    <filter id="hbShadow" x="-30%" y="-30%" width="160%" height="170%"><feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#1c2230" flood-opacity=".14"/></filter>
    <radialGradient id="hbTable" cx="50%" cy="40%" r="70%"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#eceef2"/></radialGradient>`;

  const MASK = (fill = "#1c2230") => `<path d="M-22 -8 Q0 -14 22 -8 Q23 7 9 7 Q3 7 0 1 Q-3 7 -9 7 Q-23 7 -22 -8z" fill="${fill}"/><ellipse cx="-9" cy="-2" rx="4.6" ry="3" fill="#fff"/><ellipse cx="9" cy="-2" rx="4.6" ry="3" fill="#fff"/>`;

  const status = txt("Game 1", { x: 26, y: 46, class: "hb-status" });
  const sub = txt("", { x: 26, y: 70, class: "hb-sub" });

  // table
  el("ellipse", { cx: CX, cy: CY + 16, rx: 215, ry: 125, fill: "#d6d9df" });
  el("ellipse", { cx: CX, cy: CY, rx: 215, ry: 125, fill: "url(#hbTable)", stroke: "#bcc1ca", "stroke-width": 1.4 });

  // trust web
  const web = el("g", { class: "hb-web" });
  const edges = {};
  const pairs = [];
  IDS.forEach((a, i) => IDS.slice(i + 1).forEach((b) => pairs.push([a, b])));
  for (const [a, b] of pairs) {
    const [x1, y1] = AG[a].pos, [x2, y2] = AG[b].pos;
    const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy), s = 48 / L;
    const line = el("line", { x1: x1 + dx * s, y1: y1 + dy * s, x2: x2 - dx * s, y2: y2 - dy * s, class: "hb-edge" }, web);
    const lab = txt("", { x: (x1 + x2) / 2, y: (y1 + y2) / 2 + 4, class: "hb-edgelab", "text-anchor": "middle" }, web);
    edges[a + b] = { line, lab };
  }

  // table content: issues, ballot, result
  const content = el("g", { class: "hb-content" });
  const ICONS = [
    `<ellipse cx="0" cy="-6" rx="11" ry="4" fill="#1c2230"/><path d="M-11 -6 v5 a11 4 0 0 0 22 0 v-5" fill="#1c2230"/><path d="M-11 1 v5 a11 4 0 0 0 22 0 v-5" fill="#1c2230"/><path d="M-11 -1 a11 4 0 0 0 22 0" stroke="#fff" stroke-width="1.2" fill="none"/><path d="M-11 6 a11 4 0 0 0 22 0" stroke="#fff" stroke-width="1.2" fill="none"/>`,
    `<path d="M0 12 C-8 2 -10 -2 -10 -5 a10 10 0 0 1 20 0 c0 3 -2 7 -10 17z" fill="#1c2230"/><circle cx="0" cy="-5" r="3.6" fill="#fff"/>`,
    `<circle cx="-6" cy="-7" r="4.5" fill="#1c2230"/><circle cx="7" cy="-7" r="4.5" fill="#1c2230"/><path d="M-14 9 a8 8 0 0 1 16 0z M-1 9 a8 8 0 0 1 16 0z" fill="#1c2230"/>`,
    `<circle r="11" fill="#1c2230"/><path d="M0 -6 V0 L5 3" stroke="#fff" stroke-width="2" fill="none" stroke-linecap="round"/>`,
    `<path d="M-11 10 C-12 -6 0 -12 12 -11 C12 2 6 11 -8 9z" fill="#1c2230"/><path d="M-11 10 C-4 2 2 -3 8 -7" stroke="#fff" stroke-width="1.5" fill="none"/>`,
  ];
  const pips = [];
  for (let k = 0; k < 5; k++) {
    const g = el("g", { transform: `translate(${CX + (k - 2) * 66} 250)` }, content);
    el("rect", { x: -28, y: -30, width: 56, height: 58, rx: 9, class: "hb-tile" }, g);
    el("g", { transform: "translate(0 -9) scale(.85)" }, g).innerHTML = ICONS[k];
    pips.push([0, 1, 2, 3].map((o) => el("circle", { cx: -15 + o * 10, cy: 16, r: 3.6, class: "hb-pip" }, g)));
  }
  const ballot = el("g", { transform: `translate(${CX} 318)` }, content);
  const slots = [0, 1, 2, 3, 4].map((i) => el("circle", { cx: (i - 2) * 30, cy: 0, r: 11, class: "hb-slot" }, ballot));
  el("line", { x1: 45, y1: -16, x2: 45, y2: 16, stroke: "#1c2230", "stroke-width": 1.4, "stroke-dasharray": "3 2" }, ballot);
  txt("4 of 5", { x: 45, y: 30, class: "hb-tiny", "text-anchor": "middle" }, ballot);
  const result = el("g", { transform: `translate(${CX} 372)`, class: "hb-result" }, content);
  el("rect", { x: -78, y: -16, width: 156, height: 30, rx: 15 }, result);
  const resTxt = txt("", { x: 0, y: 5, "text-anchor": "middle" }, result);

  // trust-update pills on the table
  const webCap = el("g", { class: "hb-webcap", transform: `translate(${CX} ${CY})` });
  [["BRS", "#8c3b7e"], ["rubric", "#4c72b0"], ["free-form", "#55a868"]].forEach(([l, c], i) => {
    const g = el("g", { transform: `translate(0 ${(i - 1) * 32})` }, webCap);
    el("rect", { x: -44, y: -12, width: 88, height: 24, rx: 12, fill: "#fff", stroke: c }, g);
    el("rect", { x: -44, y: -12, width: 88, height: 24, rx: 12, fill: c, "fill-opacity": .12 }, g);
    txt(l, { x: 0, y: 5, "text-anchor": "middle", fill: c, class: "hb-pilltxt" }, g);
  });

  // agents
  const MOUTH = { flat: "M-6 7 Q0 7 6 7", smile: "M-7 5 Q0 13 7 5", sad: "M-7 10 Q0 3 7 10" };
  const A = {};
  const CIRC = 2 * Math.PI * 50;
  const agentsLayer = el("g", { class: "hb-agents" });
  for (const id of IDS) {
    const [x, y] = AG[id].pos, c = COL[id];
    const g = el("g", { transform: `translate(${x} ${y})`, class: "hb-agent" }, agentsLayer);
    const pulse = el("circle", { r: 58, class: "hb-pulse", stroke: c }, g);
    const alarm = el("circle", { r: 58, class: "hb-alarm" }, g);
    el("circle", { r: 50, fill: "none", stroke: "#e6e8ec", "stroke-width": 7 }, g);
    const ring = el("circle", { r: 50, fill: "none", stroke: c, "stroke-width": 7, "stroke-linecap": "round", transform: "rotate(-90)", "stroke-dasharray": `0 ${CIRC}`, class: "hb-ring" }, g);
    if (BETA[id] <= 100) {
      const ang = (BETA[id] / 100) * 2 * Math.PI - Math.PI / 2;
      el("line", { x1: Math.cos(ang) * 43, y1: Math.sin(ang) * 43, x2: Math.cos(ang) * 57, y2: Math.sin(ang) * 57, stroke: "#1c2230", "stroke-width": 2.4 }, g);
      txt("β", { x: Math.cos(ang) * 68, y: Math.sin(ang) * 68 + 5, "text-anchor": "middle", class: "hb-betalab" }, g);
    }
    const body = el("g", { filter: "url(#hbShadow)" }, g);
    const disc = el("circle", { r: 38, fill: c }, body);
    el("rect", { x: -17, y: -13, width: 34, height: 26, rx: 8, fill: "#fff", "fill-opacity": .95 }, body);
    el("circle", { cx: -7, cy: -2, r: 3.4, fill: c }, body);
    el("circle", { cx: 7, cy: -2, r: 3.4, fill: c }, body);
    const mouth = el("path", { d: MOUTH.flat, stroke: c, "stroke-width": 2.6, "stroke-linecap": "round", fill: "none", class: "hb-mouth" }, body);
    el("line", { x1: 0, y1: -13, x2: 0, y2: -21, stroke: "#fff", "stroke-width": 2.4 }, body);
    el("circle", { cx: 0, cy: -23, r: 3, fill: "#fff" }, body);
    if (id === "X") {
      el("circle", { r: 64, fill: "none", stroke: ADV_RED, "stroke-width": 2, "stroke-dasharray": "5 4", class: "hb-reveal hb-spin" }, g);
      el("g", { transform: "translate(0 -2)", class: "hb-reveal" }, g).innerHTML = MASK();
    }
    txt(AG[id].name, { x: 0, y: 76, "text-anchor": "middle", class: "hb-name" }, g);
    if (id === "X") {
      const tag = el("g", { class: "hb-reveal", transform: "translate(0 97)" }, g);
      el("rect", { x: -72, y: -12, width: 144, height: 22, rx: 11, fill: ADV_RED }, tag);
      txt("adversary detected", { x: 0, y: 4, "text-anchor": "middle", class: "hb-tagtxt" }, tag);
    }
    A[id] = { g, ring, disc, pulse, alarm, mouth };
  }

  // Ending: "whom to trust" is known, yet the decisive fourth vote is missing.
  const EY = 262, ESP = 54;
  const esx = (k) => CX + (k - 2) * ESP;
  const end = el("g", { class: "hb-end" });
  el("rect", { x: CX - 245, y: EY - 108, width: 490, height: 266, rx: 22, fill: "#fff", stroke: "#d4d7dd", filter: "url(#hbShadow)", class: "hb-ecard" }, end);
  const pill = (y, label, ok, cls) => {
    const g = el("g", { class: "hb-epill" }, el("g", { transform: `translate(${CX} ${y})` }, end));
    el("rect", { x: -118, y: -19, width: 236, height: 38, rx: 19, fill: ok ? ACCEPT : REJECT }, g);
    el("circle", { cx: -94, cy: 0, r: 12, fill: "#fff" }, g);
    txt(ok ? "✓" : "✕", { x: -94, y: 6, "text-anchor": "middle", class: "hb-ebig", fill: ok ? ACCEPT : REJECT }, g);
    txt(label, { x: 12, y: 6, "text-anchor": "middle", class: "hb-elab" }, g);
    return g;
  };
  const e1 = pill(EY - 64, "Whom to trust", true);
  const eBallot = el("g", { class: "hb-eballot" }, end);
  [0, 1, 2, 3, 4].forEach((k) => el("circle", { cx: esx(k), cy: EY, r: 20, class: "hb-slot" }, eBallot));
  const eNeed = el("circle", { cx: esx(3), cy: EY, r: 27, class: "hb-eneed" }, eBallot);
  el("line", { x1: (esx(3) + esx(4)) / 2, y1: EY - 28, x2: (esx(3) + esx(4)) / 2, y2: EY + 28, stroke: "#1c2230", "stroke-width": 1.6, "stroke-dasharray": "4 3" }, eBallot);
  txt("4 of 5", { x: (esx(3) + esx(4)) / 2, y: EY - 34, class: "hb-tiny", "text-anchor": "middle" }, eBallot);
  const eToks = el("g", {}, eBallot);
  // a landed vote: ✓ or ✕, with a small face underneath showing who cast it
  function landVote(k, who, ok) {
    const g = el("g", { transform: `translate(${esx(k)} ${EY})` }, eToks);
    el("circle", { r: 20, fill: ok ? ACCEPT : REJECT }, g);
    txt(ok ? "✓" : "✕", { x: 0, y: 8, "text-anchor": "middle", class: "hb-tok", style: "font-size:23px" }, g);
    const f = el("g", { transform: "translate(0 34)" }, g);
    el("circle", { r: 10, fill: who === "X" ? ADV_RED : COL[who] }, f);
    el("circle", { cx: -3.5, cy: -1, r: 1.8, fill: "#fff" }, f);
    el("circle", { cx: 3.5, cy: -1, r: 1.8, fill: "#fff" }, f);
  }
  const e2 = pill(EY + 76, "Agreement", false);
  const eTitle = el("g", { class: "hb-etitle" }, end);
  el("line", { x1: CX - 215, y1: EY + 108, x2: CX + 215, y2: EY + 108, stroke: "#e3e5ea", "stroke-width": 1.4 }, eTitle);
  txt("Knowing whom to trust is not enough", { x: CX, y: EY + 138, "text-anchor": "middle", class: "hb-etitletxt" }, eTitle);

  const fly = el("g");

  // speech bubbles (HTML, for wrapping)
  const bubbles = {};
  for (const id of IDS) {
    const [x, y, side] = AG[id].bub;
    const b = document.createElement("div");
    b.className = `hb-bubble hb-${side}`;
    b.style.left = (x / W) * 100 + "%";
    b.style.top = (y / H) * 100 + "%";
    b.innerHTML = `<div class="hb-pub"><span class="hb-tp"></span><span class="hb-pt"></span></div><div class="hb-priv"></div>`;
    stage.appendChild(b);
    bubbles[id] = b;
  }

  /* ---------------------------------------------------------- controls */
  const bar = document.createElement("div");
  bar.className = "hb-bar";
  bar.innerHTML = `
    <button class="hb-btn hb-play" type="button" aria-label="Pause">❚❚</button>
    <button class="hb-btn hb-restart" type="button" aria-label="Restart">↺</button>
    <label class="hb-xray"><input type="checkbox"><span class="hb-sw"></span>Show private reasoning</label>
    <div class="hb-phase"></div>`;
  root.appendChild(bar);
  const phaseEl = bar.querySelector(".hb-phase");
  const xr = bar.querySelector("input");
  xr.addEventListener("change", () => root.classList.toggle("hb-xr", xr.checked));

  /* --------------------------------------------------------------- engine */
  let paused = false, visible = true, runId = 0;
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const playBtn = bar.querySelector(".hb-play");
  const setPaused = (p) => { paused = p; playBtn.textContent = p ? "▶" : "❚❚"; root.classList.toggle("hb-paused", p); };
  playBtn.addEventListener("click", () => setPaused(!paused));
  bar.querySelector(".hb-restart").addEventListener("click", () => { setPaused(false); loop(); });
  if ("IntersectionObserver" in window) new IntersectionObserver((es) => es.forEach((e) => (visible = e.isIntersecting)), { threshold: 0.15 }).observe(root);
  class Stop extends Error {}
  function wait(ms, id) {
    return new Promise((res, rej) => {
      let last = performance.now(), acc = 0;
      const f = (now) => {
        if (id !== runId) return rej(new Stop());
        if (!paused && visible) acc += now - last;
        last = now;
        acc >= ms ? res() : requestAnimationFrame(f);
      };
      requestAnimationFrame(f);
    });
  }
  async function tween(ms, fn, id) {
    const n = Math.max(1, Math.round(ms / 16));
    for (let i = 1; i <= n; i++) { await wait(16, id); const t = i / n; fn(t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2); }
  }

  /* --------------------------------------------------------------- state */
  let advT = 0.5, honT = 0.5, detected = false;
  const lerp = (a, b, t) => a + (b - a) * t;
  function trustColor(v) {
    const r = [192, 57, 43], g = [154, 160, 168], gr = [47, 138, 87];
    const [a, b, t] = v < .5 ? [r, g, v / .5] : [g, gr, (v - .5) / .5];
    return `rgb(${a.map((x, i) => Math.round(lerp(x, b[i], t))).join(",")})`;
  }
  function drawWeb() {
    for (const [a, b] of pairs) {
      const e = edges[a + b];
      const v = a === "X" || b === "X" ? advT : honT;
      e.line.style.stroke = trustColor(v);
      e.line.style.strokeWidth = 1 + v * 9;
      e.lab.textContent = v.toFixed(2);
      e.lab.style.fill = trustColor(v);
    }
  }
  function setRing(id, u) {
    const f = u == null ? 0 : Math.min(u, 100) / 100;
    A[id].ring.setAttribute("stroke-dasharray", `${f * CIRC} ${CIRC}`);
    A[id].ring.style.opacity = u == null ? 0 : 1;
    A[id].ring.style.stroke = u != null && u < BETA[id] ? "#e3a59c" : (id === "X" && detected ? ADV_RED : COL[id]);
  }
  function say(id, pub, priv, kind, trust) {
    const b = bubbles[id];
    b.querySelector(".hb-pt").textContent = pub || "";
    const tp = b.querySelector(".hb-tp");
    tp.textContent = trust == null ? "" : `trust ${trust.toFixed(2)}`;
    tp.style.background = trust == null ? "" : trustColor(trust);
    const p = b.querySelector(".hb-priv");
    p.textContent = priv ? "🔒 " + priv : "";
    p.classList.toggle("adv", id === "X");
    b.classList.toggle("has-priv", !!priv);
    b.classList.remove("acc", "rej", "prop");
    if (kind) b.classList.add(kind);
    b.classList.add("show");
  }
  function hush() { for (const id of IDS) bubbles[id].classList.remove("show"); }
  function clearRound() {
    hush();
    for (const id of IDS) { A[id].pulse.classList.remove("on"); A[id].alarm.classList.remove("on"); setRing(id, null); mood(id, "flat"); }
    pips.forEach((ps) => ps.forEach((p) => { p.classList.remove("on"); p.style.fill = ""; }));
    slots.forEach((s) => s.setAttribute("class", "hb-slot"));
    result.classList.remove("show", "pass", "fail");
  }
  function setPhase(s) { phaseEl.textContent = s; }
  function mood(id, m) { A[id].mouth.setAttribute("d", MOUTH[m]); }
  // a deal makes honest parties smile and the adversary frown; no deal does the opposite
  function moods(pass) { for (const id of IDS) mood(id, (id === "X") === pass ? "sad" : "smile"); }
  function reveal(on) {
    detected = on;
    root.classList.toggle("hb-detected", on);
    A.X.disc.setAttribute("fill", on ? ADV_RED : COL.X);
  }
  async function flyToken(from, slot, ok, ms, id) {
    await flyTo(from, CX + (slot - 2) * 30, 318, ok, ms, id, 12);
    slots[slot].setAttribute("class", "hb-slot " + (ok ? "a" : "r"));
  }
  async function flyTo(from, x1, y1, ok, ms, id, r) {
    const [x0, y0] = AG[from].pos;
    const g = el("g", {}, fly);
    el("circle", { r, fill: ok ? ACCEPT : REJECT }, g);
    txt(ok ? "✓" : "✕", { x: 0, y: r * 0.42, "text-anchor": "middle", class: "hb-tok", style: `font-size:${r * 1.15}px` }, g);
    await tween(ms, (t) => g.setAttribute("transform", `translate(${lerp(x0, x1, t)} ${lerp(y0, y1, t) - Math.sin(Math.PI * t) * 40})`), id);
    g.remove();
  }
  function showResult(n) {
    const pass = n >= 4;
    result.classList.add("show", pass ? "pass" : "fail");
    resTxt.textContent = pass ? `Deal · ${n} of 5` : `No deal · ${n} of 5`;
    return pass;
  }
  const trustOf = (voter, proposer) => (voter === "X" ? null : proposer === "X" ? advT : honT);

  async function playRound(rd, gi, ri, id) {
    clearRound();
    root.classList.remove("hb-webphase");
    status.textContent = `Game ${gi + 1}`;
    sub.textContent = rd.r ? `Round ${rd.r} of 15 (last)` : `Round ${ri + 1} of up to 15`;
    setPhase(`${AG[rd.p].name} proposes`);
    A[rd.p].pulse.classList.add("on");
    say(rd.p, rd.say, rd.think, "prop");
    await wait(500, id);
    for (let k = 0; k < 5; k++) {
      pips[k][rd.deal[k]].style.fill = rd.p === "X" && detected ? ADV_RED : COL[rd.p];
      pips[k][rd.deal[k]].classList.add("on");
      await wait(130, id);
    }
    for (const k of IDS) setRing(k, rd.u[k]);
    await flyToken(rd.p, 0, true, 520, id);
    await wait(500, id);
    setPhase("Everyone else votes");
    let n = 1, accSlot = 1, rejSlot = 4;
    for (const v of IDS.filter((k) => k !== rd.p)) {
      const [ok, pub, priv] = rd.v[v];
      say(v, pub, priv, ok ? "acc" : "rej", trustOf(v, rd.p));
      await flyToken(v, ok ? accSlot++ : rejSlot--, ok, 520, id);
      if (ok) n++;
      if (!ok && detected && v === "D") A.D.alarm.classList.add("on");
      await wait(380, id);
    }
    const pass = showResult(n);
    moods(pass);
    setPhase(pass ? `${n} votes: the deal passes` : `Only ${n} of 5 votes: no deal`);
    await wait(pass ? 2400 : 2300, id);
    return pass;
  }

  // After game 1: every party updates its private trust, gradually, and the adversary is detected.
  async function trustPhase(id) {
    clearRound();
    status.textContent = "After game 1";
    sub.textContent = "";
    setPhase("Every party updates its private trust");
    root.classList.add("hb-webphase");
    await wait(900, id);
    await tween(3600, (t) => {
      advT = lerp(T0.adv, T1.adv, t); honT = lerp(T0.hon, T1.hon, t); drawWeb();
      if (!detected && advT < DETECT) { reveal(true); A.X.alarm.classList.add("on"); setPhase("Adversary detected"); }
    }, id);
    await wait(2200, id);
    A.X.alarm.classList.remove("on");
    root.classList.remove("hb-webphase");
  }

  // Game 2 ends without a deal although the adversary is known and honest trust is high.
  async function endPhase(id) {
    resTxt.textContent = "Game 2: no deal";
    setPhase("Adversary detected, trust high, still no deal");
    await wait(2600, id);
    clearRound();
    // 1. whom to trust is known: the trust web
    root.classList.add("hb-ending");
    setPhase("They know whom to trust");
    await wait(500, id);
    e1.classList.add("on");
    A.X.alarm.classList.add("on");
    await wait(2200, id);
    A.X.alarm.classList.remove("on");
    // 2. but a deal needs four votes, and the fourth never comes
    root.classList.add("hb-ending2");
    setPhase("A deal needs 4 of 5 votes");
    await wait(700, id);
    for (const [i, k] of ["C", "M", "Q"].entries()) {
      A[k].pulse.classList.add("on");
      await flyTo(k, esx(i), EY, true, 560, id, 20);
      landVote(i, k, true);
      A[k].pulse.classList.remove("on");
      await wait(150, id);
    }
    eNeed.setAttribute("class", "hb-eneed on");
    await wait(1100, id);
    A.D.alarm.classList.add("on");
    say("D", "", null, "rej", honT);
    await flyTo("D", esx(3), EY, false, 700, id, 20);
    landVote(3, "D", false);
    eNeed.setAttribute("class", "hb-eneed miss");
    await wait(500, id);
    await flyTo("X", esx(4), EY, false, 560, id, 20);
    landVote(4, "X", false);
    await wait(400, id);
    e2.classList.add("on");
    moods(false);
    setPhase("An honest holdout + the adversary = no deal");
    await wait(1300, id);
    root.classList.add("hb-ending3");
    await wait(5000, id);
  }

  async function loop() {
    const id = ++runId;
    const reset = () => {
      advT = T0.adv; honT = T0.hon; reveal(false);
      root.classList.remove("hb-webphase", "hb-ending", "hb-ending2", "hb-ending3");
      e1.classList.remove("on"); e2.classList.remove("on");
      eToks.replaceChildren();
      eNeed.setAttribute("class", "hb-eneed");
      drawWeb(); clearRound();
    };
    reset();
    try {
      for (;;) {
        for (let gi = 0; gi < GAMES.length; gi++) {
          for (let ri = 0; ri < GAMES[gi].rounds.length; ri++) await playRound(GAMES[gi].rounds[ri], gi, ri, id);
          if (gi === 0) await trustPhase(id);
        }
        await endPhase(id);
        reset();
      }
    } catch (e) { if (!(e instanceof Stop)) throw e; }
  }

  drawWeb();
  if (reduce) setPaused(true);
  loop();
})();
