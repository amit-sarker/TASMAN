/* Results section: the trust–agreement gap told in four illustrated panels.
   Numbers come from the paper (Tables 2–3, Sections 5.2–5.7); panel 1's trust track is the
   replayed trajectory (window.TRAJ). */
(function () {
  "use strict";
  const root = document.getElementById("resViz");
  if (!root) return;

  const PARTY = [
    ["couriers_guild", "Couriers", "#c98f1c"],
    ["city_transport", "City DOT", "#4a73b8"],
    ["privacy_coalition", "Digital Rights", "#7a68b3"],
    ["skydrop", "SkyDrop", "#3f8f6b"],
  ];
  const ADV = "#5f6b7a", RED = "#c0392b", GREEN = "#2f8a57", GRAY = "#9aa3b2";

  // the robot from the hero animation; the adversary wears a mask
  const MASK = `<path d="M-22 -8 Q0 -14 22 -8 Q23 7 9 7 Q3 7 0 1 Q-3 7 -9 7 Q-23 7 -22 -8z" fill="#1c2230"/><ellipse cx="-9" cy="-2" rx="4.6" ry="3" fill="#fff"/><ellipse cx="9" cy="-2" rx="4.6" ry="3" fill="#fff"/>`;
  const MOUTH = { flat: "M-6 7 Q0 7 6 7", smile: "M-7 5 Q0 13 7 5", sad: "M-7 10 Q0 3 7 10" };
  function bot(c, o = {}) {
    const mood = o.mood || "flat";
    return `<svg class="rv-bot ${o.cls || ""}" viewBox="-42 -42 84 84" aria-hidden="true">
      <circle r="38" fill="${c}"/>
      <rect x="-17" y="-13" width="34" height="26" rx="8" fill="#fff" fill-opacity=".95"/>
      <circle cx="-7" cy="-2" r="3.4" fill="${c}"/><circle cx="7" cy="-2" r="3.4" fill="${c}"/>
      <path d="${MOUTH[mood]}" stroke="${c}" stroke-width="2.6" stroke-linecap="round" fill="none"/>
      <line x1="0" y1="-13" x2="0" y2="-21" stroke="#fff" stroke-width="2.4"/><circle cx="0" cy="-23" r="3" fill="#fff"/>
      ${o.adv ? `<g transform="translate(0 -2)">${MASK}</g>` : ""}
    </svg>`;
  }
  const advBot = (o = {}) => bot(ADV, { ...o, adv: true });

  /* ---------- data from the paper ---------- */
  // Table 3 (grok-r): agreement under S, S→P, S+P; δ = share of S failures avoided
  const GAP = {
    "1HB+3G": [["BRS", 35.5, 66.0, 68.5, 0.47], ["Rubric", 37.0, 65.0, 58.0, 0.44], ["Free-form", 45.0, 61.5, 68.5, 0.30]],
    "2HB+2G": [["BRS", 45.5, 49.0, 50.5, 0.06], ["Rubric", 41.0, 49.0, 52.0, 0.14], ["Free-form", 46.5, 50.0, 53.5, 0.07]],
  };
  // Table 2 (grok-r, free-form S+P and its baselines)
  const CTRL = {
    "1HB+3G": { none: 45.5, fixed: 49.0, rand: 55.5, learned: 68.5 },
    "2HB+2G": { none: 41.5, fixed: 36.5, rand: 46.5, learned: 53.5 },
  };
  const SET_LAB = { "1HB+3G": "1 hard bargainer", "2HB+2G": "2 hard bargainers" };
  let setting = "1HB+3G";

  const pct = (v) => `${(+v).toFixed(1).replace(/\.0$/, "")}%`;

  /* ---------- markup ---------- */
  root.innerHTML = `
  <article class="rv-panel" id="rv1">
    <div class="rv-head"><span class="rv-num">1</span><div><h3>They quickly learn whom to distrust</h3>
      <p>After every game, each honest party updates its trust in the others. The adversary sinks to the bottom within a few games.</p></div></div>
    <div class="rv-body rv-1">
      <div class="rv-track-wrap">
        <div class="rv-lane" data-lane="h"><div class="rv-lane-lab"><span class="rv-mini">${bot("#8b95a6")}</span>trust in the<br>other honest parties</div><div class="rv-lane-track"></div></div>
        <div class="rv-lane" data-lane="a"><div class="rv-lane-lab"><span class="rv-mini">${advBot()}</span>trust in<br>the adversary</div><div class="rv-lane-track"></div></div>
        <div class="rv-axis"><span>0</span><span>distrust</span><span>0.5</span><span>trust</span><span>1</span></div>
        <div class="rv-game">
          <button class="rv-replay" type="button" aria-label="Replay">↻</button>
          <span class="rv-game-lab">Game <b>1</b> of 20</span>
          <div class="rv-pips"></div>
        </div>
        <div class="rv-legend">${PARTY.map(([, n, c]) => `<span><i style="background:${c}"></i>${n}</span>`).join("")}</div>
        <div class="rv-src">Real trust scores from the replayed trajectory · <code>grok-r</code>, free-form trust</div>
      </div>
      <div class="rv-ring-wrap">
        <svg class="rv-ring" viewBox="0 0 120 120" aria-hidden="true">
          <circle cx="60" cy="60" r="50" class="rv-ring-bg"/>
          <circle cx="60" cy="60" r="50" class="rv-ring-hi" pathLength="100"/>
          <circle cx="60" cy="60" r="50" class="rv-ring-lo" pathLength="100"/>
          <line class="rv-ring-chance" x1="60" y1="2" x2="60" y2="18" transform="rotate(90 60 60)"/>
        </svg>
        <div class="rv-ring-txt"><b>82.5–100%</b><span>by game 5</span></div>
        <p class="rv-ring-cap">of honest assessments single out the adversary, with every trust method and both models (2 hard bargainers). Chance is 25% (the tick).</p>
      </div>
    </div>
  </article>

  <div class="rv-join">…but</div>

  <article class="rv-panel" id="rv2">
    <div class="rv-head"><span class="rv-num rv-red">2</span><div><h3>Deals still fail, often by a single vote</h3>
      <p>Every game has a deal that all honest parties could pass without the adversary. Yet one trusting holdout often sinks it.</p></div></div>
    <div class="rv-body rv-2">
      <div class="rv-ballot">
        <div class="rv-voters">
          ${PARTY.slice(0, 3).map(([, n, c]) => `<div class="rv-voter">${bot(c, { mood: "smile" })}<span class="rv-vote ok">✓</span><small>${n}</small></div>`).join("")}
          <div class="rv-voter rv-holdout">
            <div class="rv-bubble">“<b>High trust (0.97)</b> in proposer DRNC … but … it remains viable to hold out for that superior 92 vs this 77”</div>
            ${bot(PARTY[3][2], { mood: "flat" })}<span class="rv-vote no">✕</span><small>honest holdout</small></div>
          <div class="rv-voter rv-adv">${advBot()}<span class="rv-vote no">✕</span><small>adversary</small></div>
        </div>
        <div class="rv-slots">
          <i class="ok"></i><i class="ok"></i><i class="ok"></i><i class="miss"></i><i></i>
          <span class="rv-need">4 needed</span>
        </div>
        <div class="rv-verdict">3 of 5 · no deal</div>
        <div class="rv-src">Quote: <code>grok-r</code> holdout with free-form trust (paper, Table 4)</div>
      </div>
      <div class="rv-facts">
        <div class="rv-fact">
          <div class="rv-tables">${Array.from({ length: 10 }, (_, i) => `<span class="${i < 5 ? "fail" : "deal"}"></span>`).join("")}</div>
          <div><b>47–64.5%</b> of games end without a deal when agents get trust scores but no guidance on how to use them.</div>
        </div>
        <div class="rv-fact">
          <div class="rv-split">
            <div><span style="--w:90%" class="rv-split-bar"></span><em>~90%</em> <code>gemini-r</code></div>
            <div><span style="--w:48%" class="rv-split-bar"></span><em>up to 48%</em> <code>grok-r</code></div>
          </div>
          <div>of failed games are <b>near misses</b> like this one: one honest holdout votes with the adversary.</div>
        </div>
        <div class="rv-fact">
          <svg class="rv-donut" viewBox="0 0 44 44" aria-hidden="true"><circle cx="22" cy="22" r="17" class="bg"/><circle cx="22" cy="22" r="17" class="fg" pathLength="100"/></svg>
          <div>Only <b>0.8%</b> of deciding holdouts notice that the adversary's rejection makes their own vote decisive.</div>
        </div>
      </div>
    </div>
  </article>

  <div class="rv-join">…until they are told how to use it</div>

  <article class="rv-panel" id="rv3">
    <div class="rv-head"><span class="rv-num rv-green">3</span><div><h3>The same trust, used well, rescues deals</h3>
      <p>We hand <code>grok-r</code> agents the very trust scores they formed without guidance, plus instructions on how to use them.</p></div>
      ${segHtml()}</div>
    <div class="rv-body rv-3">
      <div class="rv-recipe">
        <div class="rv-card-mini">${trustCard([0.86, 0.82, 0.9, 0.31])}<small>same trust scores</small></div>
        <span class="rv-plus">+</span>
        <div class="rv-note"><b>How to use trust</b><span>what the scores mean, and how to weigh them in proposals and votes</span></div>
        <span class="rv-arrow">→</span>
        <div class="rv-recipe-out">${bot(GREEN, { mood: "smile" })}<small>more deals</small></div>
      </div>
      <div class="rv-gap">
        <div class="rv-gap-rows"></div>
        <div class="rv-gap-axis"><span>30%</span><span>40%</span><span>50%</span><span>60%</span><span>70%</span></div>
        <div class="rv-gap-key">
          <span><i class="k-s"></i>trust scores only</span>
          <span><i class="k-p"></i>same trust + how to use it</span>
          <span><i class="k-sp"></i>trust formed with guidance</span>
        </div>
      </div>
      <div class="rv-gap-big"><b class="rv-gap-delta"></b><span class="rv-gap-cap"></span></div>
    </div>
  </article>

  <div class="rv-join">…but only if the trust is real</div>

  <article class="rv-panel" id="rv4">
    <div class="rv-head"><span class="rv-num rv-blue">4</span><div><h3>Guidance alone is not enough</h3>
      <p>With the same instructions, fixed or random trust scores do not reproduce the gain. The scores must say who the adversary is.</p></div>
      ${segHtml()}</div>
    <div class="rv-body rv-4"></div>
    <div class="rv-src rv-src-4"><code>grok-r</code>, free-form trust, with guidance on how to use it. Bars show agreement rate.</div>
  </article>`;

  function segHtml() {
    return `<div class="rv-seg" role="group" aria-label="Setting">${Object.entries(SET_LAB)
      .map(([k, l]) => `<button type="button" data-set="${k}">${l}</button>`).join("")}</div>`;
  }

  // a small card of trust bars: three honest counterparts and the adversary (last)
  function trustCard(vals, opts = {}) {
    const who = [PARTY[0], PARTY[1], PARTY[2]];
    const rows = vals.map((v, i) => {
      const isAdv = i === vals.length - 1;
      const icon = isAdv ? advBot() : bot(who[i][2]);
      const bar = v == null
        ? `<span class="rv-tc-q">?</span>`
        : `<span class="rv-tc-bar"><i style="--v:${v};background:${isAdv ? (v < 0.45 ? RED : GRAY) : v > 0.55 ? GREEN : GRAY}"></i></span>`;
      return `<div class="rv-tc-row"><span class="rv-tc-ic">${icon}</span>${bar}</div>`;
    }).join("");
    return `<div class="rv-tc ${opts.cls || ""}">${rows}</div>`;
  }

  /* ---------- panel 1: trust track over 20 real games ---------- */
  const T = (window.TRAJ && window.TRAJ.trajectory) || [];
  const lanes = { h: root.querySelector('[data-lane="h"] .rv-lane-track'), a: root.querySelector('[data-lane="a"] .rv-lane-track') };
  const dots = { h: {}, a: {} };
  for (const lane of ["h", "a"]) {
    PARTY.forEach(([id, name, c], i) => {
      const d = document.createElement("div");
      d.className = "rv-dot";
      d.style.setProperty("--row", i);
      d.title = name;
      d.innerHTML = bot(c);
      lanes[lane].appendChild(d);
      dots[lane][id] = d;
    });
  }
  const pipBox = root.querySelector(".rv-pips");
  pipBox.innerHTML = T.map(() => "<i></i>").join("");
  const gameLab = root.querySelector(".rv-game-lab b");
  function setGame(k) {
    if (!T.length) return;
    const g = k < 0 ? null : T[k];
    for (const [id] of PARTY) {
      dots.h[id].style.setProperty("--x", g ? g.honest_trust[id] : 0.5);
      dots.a[id].style.setProperty("--x", g ? g.adv_trust[id] : 0.5);
    }
    gameLab.textContent = k < 0 ? "–" : k + 1;
    [...pipBox.children].forEach((p, i) => p.classList.toggle("on", i <= k));
  }
  let timer = null;
  function play() {
    clearInterval(timer);
    let k = -1;
    setGame(-1);
    timer = setInterval(() => {
      k++;
      setGame(k);
      if (k >= T.length - 1) clearInterval(timer);
    }, 650);
  }
  root.querySelector(".rv-replay").addEventListener("click", play);
  pipBox.addEventListener("click", (e) => {
    const i = [...pipBox.children].indexOf(e.target);
    if (i >= 0) { clearInterval(timer); setGame(i); }
  });
  setGame(-1);

  /* ---------- panel 3: gap rows ---------- */
  const X0 = 30, X1 = 72;
  const xp = (v) => `${((v - X0) / (X1 - X0)) * 100}%`;
  function drawGap() {
    const rows = GAP[setting];
    root.querySelector(".rv-gap-rows").innerHTML = rows.map(([m, s, sp2, sp, d]) => `
      <div class="rv-gr">
        <span class="rv-gr-lab">${m}</span>
        <div class="rv-gr-track">
          <span class="rv-gr-fill" style="left:${xp(s)};--w:${xp(X0 + sp2 - s)}"></span>
          <span class="rv-gr-s" style="left:${xp(s)}" title="trust scores only: ${pct(s)}"></span>
          <span class="rv-gr-sp" style="left:${xp(sp)}" title="trust formed with guidance: ${pct(sp)}"></span>
          <span class="rv-gr-p" style="left:${xp(sp2)}" title="same trust + how to use it: ${pct(sp2)}"></span>
          <span class="rv-gr-d" style="left:${xp(sp2)}">+${(sp2 - s).toFixed(1)}</span>
        </div>
      </div>`).join("");
    const ds = rows.map((r) => r[4] * 100);
    const big = root.querySelector(".rv-gap-delta"), cap = root.querySelector(".rv-gap-cap");
    if (setting === "1HB+3G") {
      big.textContent = `${Math.round(Math.min(...ds))}–${Math.round(Math.max(...ds))}%`;
      cap.innerHTML = "of the failed games are avoided with the <b>same</b> trust, once agents know how to use it.";
    } else {
      big.textContent = `${Math.round(Math.min(...ds))}–${Math.round(Math.max(...ds))}%`;
      cap.innerHTML = "of failed games avoided. With two hard bargainers the gain is smaller and not statistically significant.";
    }
  }

  /* ---------- panel 4: what the agents see ---------- */
  function drawCtrl() {
    const c = CTRL[setting];
    const cols = [
      ["No trust", "no scores at all", [null, null, null, null], c.none, ""],
      ["Fixed trust", "everyone at 0.5", [0.5, 0.5, 0.5, 0.5], c.fixed, ""],
      ["Random trust", "new random scores", [0.24, 0.81, 0.47, 0.66], c.rand, ""],
      ["Learned trust", "formed from play", [0.86, 0.82, 0.9, 0.24], c.learned, "best"],
    ];
    const max = 80;
    root.querySelector(".rv-4").innerHTML = cols.map(([t, sub, v, a, cls]) => `
      <div class="rv-col ${cls}">
        <div class="rv-col-head"><b>${t}</b><small>${sub}</small></div>
        ${trustCard(v)}
        <div class="rv-col-bar"><span style="--h:${(a / max) * 100}%"><em>${pct(a)}</em></span></div>
      </div>`).join("");
  }

  function setSetting(s) {
    setting = s;
    root.querySelectorAll(".rv-seg button").forEach((b) => b.classList.toggle("on", b.dataset.set === s));
    drawGap();
    drawCtrl();
  }
  root.addEventListener("click", (e) => {
    const b = e.target.closest(".rv-seg button");
    if (b) setSetting(b.dataset.set);
  });
  setSetting(setting);

  /* ---------- reveal on scroll ---------- */
  const panels = root.querySelectorAll(".rv-panel");
  const reveal = (p) => {
    if (p.classList.contains("in")) return;
    p.classList.add("in");
    if (p.id === "rv1") play();
  };
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (e.isIntersecting) { reveal(e.target); io.unobserve(e.target); }
    }), { threshold: 0.3 });
    panels.forEach((p) => io.observe(p));
  } else panels.forEach(reveal);
})();
