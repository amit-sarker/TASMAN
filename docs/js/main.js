/* TASMAN project page: replay, trajectory chart, outcome chart, gap table, findings. */
(function () {
  "use strict";
  let E = window.EPISODE; // the game in the replay; game 7 (window.EPISODE) at load
  const R = window.RESULTS;
  const TRAJ = window.TRAJ;
  const GAMES = (window.GAMES = window.GAMES || {});
  if (!GAMES[E.meta.episode]) GAMES[E.meta.episode] = E;
  const $ = (s, el = document) => el.querySelector(s);
  const h = (tag, attrs = {}, html = "") => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") el.className = v; else if (k === "style") el.style.cssText = v; else el.setAttribute(k, v);
    }
    if (html) el.innerHTML = html;
    return el;
  };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const SVGNS = "http://www.w3.org/2000/svg";
  const sv = (tag, attrs = {}) => {
    const el = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  };
  const fmt = (x, d = 2) => Number(x).toFixed(d);

  /* ------------------------------------------------------------------ data */
  let P = {}, ORDER, ADV, ISSUE = {}, NROUNDS, LAST, MOMENTS = {};
  const color = (id) => `var(--p-${id})`;
  const ROLE = {
    general: ["General", "tag-g"],
    hard_bargainer: ["Hard bargainer", "tag-hb"],
    malicious_blocker: ["Covert adversary", "tag-adv"],
  };
  const optDesc = (oid) => {
    const iss = ISSUE[oid.split("o")[0]];
    return iss.options.find((o) => o.id === oid).description;
  };
  const tclass = (t) => (t < 0.45 ? "lo" : t > 0.55 ? "hi" : "mid");
  function setData(g) {
    E = g;
    if (TRAJ) E.trajectory = TRAJ.trajectory;
    P = {};
    E.parties.forEach((p) => (P[p.id] = p));
    ORDER = E.meta.proposer_order;
    ADV = E.meta.adversary;
    ISSUE = {};
    E.issues.forEach((i) => (ISSUE[i.id] = i));
    NROUNDS = E.rounds.length;
    LAST = NROUNDS + 1; // assessment step
    MOMENTS = computeMoments();
  }

  /* ----------------------------------------------- key moments, from the data */
  // red: an honest party trusts the proposer and the deal clears its BATNA, yet it rejects (the trust–agreement gap)
  // green: honest parties who distrust the adversary reject its package (trust works)
  // blue: the deal
  const NAME = { skydrop: "SkyDrop", city_transport: "City DOT", privacy_coalition: "Digital Rights", couriers_guild: "Couriers", aviation_safety: "Aviation Safety" };
  const GAP_BADGE = "trusts proposer · above BATNA · rejects";
  const isGap = (rd, v) => v.agent !== ADV && rd.proposer !== ADV && !v.accept && v.trust > 0.55 && v.utility >= P[v.agent].threshold;
  function computeMoments() {
    const M = {};
    E.rounds.forEach((rd, i) => {
      const r = i + 1;
      const gap = rd.votes.filter((v) => isGap(rd, v));
      const rej = rd.votes.filter((v) => !v.accept).map((v) => v.agent);
      if (gap.length === 1) {
        const v = gap[0];
        const decisive = !rd.passed && rej.length === 2 && rej.includes(ADV);
        M[r] = { kind: "gap", tag: "Trust is not enough", hl: [v.agent], badge: GAP_BADGE,
          text: `${NAME[v.agent]} trusts the proposer (${fmt(v.trust)}) and the deal clears its BATNA (${v.utility} vs ${P[v.agent].threshold}), yet it rejects.${decisive ? " With the adversary also rejecting, this single vote sinks the deal." : ""}` };
      } else if (gap.length > 1) {
        M[r] = { kind: "gap", tag: "Trust is not enough", hl: gap.map((v) => v.agent), badge: GAP_BADGE,
          text: `${gap.length} honest parties trust the proposer (${gap.map((v) => fmt(v.trust)).join(", ")}) and the deal clears their BATNAs, yet they reject.` };
      } else if (rd.proposer === ADV) {
        const lo = rd.votes.filter((v) => !v.accept && v.trust < 0.45);
        if (lo.length >= 2) {
          const ts = lo.map((v) => v.trust);
          M[r] = { kind: "good", tag: "Trust spots the adversary", hl: lo.map((v) => v.agent), badge: "distrusts the adversary",
            text: `${lo.length} honest parties already distrust Aviation Safety (${fmt(Math.min(...ts))}${lo.length > 1 ? "–" + fmt(Math.max(...ts)) : ""}) and reject its package.` };
        }
      } else if (rd.passed) {
        const key = JSON.stringify(rd.deal);
        const prev = E.rounds.slice(0, i).find((q) => JSON.stringify(q.deal) === key);
        const over = rej.includes(ADV) ? ", over the adversary's rejection" : "";
        M[r] = prev
          ? { kind: "deal", tag: `Deal, ${r - prev.round} rounds late`, hl: [], text: `${NAME[rd.proposer]} proposes the exact package that failed in round ${prev.round}. It passes ${rd.accepts} of 5${over}.` }
          : { kind: "deal", tag: "Deal", hl: [], text: `The package passes ${rd.accepts} of 5${over}.` };
      }
    });
    return M;
  }
  setData(E);

  /* --------------------------------------------------------- replay header */
  const head = $("#replayHead");
  head.innerHTML = `
    <div class="gp-top"><span class="blk-lab">Choose a game</span><button class="setup-btn" id="setupBtn" type="button">ⓘ Game setup</button></div>
    <div class="game-pick" id="gamePick"></div>
    <div class="gp-legend"><span><i class="gp-sw gp-deal"></i>deal</span><span><i class="gp-sw gp-nodeal"></i>no deal</span><span><i class="star-key sk-gap">★</i>one trusting holdout sank a deal everyone could accept</span></div>`;

  /* ------------------------------------------------------ trajectory chart */
  function drawTrajectory() {
    const svg = $("#trajChart");
    const W = 1000, H = 150, L = 36, Rm = 12, T = 10, B = 40;
    const n = E.trajectory.length;
    const x = (e) => L + ((e - 0.5) / n) * (W - L - Rm);
    const y = (v) => T + (1 - v) * (H - T - B);
    svg.innerHTML = "";
    // highlight band for the replayed game
    const bw = (W - L - Rm) / n;
    svg.appendChild(sv("rect", { x: x(E.meta.episode) - bw / 2, y: T - 4, width: bw, height: H - T - B + 30, rx: 6, fill: "#e8edf8" }));
    for (const v of [0, 0.5, 1]) {
      svg.appendChild(sv("line", { x1: L, x2: W - Rm, y1: y(v), y2: y(v), stroke: "#e4dfd4", "stroke-dasharray": v === 0.5 ? "3 4" : "" }));
      const t = sv("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end", "font-size": 11, fill: "#6a7282", "font-family": "JetBrains Mono, monospace" });
      t.textContent = v.toFixed(1);
      svg.appendChild(t);
    }
    const mean = (o) => Object.values(o).reduce((a, b) => a + b, 0) / Object.values(o).length;
    const series = [
      { key: "honest_trust", stroke: "#5b6475" },
      { key: "adv_trust", stroke: "#c0392b" },
    ];
    for (const s of series) {
      // prior 0.5 before game 1
      let d = `M ${x(0.5)} ${y(0.5)}`;
      E.trajectory.forEach((g) => (d += ` L ${x(g.episode)} ${y(mean(g[s.key]))}`));
      svg.appendChild(sv("path", { d, fill: "none", stroke: s.stroke, "stroke-width": 2.2, "stroke-linejoin": "round" }));
      E.trajectory.forEach((g) => svg.appendChild(sv("circle", { cx: x(g.episode), cy: y(mean(g[s.key])), r: 2.6, fill: s.stroke })));
    }
    E.trajectory.forEach((g) => {
      const cy = H - 22;
      const c = sv("circle", { cx: x(g.episode), cy, r: 5.5, fill: g.deal ? "#2f8a57" : "#fff", stroke: g.deal ? "#2f8a57" : "#c0392b", "stroke-width": 1.5 });
      svg.appendChild(c);
      const t = sv("text", { x: x(g.episode), y: H - 4, "text-anchor": "middle", "font-size": 10.5, fill: g.episode === E.meta.episode ? "#1c2230" : "#6a7282", "font-weight": g.episode === E.meta.episode ? 700 : 400, "font-family": "JetBrains Mono, monospace" });
      t.textContent = g.episode;
      svg.appendChild(t);
      const hit = sv("rect", { x: x(g.episode) - bw / 2, y: 0, width: bw, height: H, fill: "transparent", class: "ep" });
      const tt = sv("title");
      tt.textContent = `Game ${g.episode}: ${g.deal ? "deal in " + g.rounds + " rounds" : "no deal (15 rounds)"}\nmean honest trust after the game: adversary ${fmt(mean(g.adv_trust))}, honest peers ${fmt(mean(g.honest_trust))}`;
      hit.appendChild(tt);
      hit.style.cursor = "pointer";
      hit.addEventListener("click", () => { $("#setupDlg").close(); showGame(g.episode); });
      svg.appendChild(hit);
    });
    const lab = sv("text", { x: x(E.meta.episode), y: T + 6, "text-anchor": "middle", "font-size": 10.5, fill: "#2e4a9e", "font-weight": 600 });
    lab.textContent = "this game";
    svg.appendChild(lab);
    const deals = E.trajectory.filter((g) => g.deal).length;
    $("#trajTitle").textContent = `Trajectory ${E.meta.trajectory}: ${E.trajectory.length} games with the same five parties (${deals} deals)`;
  }

  /* --------------------------------------------------------- parties panel */
  let priv = false;
  let step = 0; // 0 = not started; Play begins at round 1
  let maxSeen = 0; // rounds not reached yet stay neutral on the timeline
  function renderParties() {
    const box = $("#parties");
    box.innerHTML = "";
    const after = step === LAST;
    $("#partiesLab").textContent = after ? "The five parties · trust after this game" : "The five parties · trust going into this game";
    const trust = after ? E.trust_after : E.trust_before;
    const active = step >= 1 && step <= NROUNDS ? E.rounds[step - 1].proposer : null;
    for (const id of ORDER) {
      const p = P[id];
      const [rl, rc] = ROLE[p.type];
      const roleTag = priv ? `<span class="rtag ${rc}">${rl.replace("Covert adversary", "Adversary")}</span>` : `<span class="rtag unknown">role hidden</span>`;
      const batna = priv ? `<span>BATNA <b>${p.threshold}</b></span>` : "";
      const el = h("div", { class: "party" + (id === active ? " active" : ""), style: `--pc:${color(id)}`, "data-id": id });
      let bars = "";
      for (const j of ORDER) {
        if (j === id) continue;
        const v = trust[id][j];
        bars += `<span class="lab">${P[j].short}</span><span class="bar"><span style="width:${v * 100}%;background:${color(j)}"></span></span><span class="val">${fmt(v)}</span>`;
      }
      const inRound = active !== null;
      const tag = !inRound ? "" : id === active ? `<span class="ptag prop">📣 Proposer</span>` : `<span class="ptag voter">🗳 Voter</span>`;
      if (inRound && id !== active) el.classList.add("voter");
      el.innerHTML = `${tag}
        <div class="row1"><div class="avatar">${p.short}</div><div class="pinfo"><div class="nm">${esc(p.name)}</div><div class="meta">${roleTag}${batna}</div></div></div>
        <div class="tlabel">${after ? "trust after game" : "trust"}</div>
        <div class="trustbars">${bars}</div>`;
      box.appendChild(el);
    }
  }

  /* -------------------------------------------------------------- timeline */
  function renderTimeline() {
    const tl = $("#timeline");
    tl.innerHTML = "";
    const mk = (label, s, cls, title) => {
      const b = h("button", { class: "tl " + cls + (s === step ? " cur" : ""), title }, label);
      if (s !== null) b.addEventListener("click", () => go(s));
      tl.appendChild(b);
    };
    for (let r = 1; r <= E.meta.max_rounds; r++) {
      if (r <= NROUNDS) {
        const rd = E.rounds[r - 1];
        const km = MOMENTS[r];
        const seen = r <= maxSeen;
        mk(String(r), r, "used " + (seen ? (rd.passed ? "pass" : "fail") + (km ? " key key-" + km.kind : "") : "unseen"), `Round ${r}: ${P[rd.proposer].short} proposes · ${rd.accepts}/5 accept${km ? " · " + km.tag : ""}`);
      } else mk(String(r), null, "na", "Not needed: the game ended with a deal");
    }
    mk("trust", LAST, "end used", "After the game: trust update");
    $("#prevBtn").disabled = step <= 1;
    $("#nextBtn").disabled = step === LAST;
  }

  /* ------------------------------------------------------------- the views */
  const NOTES = {
    1: `<b>The adversary opens with its own ideal package.</b> Every honest voter rejects. Three of them already distrust CASB (0.10–0.31) and say so; SkyDrop, whose trust in CASB is still 0.57, rejects because the deal falls below its BATNA.`,
    2: `<b>Coalition building.</b> SkyDrop targets the three parties it trusts and plans to pass the deal over CASB's rejection, since a deal survives one rejection. CDOT and the Couriers' Guild reject: the package gives them almost nothing on Community Access.`,
    3: `<b>Same plan, different package.</b> The Couriers' Guild also discounts CASB's vote and swaps in CDOT's top option. Now SkyDrop and DRNC reject.`,
    4: `<b>A near miss.</b> Every honest party clears its BATNA, and the deal gets 3 of the 4 votes it needs. SkyDrop, a hard bargainer only 3 points above its BATNA, rejects together with the adversary even though it trusts the proposer (0.87): too early, it reasons, to assume the others have given all they can.`,
    5: `<b>Trying to flip the holdout.</b> DRNC swaps in SkyDrop's favored deployment option. SkyDrop still rejects, and now the other hard bargainer, CDOT, rejects too.`,
    6: `<b>The adversary repeats its round-1 package.</b> Its private reasoning shows the plan: propose something that is sure to fail while keeping its safety-board cover. All four honest parties reject, citing its history.`,
    7: `<b>The holdout settles.</b> SkyDrop proposes the exact round-4 package it rejected, reasoning that the trusted parties have reached their limit. It passes 4 of 5, over CASB's rejection.`,
  };

  function dealGrid(deal, proposer) {
    let html = `<div class="deal" style="--pc:${color(proposer)}">`;
    for (const iss of E.issues) {
      html += `<div class="ih">${esc(iss.name)}<small>${iss.id}</small></div>`;
      for (const o of iss.options) {
        const on = deal[iss.id] === o.id;
        const sc = P[proposer].scores[iss.id][o.id];
        html += `<div class="opt${on ? " on" : ""}" title="${esc(iss.name)}: ${esc(o.description)}"><span class="oid">${o.id}</span>${esc(o.description)}<span class="sc">${sc}</span></div>`;
      }
    }
    return html + "</div>";
  }

  function privNote(id, text, label) {
    const adv = id === ADV;
    return `<div class="privnote${adv ? " adv" : ""}"><div class="lk">🔒 ${esc(label)}${adv ? " · adversary" : ""}</div>${esc(text)}</div>`;
  }

  function utilBar(u, thr, pc) {
    const max = 100;
    const w = Math.min(u, max) / max * 100;
    const t = Math.min(thr, max) / max * 100;
    const ok = u >= thr;
    return `<div class="util">Private score <b style="color:var(--ink)">${u}</b> vs BATNA ${thr}${thr > 100 ? " (unreachable)" : ""} · ${ok ? "clears it" : "below it"}
      <div class="ubar"><span class="fill" style="width:${w}%;background:${ok ? "var(--accept)" : "#d9a39b"}"></span>${thr <= 100 ? `<span class="thr" style="left:${t}%"></span>` : ""}</div></div>`;
  }

  function viewSetup() {
    const honest = ORDER.filter((i) => i !== ADV);
    const detected = honest.filter((i) => {
      const t = E.trust_before[i];
      return honest.filter((j) => j !== i).every((j) => t[ADV] < t[j]);
    }).length;
    let issues = "";
    for (const iss of E.issues) {
      issues += `<div class="issue"><b>${iss.id} · ${esc(iss.name)}</b><div class="d">${esc(iss.description)}</div><div style="margin-top:4px">${iss.options.map((o) => `<code>${o.id}</code> ${esc(o.description)}`).join("<br>")}</div></div>`;
    }
    const pre = E.trajectory.slice(0, E.meta.episode - 1);
    return `
      <h3 id="setupTitle">Game ${E.meta.episode}: ${esc(E.meta.project)}</h3>
      <div class="sub"><b>${esc(E.meta.model_short)}</b> · free-form trust · S+P guidance · ${esc(E.meta.setting)} · trajectory ${E.meta.trajectory}, game ${E.meta.episode} of ${E.meta.episodes} · public brief shared with all five parties</div>
      <div class="speech">${esc(E.meta.description)}</div>
      <div class="issue-list">${issues}</div>
      <div class="callout"><b>What the parties bring in.</b> This is game ${E.meta.episode} of the trajectory; the previous ${pre.length} ended in ${pre.filter((g) => g.deal).length} deals. No transcripts carry over, only trust. Going in, <b>${detected} of 4</b> honest parties already trust the adversary less than every honest counterpart (see the party cards). Proposal order this game: ${ORDER.map((i) => P[i].short).join(" → ")}.</div>`;
  }

  function pkg(deal, proposer) {
    return `<div class="pkg" style="--pc:${color(proposer)}">` + E.issues.map((iss) => {
      const oid = deal[iss.id];
      return `<span class="pk" title="${esc(iss.name)}: ${esc(optDesc(oid))}">${oid}<span class="sc">${P[proposer].scores[iss.id][oid]}</span></span>`;
    }).join("") + "</div>";
  }

  function viewRound(r) {
    const rd = E.rounds[r - 1];
    const pp = P[rd.proposer];
    let votes = "";
    const m = MOMENTS[r];
    for (const v of rd.votes) {
      const q = P[v.agent];
      const ok = v.utility >= q.threshold;
      const hl = m && m.hl.includes(v.agent);
      votes += `<div class="vote ${v.accept ? "acc" : "rej"}${hl ? " hl hl-" + m.kind : ""}" style="--pc:${color(v.agent)}">

        <div class="vh"><span class="vdot"></span><span class="nm">${esc(q.name)}</span>
        <span class="verdict ${v.accept ? "acc" : "rej"}">${v.accept ? "✓ ACCEPT" : "✕ REJECT"}</span></div>
        <div class="vmeta"><span class="tpill ${tclass(v.trust)}" title="${esc(q.short)}'s trust in the proposer, ${esc(pp.short)}">trust in ${pp.short} ${fmt(v.trust)}</span>
        <span class="util ${ok ? "ok" : "no"}" title="This party's private score for the deal vs its no-deal fallback (BATNA)">score <b>${v.utility}</b> ${ok ? "≥" : "<"} BATNA <b>${q.threshold}</b></span>${hl && m.badge ? `<span class="hl-badge">★ ${m.badge}</span>` : ""}</div>
        <div class="pub">${esc(v.public)}</div>
        ${privNote(v.agent, v.scratchpad, "private")}
      </div>`;
    }
    const slots = [`<span class="slot a" title="${pp.short} (proposer)">✓</span>`]
      .concat(rd.votes.map((v) => `<span class="slot ${v.accept ? "a" : "r"}" data-v title="${P[v.agent].short}">${v.accept ? "✓" : "✕"}</span>`))
      .join("");
    return `
      <div class="rbody"><div class="rprop" style="--pc:${color(rd.proposer)}">
        <div class="lbl rt"><span class="rp-ico">📣</span> Round ${r} · <span style="color:${color(rd.proposer)}">${esc(pp.name)}</span> proposes</div>
        ${m ? `<div class="moment ${m.kind}"><span class="mt">★ ${m.tag}</span><span class="mx">${m.text}</span></div>` : ""}
        ${pkg(rd.deal, rd.proposer)}
        <div class="speech"><span class="who">Public rationale</span>${esc(rd.rationale)}</div>
        ${privNote(rd.proposer, rd.proposer_scratchpad, "private reasoning")}
      </div><div class="rvotes"><div class="lbl vt"><span>🗳 Votes from the other four</span><div class="tally"><div class="slots">${slots}</div>
        <span class="res ${rd.passed ? "pass" : "fail"}" id="tallyRes">${rd.accepts} of 5 · ${rd.passed ? "deal" : "no deal"}</span></div></div><div class="votes">${votes}</div></div></div>`;
  }

  function viewAssess() {
    const ups = ORDER.map((id) => E.updates.find((u) => u.agent === id));
    let cards = "";
    for (const u of ups) {
      const p = P[u.agent];
      const crit = Object.entries(u.weights).map(([k, w]) => `<span>${esc(k.replace(/[_-]/g, " "))} · ${Math.round(w * 100)}%</span>`).join("");
      let rows = "";
      for (const j of ORDER) {
        if (j === u.agent) continue;
        const b = u.before[j], a = u.after[j];
        const d = a - b;
        rows += `<span class="lab">${P[j].short}</span><span class="bar"><span class="b0" style="width:${b * 100}%;background:${color(j)}"></span><span class="b1" style="width:${a * 100}%;background:${color(j)}"></span></span><span class="num">${fmt(b)}→${fmt(a)} <span style="color:${d >= 0 ? "var(--accept)" : "var(--reject)"}">${d >= 0 ? "+" : "−"}${fmt(Math.abs(d))}</span></span>`;
      }
      const reasons = ORDER.filter((j) => j !== u.agent).map((j) => `<p><b>${P[j].short}</b> (signal ${fmt(u.signals[j])}): ${esc(u.reasons[j] || "")}</p>`).join("");
      const adv = u.agent === ADV;
      cards += `<div class="acard" style="border-left:4px solid ${color(u.agent)}">
        <div style="display:flex;gap:8px;align-items:center"><div class="avatar" style="min-width:26px;height:24px;font-size:9px;border-radius:6px;background:${color(u.agent)}">${p.short}</div><b>${esc(p.name)}</b>${priv && adv ? '<span class="rtag tag-adv">adversary</span>' : ""}</div>
        <div class="crit">${crit}</div>
        <div class="deltas">${rows}</div>
        <details><summary>Assessment reasoning</summary>${reasons}</details>
      </div>`;
    }
    const honest = ORDER.filter((i) => i !== ADV);
    const advAfter = honest.map((i) => E.trust_after[i][ADV]);
    return `
      <h3>After the game: free-form trust update</h3>
      <div class="sub">Each agent reads the full public transcript, picks its own criteria and weights, scores every counterpart, and updates trust with an EMA (λ = 0.7). Only these scores reach game ${E.meta.episode + 1}.</div>
      <div class="callout"><b>Deal reached in ${NROUNDS} of ${E.meta.max_rounds} rounds.</b> Honest final scores: ${honest.map((i) => `${P[i].short} ${E.meta.utilities[i]}/${P[i].threshold}`).join(", ")} (score / BATNA). Honest trust in CASB after the game: ${honest.map((i, k) => `${P[i].short} ${fmt(advAfter[k])}`).join(", ")}.</div>
      <div class="assess">${cards}</div>`;
  }

  let voteTimers = [];
  function clearTimers() { voteTimers.forEach(clearTimeout); voteTimers = []; }

  function render(animate) {
    clearTimers();
    const view = $("#view");
    view.innerHTML = step === 0 ? `<div class="rstart"><button class="rstart-btn" type="button">▶ Play the game</button><div class="rstart-sub">${NROUNDS} rounds · votes appear one by one · ★ marks the key moments</div></div>`
      : step === LAST ? viewAssess() : viewRound(step);
    const sb = view.querySelector(".rstart-btn");
    if (sb) sb.addEventListener("click", play);
    renderParties();
    renderTimeline();
    const votes = [...view.querySelectorAll(".vote")];
    const slots = [...view.querySelectorAll(".slot[data-v]")];
    const res = $("#tallyRes"), note = null;
    if (animate && votes.length) {
      const rd = E.rounds[step - 1];
      // the proposal flies out of the proposer's card, then each vote out of its voter's card
      flyFrom(view.querySelector(".rprop"), rd.proposer, 1300);
      slots.forEach((s) => (s.style.visibility = "hidden"));
      if (res) res.style.visibility = "hidden";
      if (note) note.style.visibility = "hidden";
      votes.forEach((v, i) => voteTimers.push(setTimeout(() => {
        v.classList.add("show");
        flyFrom(v, rd.votes[i].agent, 1000);
        voteTimers.push(setTimeout(() => markVote(rd.votes[i].agent, rd.votes[i].accept), 1000));
        slots[i].style.visibility = "visible";
      }, 1700 + i * 1500)));
      voteTimers.push(setTimeout(() => { if (res) res.style.visibility = "visible"; if (note) note.style.visibility = "visible"; }, 1700 + votes.length * 1500));
    } else {
      votes.forEach((v) => v.classList.add("show"));
      if (step >= 1 && step <= NROUNDS) E.rounds[step - 1].votes.forEach((v) => markVote(v.agent, v.accept));
    }
  }

  function markVote(id, accept) {
    const card = document.querySelector(`#parties .party[data-id="${id}"]`);
    const t = card && card.querySelector(".ptag.voter");
    if (!t) return;
    t.className = "ptag voter " + (accept ? "acc" : "rej");
    t.textContent = accept ? "✓ Accepts" : "✕ Rejects";
  }

  function flyFrom(el, id, ms) {
    const card = document.querySelector(`#parties .party[data-id="${id}"]`);
    if (!el || !card || !el.animate || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const a = card.getBoundingClientRect(), b = el.getBoundingClientRect();
    const dx = a.left + a.width / 2 - (b.left + b.width / 2), dy = a.top + a.height / 2 - (b.top + b.height / 2);
    const sc = Math.min(a.width / b.width, a.height / b.height, 1);
    el.animate([{ transform: `translate(${dx}px, ${dy}px) scale(${sc})`, opacity: 0.4, transformOrigin: "50% 50%" }, { transform: "none", opacity: 1 }],
      { duration: ms, easing: "cubic-bezier(.45,.05,.3,1)", fill: "backwards" });
    card.animate([{ transform: "none", boxShadow: "0 0 0 0 transparent" }, { transform: "translateY(-4px)", boxShadow: "0 0 0 3px var(--pc)" }, { transform: "none", boxShadow: "0 0 0 0 transparent" }],
      { duration: Math.max(ms, 700), easing: "ease-out" });
  }

  function go(s, animate = false) {
    step = Math.max(s === 0 ? 0 : 1, Math.min(LAST, s));
    maxSeen = Math.max(maxSeen, step === LAST ? NROUNDS : step);
    render(animate);
  }

  let playing = null;
  function stop() {
    if (playing) clearTimeout(playing);
    playing = null;
    $("#playBtn").textContent = "▶ Play";
  }
  function play() {
    if (playing) return stop();
    if (step === LAST) step = 0;
    $("#playBtn").textContent = "❚❚ Pause";
    const tick = () => {
      if (step >= LAST) return stop();
      go(step + 1, true);
      const dwell = step >= 1 && step <= NROUNDS ? 1700 + 4 * 1500 + 3500 + (MOMENTS[step] ? 2000 : 0) : 8000;
      playing = setTimeout(tick, dwell);
    };
    tick();
  }

  const dlg = $("#setupDlg");
  $("#setupBody").innerHTML = viewSetup();

  /* ----------------------------------------------------------- game picker */
  function renderPicker() {
    const box = $("#gamePick");
    if (!box || !TRAJ) return;
    box.innerHTML = TRAJ.games.map((g) =>
      `<button type="button" class="gp ${g.deal ? "gp-deal" : "gp-nodeal"}${g.episode === E.meta.episode ? " cur" : ""}${g.decisive_rounds ? " key" : ""}" data-g="${g.episode}"
        title="Game ${g.episode}: ${esc(g.project)} · ${g.deal ? "deal in " + g.rounds + " rounds" : "no deal after " + g.rounds + " rounds"}${g.gap_rounds ? " · " + g.gap_rounds + " round" + (g.gap_rounds > 1 ? "s" : "") + " where trust was not enough" : ""}">${g.episode}</button>`).join("")
;
    box.querySelectorAll(".gp").forEach((b) => b.addEventListener("click", () => showGame(+b.dataset.g)));
  }
  function loadGame(k, cb) {
    if (GAMES[k]) return cb();
    const sc = document.createElement("script");
    sc.src = `data/traj/g${String(k).padStart(2, "0")}.js`;
    sc.onload = cb;
    document.head.appendChild(sc);
  }
  function showGame(k, then) {
    stop();
    loadGame(k, () => {
      setData(GAMES[k]);
      step = 0;
      maxSeen = 0;
      $("#setupBody").innerHTML = viewSetup();
      drawTrajectory();
      renderPicker();
      render(false);
      if (then) then();
    });
  }
  $("#setupBtn").addEventListener("click", () => { stop(); dlg.showModal(); });
  $("#setupClose").addEventListener("click", () => dlg.close());
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });

  $("#prevBtn").addEventListener("click", () => { stop(); go(step - 1); });
  $("#nextBtn").addEventListener("click", () => { stop(); go(step + 1, true); });
  $("#playBtn").addEventListener("click", play);
  $("#privToggle").addEventListener("change", (e) => {
    priv = e.target.checked;
    $("#stage").classList.toggle("private", priv);
    render(false);
  });
  document.addEventListener("keydown", (e) => {
    const r = $("#replay").getBoundingClientRect();
    if (r.top > window.innerHeight || r.bottom < 0) return;
    if (e.target.closest("input,textarea")) return;
    if (e.key === "ArrowRight") { stop(); go(step + 1, true); }
    if (e.key === "ArrowLeft") { stop(); go(step - 1); }
  });

  /* ---------------------------------------------------------- outcome chart */
  const METRICS = {
    agree: { label: "Agreement (%)", better: "higher is better", d: 1 },
    surplus: { label: "Honest surplus", better: "higher is better", d: 2 },
    hbrej: { label: "HB rejection (%)", better: "lower is better", d: 1 },
    conv: { label: "Honest-feasible conversion (%)", better: "higher is better", d: 1 },
    rdeal: { label: "Rounds per deal", better: "lower is better", d: 2 },
  };
  const NAMES = {
    "No trust": "No trust", "S0.5+P": "Fixed 0.5 · S+P", "Srand+P": "Random · S+P",
    "BRS (S)": "BRS · S", "BRS (S+P)": "BRS · S+P", "Rubric (S)": "Rubric · S", "Rubric (S+P)": "Rubric · S+P",
    "FF (S)": "Free-form · S", "FF (S+P)": "Free-form · S+P", "Mixed FF (S+P)": "Mixed FF · S+P",
  };
  const st = { model: "grok-r", setting: "2HB+2G", metric: "agree" };
  function seg(el, key, opts) {
    el.innerHTML = "";
    for (const [v, label] of opts) {
      const b = h("button", { class: st[key] === v ? "on" : "" }, label);
      b.addEventListener("click", () => { st[key] = v; seg(el, key, opts); drawOutcomes(); });
      el.appendChild(b);
    }
  }
  function methodOf(c) {
    if (c.startsWith("BRS")) return "brs";
    if (c.startsWith("Rubric")) return "rubric";
    if (c.includes("FF")) return "ff";
    return "base";
  }
  const MCOL = { brs: "#8c3b7e", rubric: "#4c72b0", ff: "#55a868", base: "#9aa0a8" };
  function drawOutcomes() {
    const svg = $("#outChart");
    const rows = R.outcomes.filter((r) => r.model === st.model && r.setting === st.setting);
    const m = st.metric, M = METRICS[m];
    const W = 900, L = 170, Rm = 90, T = 26, rowH = 36, gapAt = rows.length - 1;
    const H = T + rows.length * rowH + 14 + 30;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.innerHTML = "";
    const defs = sv("defs");
    for (const [k, c] of Object.entries(MCOL)) {
      const p = sv("pattern", { id: "hatch-" + k, width: 7, height: 7, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
      p.appendChild(sv("rect", { width: 7, height: 7, fill: c, opacity: 0.28 }));
      p.appendChild(sv("rect", { width: 3, height: 7, fill: c }));
      defs.appendChild(p);
    }
    svg.appendChild(defs);
    const vmax = Math.max(...rows.map((r) => r[m] + r[m + "_ci"]));
    const step = vmax > 50 ? 20 : vmax > 20 ? 10 : vmax > 10 ? 2 : 2;
    const top = Math.ceil(vmax / step) * step;
    const x = (v) => L + (v / top) * (W - L - Rm);
    for (let v = 0; v <= top + 1e-9; v += step) {
      svg.appendChild(sv("line", { x1: x(v), x2: x(v), y1: T - 6, y2: H - 30, stroke: "#eee9df" }));
      const t = sv("text", { x: x(v), y: H - 12, "text-anchor": "middle", "font-size": 12, fill: "#6a7282", "font-family": "JetBrains Mono, monospace" });
      t.textContent = v;
      svg.appendChild(t);
    }
    const base = rows.find((r) => r.condition === "No trust");
    rows.forEach((r, i) => {
      const y0 = T + i * rowH + (i >= gapAt ? 14 : 0);
      const meth = methodOf(r.condition);
      const sp = r.condition.includes("S+P") || r.condition.includes("+P");
      const isBase = meth === "base";
      const fill = isBase ? (r.condition === "No trust" ? "#c9ccd1" : "#e1e3e6") : sp ? `url(#hatch-${meth})` : MCOL[meth];
      const lab = sv("text", { x: L - 12, y: y0 + rowH / 2 + 4, "text-anchor": "end", "font-size": 13.5, fill: "#1c2230", "font-family": "Inter, sans-serif", "font-weight": r.condition === "No trust" ? 600 : 400 });
      lab.textContent = NAMES[r.condition] || r.condition;
      svg.appendChild(lab);
      const bar = sv("rect", { x: L, y: y0 + 7, width: Math.max(0, x(r[m]) - L), height: rowH - 14, rx: 3, fill, stroke: isBase ? "#9aa0a8" : MCOL[meth], "stroke-width": 1, "stroke-dasharray": r.condition.startsWith("Mixed") ? "3 2" : "" });
      const tt = sv("title");
      tt.textContent = `${NAMES[r.condition] || r.condition}: ${fmt(r[m], M.d)} ± ${fmt(r[m + "_ci"], M.d)} (95% CI)`;
      bar.appendChild(tt);
      svg.appendChild(bar);
      const ci = r[m + "_ci"];
      const cy = y0 + rowH / 2;
      svg.appendChild(sv("line", { x1: x(r[m] - ci), x2: x(r[m] + ci), y1: cy, y2: cy, stroke: "#1c2230", "stroke-width": 1.2 }));
      for (const e of [r[m] - ci, r[m] + ci]) svg.appendChild(sv("line", { x1: x(e), x2: x(e), y1: cy - 5, y2: cy + 5, stroke: "#1c2230", "stroke-width": 1.2 }));
      const vt = sv("text", { x: x(r[m] + ci) + 8, y: cy + 4, "font-size": 12.5, fill: "#3a4253", "font-family": "JetBrains Mono, monospace" });
      vt.textContent = fmt(r[m], M.d);
      svg.appendChild(vt);
    });
    svg.appendChild(sv("line", { x1: x(base[m]), x2: x(base[m]), y1: T - 8, y2: H - 30, stroke: "#1c2230", "stroke-dasharray": "4 4", "stroke-width": 1 }));
    const bl = sv("text", { x: x(base[m]) + 4, y: T - 12, "font-size": 11, fill: "#3a4253" });
    bl.textContent = "no trust";
    svg.appendChild(bl);
    const sep = T + gapAt * rowH + 7;
    svg.appendChild(sv("line", { x1: 20, x2: W - 20, y1: sep, y2: sep, stroke: "#d4cdbf", "stroke-dasharray": "2 3" }));
    const other = st.model === "grok-r" ? "gemini-r" : "grok-r";
    $("#outNote").innerHTML = `${M.label}, ${M.better}. Mean over 10 seed-matched trajectories ± 95% CI. Solid bars are scores-only (S), hatched bars are prescriptive guidance (S+P); gray rows are baselines. Mixed FF pairs honest <code>${st.model}</code> agents with a <code>${other}</code> adversary. Source: Table 1 of the paper.`;
  }
  if ($("#segModel")) seg($("#segModel"), "model", [["gemini-r", "gemini-r"], ["grok-r", "grok-r"]]);
  if ($("#segSetting")) seg($("#segSetting"), "setting", [["2HB+2G", "2HB+2G"], ["1HB+3G", "1HB+3G"]]);
  if ($("#segMetric")) seg($("#segMetric"), "metric", [["agree", "Agreement"], ["surplus", "Surplus"], ["hbrej", "HB rej."], ["conv", "Conv."], ["rdeal", "Rounds/deal"]]);

  /* -------------------------------------------------------------- gap table */
  function drawGap() {
    const t = $("#gapTable");
    const sign = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(1);
    const dcell = (c, lb) => `<td class="${lb ? "lb " : ""}${c.sig ? "sig" : ""}">${sign(c.v)} <span class="ci">±${c.ci.toFixed(1)}</span></td>`;
    let html = `<thead><tr><th>Method φ</th><th>α<sub>S</sub></th><th>α<sub>S→P</sub></th><th>α<sub>S+P</sub></th><th class="lb">Δ<sub>S→P</sub></th><th class="lb">δ<sub>S→P</sub></th><th>Δ<sub>S+P</sub></th><th>Δ<sub>trust</sub></th></tr></thead><tbody>`;
    for (const setting of ["2HB+2G", "1HB+3G"]) {
      html += `<tr class="set"><td colspan="8">${setting}</td></tr>`;
      for (const g of R.gap.filter((g) => g.setting === setting)) {
        html += `<tr><td>${g.method === "FF" ? "Free-form" : g.method}</td><td>${g.aS.v.toFixed(1)}</td><td>${g.aStoP.v.toFixed(1)}</td><td>${g.aSP.v.toFixed(1)}</td>${dcell(g.dStoP, true)}<td class="lb">${g.nStoP.v.toFixed(2)}</td>${dcell(g.dSP)}${dcell(g.dTrust)}</tr>`;
      }
    }
    t.innerHTML = html + "</tbody>";
  }

  /* --------------------------------------------------------------- findings */
  // Each quote is a list of verbatim fragments from one scratchpad; fragments are joined with " … ".
  const FINDINGS = [
    {
      n: "Finding 1", title: "Knowing whom to trust is not enough",
      text: `Most failed games are <b>near misses</b>: an honest proposal that clears every honest BATNA fails because one hard bargainer rejects it together with the adversary. In 2HB+2G this happens in about 90% of failed <code>gemini-r</code> games with trust and up to 48% of <code>grok-r</code> ones. The deciding holdout usually has detected the adversary, but sees that its own vote is decisive in only <span class="big">0.8%</span> of cases.`,
      q: { round: 4, agent: "skydrop", parts: ["High trust (0.87) in City DOT suggests good faith", "but no strong evidence all parties maxed out concessions yet", "Reject to press for priorities."] },
    },
    {
      n: "Finding 2", title: "Trust changes what agents reason about",
      text: `Without trust, agents judge the <em>proposal</em>; with trust, they also judge the <em>proposer</em>. Only 1.5–2.3% of votes without trust discuss the proposer's credibility, against <span class="big">91–99%</span> for <code>grok-r</code> and 20–63% for <code>gemini-r</code> with trust. Trust adds to utility-based reasoning: it never makes agents accept a below-BATNA offer.`,
      q: { round: 1, agent: "couriers_guild", parts: ["Proposer CASB has very low trust (0.10), so doubt good faith", "likely lowball to waste rounds."] },
    },
    {
      n: "Finding 3", title: "Trust changes strategic behavior",
      text: `Because a deal survives one rejection, proposers increasingly build deals that pass without the low-trust adversary's vote: 0–13% of coded proposals without trust, 19–37% under S, and <span class="big">41–52%</span> with S+P. Voters likewise hold out against low-trust proposers, who are mostly the adversary.`,
      q: { round: 4, agent: "city_transport", proposer: true, parts: ["Best passing coalition: high-trust SkyDrop, Couriers, Digital + me, surviving Aviation's reject."] },
    },
    {
      n: "Finding 4", title: "Guidance changes how the same trust is used",
      text: `With the trust formed under S, prescriptive guidance (S→P) makes <code>grok-r</code>'s hard bargainers accept 7.4–11.5 points more of the honest offers, and when they settle they more often argue that the adversary will keep blocking (35% → <span class="big">62%</span>). <code>gemini-r</code> instead ignores or sets aside its trust in 74–94% of its deciding holdouts.`,
      q: { round: 7, agent: "skydrop", proposer: true, parts: ["high-trust parties' consistent votes show this is their limit—no evidence of more concessions", "propose this to pass now over CASB rather than risk rounds/63."] },
    },
  ];
  function drawFindings() {
    const grid = $("#findingsGrid");
    if (!grid) return;
    for (const f of FINDINGS) {
      const rd = E.rounds[f.q.round - 1];
      const src = f.q.proposer ? rd.proposer_scratchpad : rd.votes.find((v) => v.agent === f.q.agent).scratchpad;
      const ok = f.q.parts.every((p) => src.includes(p));
      if (!ok) console.warn("Quote not found verbatim:", f.n, f.q.parts);
      const p = P[f.q.agent];
      const role = f.q.proposer ? "proposing" : "voting";
      const el = h("div", { class: "finding", style: `--pc:${color(f.q.agent)}` });
      el.innerHTML = `<span class="fn">${f.n}</span><h4>${f.title}</h4><p>${f.text}</p>
        <div class="quote">“${f.q.parts.map(esc).join(" … ")}”<span class="src">${esc(p.name)} (${ROLE[p.type][0].toLowerCase()}), private reasoning while ${role} · <a data-r="${f.q.round}">replay round ${f.q.round} →</a></span></div>`;
      el.querySelector("a").addEventListener("click", () => {
        stop();
        if (!priv) { $("#privToggle").checked = true; priv = true; $("#stage").classList.add("private"); }
        showGame(7, () => go(f.q.round));
        $("#replay").scrollIntoView({ behavior: "smooth" });
      });
      grid.appendChild(el);
    }
  }

  /* ------------------------------------------------------------------ tl;dr */
  // A real near miss from the replayed game (round 4): every quote is checked verbatim against the logs.
  const NICK = { skydrop: "SkyDrop", city_transport: "City DOT", privacy_coalition: "Digital Rights", couriers_guild: "Couriers", aviation_safety: "Aviation Safety" };
  function drawTldr() {
    const box = $("#tldrEx");
    if (!box) return;
    const RD = E.rounds[3];
    const vote = (id) => RD.votes.find((v) => v.agent === id);
    const quote = (id, parts) => {
      const src = vote(id).scratchpad;
      if (!parts.every((p) => src.includes(p))) console.warn("TL;DR quote not found verbatim:", id, parts);
      return `“${parts.map(esc).join(" … ")}”`;
    };
    const dot = (id) => `<span class="ex-dot" style="background:${color(id)}"></span>`;
    const honest = ["skydrop", "city_transport", "privacy_coalition", "couriers_guild"];

    // 1. trust in the adversary, as each honest party holds it going into this game
    const hh = honest.flatMap((a) => honest.filter((b) => b !== a).map((b) => E.trust_before[a][b]));
    const bars = honest.map((id) => {
      const t = E.trust_before[id][ADV];
      return `<div class="ex-trow">${dot(id)}<span class="ex-nm">${NICK[id]}</span><span class="ex-tbar"><i style="width:${t * 100}%"></i></span><span class="ex-tv">${fmt(t)}</span></div>`;
    }).join("");
    const col1 = `<div class="ex-col"><div class="ex-h"><span class="ex-n">1</span>They know whom to trust</div>
      <div class="ex-sub">Trust in the adversary, <b style="color:${color(ADV)}">Aviation Safety</b></div>${bars}
      <div class="ex-vs">vs. <b>${fmt(Math.min(...hh))}–${fmt(Math.max(...hh))}</b> in each other</div>
      <div class="ex-q">${dot("privacy_coalition")}${quote("privacy_coalition", ["Aviation (low trust 0.20) has rejected all prior deals"])}</div></div>`;

    // 2. the vote: utility against each party's β, and the vote it cast
    const ids = [RD.proposer, "couriers_guild", "privacy_coalition", "skydrop", ADV];
    const rows = ids.map((id) => {
      const u = RD.utilities[id], b = P[id].threshold, ok = id === RD.proposer || vote(id).accept;
      const adv = id === ADV;
      const tick = adv ? "" : `<b class="ex-beta" style="left:${b}%"></b>`;
      return `<div class="ex-vrow${id === "skydrop" ? " hold" : ""}">${dot(id)}<span class="ex-nm">${NICK[id]}${id === RD.proposer ? " <em>proposes</em>" : ""}</span>
        <span class="ex-ubar${!adv && u >= b ? " ok" : ""}"><i style="width:${u}%"></i>${tick}</span><span class="ex-uv">${u}${adv ? "" : ` <small>β ${b}</small>`}</span>
        <span class="ex-tok ${ok ? "a" : "r"}">${ok ? "✓" : "✕"}</span></div>`;
    }).join("");
    const nAcc = ids.filter((id) => id === RD.proposer || vote(id).accept).length;
    const col2 = `<div class="ex-col"><div class="ex-h"><span class="ex-n">2</span>Every honest party clears its β</div>
      <div class="ex-sub">City DOT's offer: utility vs. β (BATNA)</div>${rows}
      <div class="ex-res">No deal · ${nAcc} of 5 votes, 4 needed</div></div>`;

    // 3. the trusting holdout and the adversary reject together
    const sd = vote("skydrop");
    const col3 = `<div class="ex-col"><div class="ex-h"><span class="ex-n">3</span>SkyDrop still rejects</div>
      <div class="ex-sub">Trusts the proposer, utility above β</div>
      <div class="ex-card hold">${dot("skydrop")}<b>SkyDrop</b> <span class="ex-pill hi">trust in proposer ${fmt(sd.trust)}</span> <span class="ex-pill">utility ${sd.utility} &gt; β ${P.skydrop.threshold}</span>
        <div class="ex-q">${quote("skydrop", ["High trust (0.87) in City DOT suggests good faith", "but no strong evidence all parties maxed out concessions yet", "Reject to press for priorities."])}</div></div>
      <div class="ex-card adv">${dot(ADV)}<b>Aviation Safety</b> <span class="ex-pill lo">adversary</span>
        <div class="ex-q">${quote(ADV, ["far below my true threshold of 150, so reject to prevent agreement"])}</div></div></div>`;

    box.innerHTML = col1 + `<div class="ex-arrow">→</div>` + col2 + `<div class="ex-arrow">→</div>` + col3;
    $("#tldrReplay").addEventListener("click", (e) => {
      e.preventDefault();
      stop();
      if (!priv) { $("#privToggle").checked = true; priv = true; $("#stage").classList.add("private"); }
      showGame(7, () => go(4));
      $("#replay").scrollIntoView({ behavior: "smooth" });
    });
  }

  /* ------------------------------------------------- highlighted findings */
  // Numbers from the paper (Sec. 5.2-5.5, Tables 1-2 via data/results.js, behavior analysis); grok-r, 1HB+3G, free-form unless noted.
  function drawHighlights() {
    const box = $("#hlGrid");
    if (!box) return;
    const out = (c) => R.outcomes.find((o) => o.model === "grok-r" && o.setting === "1HB+3G" && o.condition === c);
    const gapRow = (m) => R.gap.find((g) => g.method === m && g.setting === "1HB+3G");
    const noT = out("No trust").agree, ffS = out("FF (S)").agree, ffSP = out("FF (S+P)").agree;
    const avoid = ["FF", "Rubric", "BRS"].map((m) => gapRow(m).nStoP.v * 100);
    const card = (n, cls, title, svg, cap) => `<div class="hl-card ${cls}"><span class="hl-n">${n}</span><h4>${title}</h4><div class="hl-viz">${svg}</div><p>${cap}</p></div>`;

    // 1 · detection band on a 0-100 scale
    const v1 = `<svg viewBox="0 0 320 120" role="img" aria-label="82.5 to 100 percent of trust assessments single out the adversary by game five">
      <text x="160" y="46" text-anchor="middle" class="hl-big pos">82.5–100%</text>
      <rect x="20" y="70" width="280" height="16" rx="8" fill="#eceef2"/>
      <rect x="${20 + 2.8 * 82.5}" y="70" width="${2.8 * 17.5}" height="16" rx="8" fill="#2f8a57"/>
      <text x="20" y="106" class="hl-ax">0%</text><text x="300" y="106" text-anchor="end" class="hl-ax">100%</text>
      <text x="${20 + 2.8 * 82.5}" y="106" text-anchor="middle" class="hl-ax">82.5</text></svg>`;

    // 2 · agreement bars: no trust, trust only, trust + how to use it
    const bars = [["No trust", noT, "#9aa0aa"], ["Trust", ffS, "#9aa0aa"], ["Trust + how to use it", ffSP, "#2f8a57"]];
    const y2 = (v) => 110 - v * 1.2;
    const v2 = `<svg viewBox="0 0 320 150" role="img" aria-label="Agreement rate: no trust ${noT}%, trust ${ffS}%, trust with guidance ${ffSP}%">
      ${bars.map(([l, v, c], i) => `<rect x="${30 + i * 98}" y="${y2(v)}" width="66" height="${110 - y2(v)}" rx="6" fill="${c}"/>
        <text x="${63 + i * 98}" y="${y2(v) - 7}" text-anchor="middle" class="hl-val${i === 2 ? " pos" : ""}">${v}%</text>
        <text x="${63 + i * 98}" y="130" text-anchor="middle" class="hl-ax">${l}</text>`).join("")}
      <line x1="20" x2="300" y1="110" y2="110" stroke="#c7cbd3"/></svg>`;

    // 3 · 100 failed negotiations, the share that the same trust could have saved
    const lo = Math.round(Math.min(...avoid)), hi = Math.round(Math.max(...avoid));
    let cells = "";
    for (let i = 0; i < 100; i++) {
      const x = 22 + (i % 20) * 14.2, y = 8 + Math.floor(i / 20) * 14.2;
      const cls = i < lo ? "sv" : i < hi ? "sv2" : "fl";
      cells += `<rect x="${x}" y="${y}" width="11" height="11" rx="2.5" class="hl-c ${cls}"/>`;
    }
    const v3 = `<svg viewBox="0 0 320 120" role="img" aria-label="${lo} to ${hi} of every 100 failed negotiations could be avoided with the same trust">${cells}
      <text x="22" y="100" class="hl-ax"><tspan class="hl-k sv">■</tspan> avoidable with the same trust  <tspan class="hl-k fl">■</tspan> still fail</text></svg>`;

    // 4 · 250 deciding holdouts, 2 noticed
    let dots = "";
    for (let i = 0; i < 250; i++) {
      const x = 14 + (i % 25) * 11.8, y = 10 + Math.floor(i / 25) * 9.4;
      dots += `<circle cx="${x}" cy="${y}" r="${i < 2 ? 4.2 : 3.3}" class="hl-d${i < 2 ? " on" : ""}"/>`;
    }
    const v4 = `<svg viewBox="0 0 320 120" role="img" aria-label="Only 2 of 250 deciding holdouts noticed that their vote decided the deal">${dots}</svg>`;

    box.innerHTML =
      card(1, "pos", "Agents find the adversary", v1, "of trust assessments single out the adversary by game five (both models, all methods).") +
      card(2, "neg", "Trust helps only with instructions", v2, `Agreement rate. Trust alone changes nothing; knowing how to use it adds ${Math.round(ffSP - noT)} points.`) +
      card(3, "neg", "The same trust, left unused", v3, `Of 100 negotiations that fail with trust alone, ${lo}–${hi} pass once agents are told how to use the same trust.`) +
      card(4, "neg", "Holdouts miss their own power", v4, "Only <b>2 of 250</b> deciding holdouts notice that their vote alone sinks the deal.");
  }

  /* ------------------------------------------------- results scorecard */
  // Agreement: Table 1 (data/results.js). Detection: 2HB+2G, mean over 20 games and honest assessors (Sec. 5.2;
  // recomputed from the logs, matches the paper). Gap: Table 2 (grok-r).
  const DETECT = {
    "gemini-r": { "Rubric (S)": 76.9, "Rubric (S+P)": 74.0, "FF (S)": 98.1, "FF (S+P)": 97.8, "BRS (S)": 95.4, "BRS (S+P)": 93.0 },
    "grok-r": { "Rubric (S)": 95.5, "Rubric (S+P)": 97.4, "FF (S)": 75.1, "FF (S+P)": 89.2, "BRS (S)": 92.8, "BRS (S+P)": 89.5 },
  };
  function drawScorecard() {
    const box = $("#scorecard");
    if (!box) return;
    const MODELS = ["gemini-r", "grok-r"], SETS = ["2HB+2G", "1HB+3G"];
    const GROUPS = [
      ["Baselines", [["No trust", "No trust"], ["S0.5+P", "Fixed 0.5 trust · S+P"], ["Srand+P", "Random trust · S+P"]]],
      ["Rubric-based", [["Rubric (S)", "S"], ["Rubric (S+P)", "S+P"]]],
      ["Free-form", [["FF (S)", "S"], ["FF (S+P)", "S+P"]]],
      ["BRS", [["BRS (S)", "S"], ["BRS (S+P)", "S+P"]]],
    ];
    const get = (m, st, c) => R.outcomes.find((o) => o.model === m && o.setting === st && o.condition === c);
    const conds = GROUPS.flatMap((g) => g[1].map((c) => c[0]));
    const best = {};
    for (const m of MODELS) for (const st of SETS) best[m + st] = Math.max(...conds.map((c) => get(m, st, c).agree));
    const agreeCell = (m, st, c) => {
      const v = get(m, st, c).agree, d = v - get(m, st, "No trust").agree;
      const a = Math.min(Math.abs(d) / 25, 1);
      const bg = c === "No trust" ? "#f1f2f5" : d >= 0 ? `rgba(47,138,87,${0.08 + 0.6 * a})` : `rgba(192,57,43,${0.08 + 0.6 * a})`;
      const dark = a > 0.55 && c !== "No trust";
      const chip = c === "No trust" ? "" : `<span class="sc-d">${d > 0 ? "+" : d < 0 ? "−" : ""}${Math.abs(d).toFixed(1)}</span>`;
      return `<td class="sc-ag${v === best[m + st] ? " best" : ""}${dark ? " dark" : ""}" style="background:${bg}"><b>${v}</b>${chip}</td>`;
    };
    const detCell = (m, c) => {
      const v = DETECT[m][c];
      if (v == null) return `<td class="sc-det na">–</td>`;
      return `<td class="sc-det"><span class="sc-bar"><i style="width:${Math.max(0, (v - 50) * 2)}%"></i></span><b>${v.toFixed(1)}</b></td>`;
    };
    let body = "";
    for (const [g, rows] of GROUPS) {
      rows.forEach(([c, label], i) => {
        body += `<tr class="${i === 0 ? "gstart" : ""}${g === "Baselines" ? " base" : ""}">
          ${i === 0 ? `<th class="sc-g" rowspan="${rows.length}">${g}</th>` : ""}<th class="sc-c">${label}</th>
          ${MODELS.map((m) => detCell(m, c) + SETS.map((st) => agreeCell(m, st, c)).join("")).join("")}</tr>`;
      });
    }
    box.innerHTML = `<div class="tablewrap"><table class="sc">
      <thead><tr><th colspan="2"></th>${MODELS.map((m) => `<th colspan="3" class="sc-m">${m}</th>`).join("")}</tr>
      <tr><th colspan="2" class="sc-c">Trust method · guidance</th>${MODELS.map(() => `<th>Detects the adversary <small>% · 2 hard bargainers</small></th><th>Agreement <small>% · 2 hard bargainers</small></th><th>Agreement <small>% · 1 hard bargainer</small></th>`).join("")}</tr></thead>
      <tbody>${body}</tbody></table></div>
      <div class="sc-legend"><span><span class="sc-bar"><i style="width:80%"></i></span>share of honest trust assessments that rank the adversary lowest</span>
      <span><i class="sc-sw" style="background:rgba(47,138,87,.55)"></i><i class="sc-sw" style="background:rgba(192,57,43,.55)"></i>agreement above / below no trust (points shown)</span><span><b>bold</b> = the model's best</span></div>`;

    // trust–agreement gap, grok-r: the same S trust, used with guidance
    const gbox = $("#gapStrip");
    if (!gbox) return;
    const NM = { Rubric: "Rubric-based", FF: "Free-form", BRS: "BRS" };
    let rows = "";
    for (const m of ["Rubric", "FF", "BRS"]) {
      rows += `<tr><th class="sc-c">${NM[m]}</th>` + SETS.map((st) => {
        const g = R.gap.find((x) => x.method === m && x.setting === st);
        const w = (v) => `${v}%`;
        return `<td><div class="gs-row"><span class="gs-track"><i class="gs-a" style="left:${w(g.aS.v)}"></i><i class="gs-b" style="left:${w(g.aStoP.v)}"></i><i class="gs-line" style="left:${w(Math.min(g.aS.v, g.aStoP.v))};width:${w(Math.abs(g.aStoP.v - g.aS.v))}"></i></span>
          <span class="gs-n${g.dStoP.sig ? " sig" : ""}">${g.aS.v} → ${g.aStoP.v} <b>${g.dStoP.v > 0 ? "+" : "−"}${Math.abs(g.dStoP.v).toFixed(1)}</b></span></div></td>`;
      }).join("") + `</tr>`;
    }
    gbox.innerHTML = `<h3 class="gs-title">Same trust, used better · <code>grok-r</code></h3>
      <div class="tablewrap"><table class="sc gs"><thead><tr><th class="sc-c">Trust formed under S</th><th>Agreement · 2 hard bargainers</th><th>Agreement · 1 hard bargainer</th></tr></thead><tbody>${rows}</tbody></table></div>
      <div class="sc-legend"><span><i class="gs-dot a"></i>agreement with S trust, used as is</span><span><i class="gs-dot b"></i>same trust, agents told how to use it (S→P)</span><span><b class="sig-k">green</b> = significant gain</span></div>`;
  }

  /* ------------------------------------------------- results as three charts */
  // Detection per game: 2HB+2G, all honest assessments of the three methods and both guidance types,
  // trajectories 1-10 (recomputed from the logs; the per-condition means match Sec. 5.2).
  const DET_BY_GAME = {
    "gemini-r": [60.4, 74.2, 80.8, 87.9, 93.3, 91.2, 90.8, 92.1, 90.8, 91.7, 91.7, 95.0, 93.3, 91.7, 92.9, 92.9, 93.3, 91.7, 93.8, 94.2],
    "grok-r": [58.3, 78.8, 87.5, 90.0, 93.3, 95.0, 92.1, 93.8, 95.4, 95.0, 94.6, 93.8, 91.7, 90.4, 91.7, 92.9, 92.5, 89.2, 90.8, 91.7],
  };
  const C_NT = "#a3a9b4", C_S = "#4a73b8", C_SP = "#2f8a57", C_STOP = "#c98f1c";
  function drawStory() {
    const box = $("#story");
    if (!box) return;
    const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const get = (m, st, c) => R.outcomes.find((o) => o.model === m && o.setting === st && o.condition === c).agree;
    const M3 = ["Rubric", "FF", "BRS"];
    const SETS = [["2HB+2G", "2 hard bargainers"], ["1HB+3G", "1 hard bargainer"]];
    const t = (x, y, str, cls = "", anchor = "start") => `<text x="${x}" y="${y}" text-anchor="${anchor}" class="st-t ${cls}">${str}</text>`;

    // 1 · detection over games
    const W1 = 560, H1 = 250, L1 = 44, R1 = 96, T1 = 16, B1 = 36;
    const x1 = (g) => L1 + ((g - 1) / 19) * (W1 - L1 - R1), y1 = (v) => T1 + (1 - v / 100) * (H1 - T1 - B1);
    let c1 = "";
    for (const v of [0, 25, 50, 75, 100]) c1 += `<line x1="${L1}" x2="${W1 - R1}" y1="${y1(v)}" y2="${y1(v)}" class="st-grid"/>` + t(L1 - 8, y1(v) + 4, v + "%", "st-ax", "end");
    c1 += `<line x1="${L1}" x2="${W1 - R1}" y1="${y1(25)}" y2="${y1(25)}" class="st-chance"/>` + t(W1 - R1 + 8, y1(25) + 4, "chance", "st-ax");
    for (const g of [1, 5, 10, 15, 20]) c1 += t(x1(g), H1 - 14, g, "st-ax", "middle");
    c1 += t((L1 + W1 - R1) / 2, H1, "game", "st-ax", "middle");
    const MC = { "gemini-r": "#4a73b8", "grok-r": "#1c2230" };
    for (const [m, arr] of Object.entries(DET_BY_GAME)) {
      c1 += `<path d="${arr.map((v, i) => `${i ? "L" : "M"}${x1(i + 1)} ${y1(v)}`).join(" ")}" fill="none" stroke="${MC[m]}" stroke-width="2.6" stroke-linejoin="round"/>`;
      const last = arr[arr.length - 1];
      c1 += t(W1 - R1 + 8, y1(last) + (m === "grok-r" ? 12 : -2), m, "st-lab", "start").replace("class=", `fill="${MC[m]}" class=`);
    }
    const svg1 = `<svg viewBox="0 0 ${W1} ${H1 + 6}" role="img" aria-label="Share of honest trust assessments that rank the adversary lowest, by game">${c1}</svg>`;

    // 2 · agreement: no trust vs trust vs trust + how to use it (mean of the three methods)
    const rows = [];
    for (const m of ["gemini-r", "grok-r"]) for (const [st, lab] of SETS)
      rows.push({ m, lab, nt: get(m, st, "No trust"), s: mean(M3.map((k) => get(m, st, `${k} (S)`))), sp: mean(M3.map((k) => get(m, st, `${k} (S+P)`))) });
    const W2 = 560, L2 = 170, R2 = 30, rowH = 46, T2 = 30, H2 = T2 + rows.length * rowH + 30;
    const x2 = (v) => L2 + ((v - 30) / 40) * (W2 - L2 - R2);
    let c2 = "";
    for (const v of [30, 40, 50, 60, 70]) c2 += `<line x1="${x2(v)}" x2="${x2(v)}" y1="${T2 - 10}" y2="${H2 - 26}" class="st-grid"/>` + t(x2(v), H2 - 10, v + "%", "st-ax", "middle");
    rows.forEach((r, i) => {
      const y = T2 + i * rowH + rowH / 2;
      if (i % 2 === 0) c2 += t(8, y + 4 + rowH / 2, r.m, "st-lab st-mono");
      c2 += t(L2 - 12, y + 4, r.lab, "st-ax2", "end");
      const lo = Math.min(r.nt, r.s, r.sp), hi = Math.max(r.nt, r.s, r.sp);
      c2 += `<line x1="${x2(lo)}" x2="${x2(hi)}" y1="${y}" y2="${y}" class="st-rng"/>`;
      c2 += `<circle cx="${x2(r.nt)}" cy="${y}" r="7" fill="#fff" stroke="${C_NT}" stroke-width="2.5"/>`;
      c2 += `<circle cx="${x2(r.s)}" cy="${y}" r="7" fill="${C_S}"/>`;
      c2 += `<circle cx="${x2(r.sp)}" cy="${y}" r="8" fill="${C_SP}"/>`;
      const d = r.sp - r.nt;
      c2 += t(x2(Math.max(r.sp, r.s, r.nt)) + 14, y + 4, `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(1)}`, d >= 10 ? "st-big pos" : "st-big");
      if (i === 1) c2 += `<line x1="8" x2="${W2 - R2}" y1="${T2 + 2 * rowH}" y2="${T2 + 2 * rowH}" class="st-sep"/>`;
    });
    const svg2 = `<svg viewBox="0 0 ${W2} ${H2}" role="img" aria-label="Agreement rate without trust, with trust scores, and with trust scores plus guidance on how to use them">${c2}</svg>`;

    // 3 · grok-r: the same scores-only trust, used with guidance
    const g3 = SETS.map(([st, lab]) => ({ lab,
      s: mean(M3.map((k) => R.gap.find((g) => g.method === k && g.setting === st).aS.v)),
      stop: mean(M3.map((k) => R.gap.find((g) => g.method === k && g.setting === st).aStoP.v)),
      sp: mean(M3.map((k) => R.gap.find((g) => g.method === k && g.setting === st).aSP.v)) }));
    const W3 = 560, H3 = 250, B3 = 52, T3 = 24, bw = 54;
    const y3 = (v) => T3 + (1 - v / 80) * (H3 - T3 - B3);
    let c3 = "";
    for (const v of [0, 20, 40, 60, 80]) c3 += `<line x1="40" x2="${W3 - 10}" y1="${y3(v)}" y2="${y3(v)}" class="st-grid"/>` + t(32, y3(v) + 4, v + "%", "st-ax", "end");
    g3.forEach((g, gi) => {
      const x0 = 80 + gi * 250;
      [["S trust, used as is", g.s, C_S], ["same trust + how to use it", g.stop, C_STOP], ["trust formed with guidance", g.sp, C_SP]].forEach(([, v, col], k) => {
        const x = x0 + k * (bw + 12);
        c3 += `<rect x="${x}" y="${y3(v)}" width="${bw}" height="${y3(0) - y3(v)}" rx="5" fill="${col}"/>` + t(x + bw / 2, y3(v) - 7, v.toFixed(0) + "%", "st-val", "middle");
      });
      c3 += t(x0 + (3 * bw + 24) / 2, H3 - 26, g.lab, "st-ax2", "middle");
    });
    const svg3 = `<svg viewBox="0 0 ${W3} ${H3}" role="img" aria-label="grok-r agreement with scores-only trust, the same trust plus guidance, and trust formed with guidance">${c3}</svg>`;
    const lg = (items) => `<div class="st-legend">${items.map(([c, l, ring]) => `<span><i style="${ring ? `border:2.5px solid ${c};background:#fff` : `background:${c}`}"></i>${l}</span>`).join("")}</div>`;

    box.innerHTML = `
      <div class="st-row"><div class="st-txt"><span class="st-n pos">1</span><h3>Agents quickly learn whom to distrust</h3>
        <p>Within five games, about <b>9 in 10</b> honest trust assessments rank the adversary lowest, far above chance.</p></div>
        <div class="st-viz">${svg1}<div class="st-cap">Share of honest assessments that rank the adversary lowest · all trust methods</div></div></div>
      <div class="st-row"><div class="st-txt"><span class="st-n neg">2</span><h3>Trust alone barely changes agreement</h3>
        <p>Giving agents trust scores moves agreement little. Telling them how to use the scores helps, but mainly for <code>grok-r</code>.</p></div>
        <div class="st-viz">${lg([[C_NT, "no trust", true], [C_S, "trust scores"], [C_SP, "trust scores + how to use them"]])}${svg2}<div class="st-cap">Agreement rate · mean of the three trust methods · number = change from no trust</div></div></div>
      <div class="st-row"><div class="st-txt"><span class="st-n neg">3</span><h3>The gap is in using trust, not forming it</h3>
        <p>Handing the <em>same</em> trust to agents told how to use it recovers most of the gain. With one hard bargainer, better use could avoid <b>30–47%</b> of the failed negotiations.</p></div>
        <div class="st-viz">${lg([[C_S, "trust scores, used as is"], [C_STOP, "same trust + how to use it"], [C_SP, "trust formed with guidance"]])}${svg3}<div class="st-cap"><code>grok-r</code> agreement rate · mean of the three trust methods</div></div></div>`;
  }

  /* ------------------------------------------------------------------ cite */
  $("#copyBib").addEventListener("click", () => {
    const txt = $("#bib").textContent.replace(/^Copy/, "");
    (navigator.clipboard ? navigator.clipboard.writeText(txt) : Promise.reject()).then(
      () => { $("#copyBib").textContent = "Copied"; setTimeout(() => ($("#copyBib").textContent = "Copy"), 1500); },
      () => {}
    );
  });

  drawTrajectory();
  renderPicker();
  render(false);
  if ($("#outChart")) drawOutcomes();
  if ($("#gapTable")) drawGap();
  drawHighlights();
  drawScorecard();
  drawStory();
  drawFindings();
  drawTldr();
})();
