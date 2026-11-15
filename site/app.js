// Renders charts and fills numbers from window.DATA (built by build.py).
(function () {
  const D = window.DATA;
  const tip = document.getElementById("tip");
  const fmt = (v, d = 1) => (v == null ? "–" : Number(v).toFixed(d));

  const row2 = (id) => D.exp2d.rows.find((r) => r.id === id) || {};
  const row3 = (id) => D.exp3d.rows.find((r) => r.id === id) || {};
  const sep3 = (label) => (D.exp3d.separation.find((r) => r.label === label) || {}).value;
  const long3 = (setup, readings) =>
    (D.exp3d.long.find((r) => r.setup === setup && r.readings === readings) || {}).depth;

  // ---- values bound into the prose: <span data-v="key" data-d="decimals">
  const V = {
    "updated": D.updated,
    "2d.frames.clean": row2("frames").clean,
    "2d.mask16.clean": row2("mask16").clean,
    "2d.random16.clean": row2("random16").clean,
    "2d.kalman.clean": row2("kalman").clean,
    "2d.kalman.min": Math.min(...["clean", "flicker", "texture", "distractor"].map((c) => row2("kalman")[c])),
    "2d.kalman.max": Math.max(...["clean", "flicker", "texture", "distractor"].map((c) => row2("kalman")[c])),
    "2d.mask16.flicker": row2("mask16").flicker,
    "2d.mask16.texture": row2("mask16").texture,
    "2d.norm16.flicker": row2("norm16").flicker,
    "2d.norm16.clean": row2("norm16").clean,
    "2d.norm64m.texture": row2("norm64_mixed").texture,
    "2d.framesm.clean": row2("frames_mixed").clean,
    "2d.framesm.flicker": row2("frames_mixed").flicker,
    "2d.framesm.texture": row2("frames_mixed").texture,
    "2d.norm64m.clean": row2("norm64_mixed").clean,
    "2d.norm64m.flicker": row2("norm64_mixed").flicker,
    "3d.mono.random": row3("mono_frames").random,
    "3d.stereo.random": row3("stereo_frames").random,
    "3d.mono.fixed": row3("mono_frames").fixed,
    "3d.stereo.fixed": row3("stereo_frames").fixed,
    "3d.cmono.random": row3("classical_mono").random,
    "3d.cstereo.random": row3("classical_stereo").random,
    "3d.cstereo.near": row3("classical_stereo").near,
    "3d.cstereo.far": row3("classical_stereo").far,
    "3d.stereo.near": row3("stereo_frames").near,
    "3d.stereo.far": row3("stereo_frames").far,
    "3d.mmono.random": row3("mono_mask_32").random,
    "3d.msep.random": row3("sep_mask_32").random,
    "3d.mbino.random": row3("bino_mask_32").random,
    "3d.sep3": sep3("2 eyes, 3 cm apart"),
    "3d.sep30": sep3("2 eyes, 30 cm apart"),
    "3d.long.sep32": long3("sep_mask", 32),
    "3d.long.bino32": long3("bino_mask", 32),
    "3d.probe.mono": (D.exp3d.probe.find((p) => p.label === "1 eye") || {}).trained,
    "3d.probe.stereo": (D.exp3d.probe.find((p) => p.label === "2 eyes") || {}).trained,
    "3d.probe.untr.mono": (D.exp3d.probe.find((p) => p.label === "1 eye") || {}).untrained,
    "3d.probe.untr.stereo": (D.exp3d.probe.find((p) => p.label === "2 eyes") || {}).untrained,
    "3d.probe.sup": D.exp3d.supervised_stereo_r2,
  };
  document.querySelectorAll("[data-v]").forEach((el) => {
    const v = V[el.dataset.v];
    el.textContent = typeof v === "number" ? fmt(v, el.dataset.d ? +el.dataset.d : 1) : v ?? "–";
  });

  // ---- tooltip
  function showTip(e, html) {
    tip.innerHTML = html;
    tip.style.display = "block";
    const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = x + "px";
    tip.style.top = e.clientY + 14 + "px";
  }
  const hideTip = () => (tip.style.display = "none");

  // ---- horizontal (grouped) bar chart
  // spec: {title, sub, unit, decimals, series:[{name, color}], rows:[{label, values:[], notes:[]}], note}
  function barChart(el, spec) {
    const all = spec.rows.flatMap((r) => r.values).filter((v) => v != null);
    const max = Math.max(...all) * 1.12;
    const d = spec.decimals ?? 1;
    let h = `<p class="chart-title">${spec.title}</p>`;
    if (spec.sub) h += `<p class="chart-sub">${spec.sub}</p>`;
    if (spec.series.length > 1)
      h += `<div class="legend">${spec.series
        .map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("")}</div>`;
    h += `<div class="bars">`;
    for (const r of spec.rows) {
      h += `<div class="rowlab">${r.label}</div><div class="track"><div class="group">`;
      r.values.forEach((v, i) => {
        if (v == null) return;
        const s = spec.series[i];
        const t = `${r.label}${spec.series.length > 1 ? " · " + s.name : ""}<br><b>${fmt(v, d)} ${spec.unit}</b>` +
          (r.notes && r.notes[i] ? `<br>${r.notes[i]}` : "");
        h += `<div class="bar" style="width:${(v / max) * 100}%;background:${s.color}" data-tip="${t.replace(/"/g, "&quot;")}">` +
          `<span class="barval">${fmt(v, d)}</span></div>`;
      });
      h += `</div></div>`;
    }
    h += `</div>`;
    if (spec.note) h += `<p class="axisnote">${spec.note}</p>`;
    // table view
    h += `<details><summary>Show as table</summary><div class="scroll"><table><tr><th></th>${spec.series
      .map((s) => `<th class="num">${s.name || spec.unit}</th>`).join("")}</tr>${spec.rows
      .map((r) => `<tr><td>${r.label}</td>${r.values.map((v) => `<td class="num">${fmt(v, d)}</td>`).join("")}</tr>`)
      .join("")}</table></div></details>`;
    el.innerHTML = h;
    el.querySelectorAll(".bar").forEach((b) => {
      b.addEventListener("mousemove", (e) => showTip(e, b.dataset.tip));
      b.addEventListener("mouseleave", hideTip);
    });
  }

  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const C1 = css("--series-1"), C2 = css("--series-2");

  // ---- heat table (log-scaled single-hue ramp)
  function heatTable(el, spec) {
    const steps = [0, 1, 2, 3, 4, 5, 6].map((i) => css(`--seq-${i}`));
    const lo = Math.log(spec.min), hi = Math.log(spec.max);
    const cell = (v) => {
      const t = Math.max(0, Math.min(1, (Math.log(v) - lo) / (hi - lo)));
      const i = Math.round(t * 6);
      return `<td class="cell" style="background:${steps[i]};color:${i >= 4 ? "#fff" : "#0b0b0b"}" ` +
        `data-tip="${spec.tip(v)}">${fmt(v, 1)}</td>`;
    };
    let h = `<p class="chart-title">${spec.title}</p><p class="chart-sub">${spec.sub}</p><div class="scroll"><table class="heat"><tr><th></th>` +
      spec.cols.map((c) => `<th style="text-align:center">${c.label}</th>`).join("") + `</tr>`;
    for (const r of spec.rows)
      h += `<tr><td>${r.label}</td>${spec.cols.map((c) => cell(r[c.key])).join("")}</tr>`;
    h += `</table></div><p class="axisnote">${spec.note}</p>`;
    el.innerHTML = h;
    el.querySelectorAll("[data-tip]").forEach((b) => {
      b.addEventListener("mousemove", (e) => showTip(e, b.dataset.tip));
      b.addEventListener("mouseleave", hideTip);
    });
  }

  // ================= 2D charts
  const pick = ["frames", "kalman", "mask16", "random16"].map(row2);
  barChart(document.getElementById("c2d-clean"), {
    title: "Clean scenes: how far off is the prediction 100 ms ahead?",
    sub: "Error in pixels on a 64×64 sensor (lower is better)",
    unit: "px", series: [{ name: "", color: C1 }],
    rows: pick.map((r) => ({ label: `${r.label} <span style="color:var(--text-muted)">(${r.readings} readings)</span>`,
      values: [r.clean], notes: [`${r.readings} readings per frame`] })),
  });

  const curve = D.exp2d.curve;
  barChart(document.getElementById("c2d-curve"), {
    title: "More readings help, and learning the masks helps more",
    sub: "Error 100 ms ahead (px), clean scenes, by number of readings per frame",
    unit: "px", series: [{ name: "Learned masks", color: C1 }, { name: "Random masks", color: C2 }],
    rows: curve.map((c) => ({ label: `${c.k} readings`, values: [c.learned, c.random] })),
    note: `For reference: the full 4096-pixel image gets ${fmt(row2("frames").clean)} px.`,
  });

  const conds = [
    { key: "clean", label: "Clean" }, { key: "fast", label: "Faster ball*" },
    { key: "flicker", label: "Flickering light" }, { key: "texture", label: "Busy background" },
    { key: "distractor", label: "2nd ball*" }];
  heatTable(document.getElementById("c2d-heat"), {
    title: "Robustness: what happens when conditions change?",
    sub: "Error 100 ms ahead in pixels. Darker = worse.",
    min: 0.5, max: 25, cols: conds,
    rows: ["frames", "frames_mixed", "events", "kalman", "mask16", "norm16", "norm64_mixed"].map(row2),
    tip: (v) => `<b>${fmt(v, 2)} px</b> error 100 ms ahead`,
    note: "* never seen during training by any model. Rows marked \"varied\" were trained on a mix of clean, flickering and busy scenes; all other rows on clean scenes only. Hover a cell for the exact value.",
  });

  // ================= 3D charts
  const rows3 = D.exp3d.rows;
  barChart(document.getElementById("c3d-main"), {
    title: "Does the second eye help judge distance?",
    sub: "Depth error 100 ms ahead, in cm (lower is better). Eyes 10 cm apart.",
    unit: "cm", series: [{ name: "Ball size known", color: C1 }, { name: "Ball size random", color: C2 }],
    rows: rows3.map((r) => ({ label: r.label, values: [r.fixed, r.random] })),
  });

  barChart(document.getElementById("c3d-sep"), {
    title: "Wider-set eyes judge depth better",
    sub: "CNN, random ball size: depth error 100 ms ahead (cm)",
    unit: "cm", series: [{ name: "", color: C1 }],
    rows: D.exp3d.separation.map((s) => ({ label: s.label, values: [s.value] })),
  });

  barChart(document.getElementById("c3d-probe"), {
    title: "Did depth emerge without being taught?",
    sub: "How well depth can be read out of the model's memory (R², 1 = perfectly, 0 = not at all). Models trained only to predict future images.",
    unit: "R²", decimals: 2,
    series: [{ name: "After training", color: C1 }, { name: "Untrained network (control)", color: C2 }],
    rows: D.exp3d.probe.map((p) => ({ label: p.label, values: [p.trained, p.untrained] })),
    note: `For comparison, a two-eye model trained directly on 3D positions reaches R² ${fmt(D.exp3d.supervised_stereo_r2, 2)}.`,
  });
})();
