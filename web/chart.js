/* Minimal dependency-free canvas line chart for the outbreak page.
   Supports a left axis (population counts) and a right axis (Rt), a set of
   toggleable line series, horizontal reference lines (e.g. ICU capacity) and
   a moving playhead so the epidemic can be drawn day-by-day or scrubbed. */

const COL = {
  ink: "#0c0c0d",
  grid: "rgba(236,231,221,0.10)",
  axis: "rgba(236,231,221,0.45)",
  text: "#9b978e",
};

export class LineChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.data = null;
    this.pad = { l: 56, r: 48, t: 14, b: 30 };
    this._resize();
    this._onResize = () => { this._resize(); this.render(this._upto); };
    window.addEventListener("resize", this._onResize);
  }

  destroy() { window.removeEventListener("resize", this._onResize); }

  _resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = this.canvas.clientWidth || 600;
    const h = Math.max(280, Math.round(w * 0.52));
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.canvas.style.height = h + "px";
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.W = w; this.H = h;
  }

  setData(data) {
    this.data = data;
    this._computeScales();
    this._upto = data.days.length - 1;
    this.render(this._upto);
  }

  _computeScales() {
    const d = this.data;
    let leftMax = 1, rtMax = 2;
    d.series.forEach((s) => {
      if (s.axis === "right") {
        s.values.forEach((v) => { if (v > rtMax) rtMax = v; });
      } else {
        s.values.forEach((v) => { if (v > leftMax) leftMax = v; });
      }
    });
    (d.hlines || []).forEach((l) => {
      if (l.axis !== "right" && l.y > leftMax) leftMax = l.y;
    });
    this.leftMax = leftMax * 1.08;
    this.rtMax = Math.min(Math.max(rtMax * 1.1, 2), 16);
    this.xMax = d.days[d.days.length - 1] || 1;
  }

  _x(day) {
    const { l, r } = this.pad;
    return l + (day / this.xMax) * (this.W - l - r);
  }
  _yLeft(v) {
    const { t, b } = this.pad;
    return t + (1 - v / this.leftMax) * (this.H - t - b);
  }
  _yRight(v) {
    const { t, b } = this.pad;
    return t + (1 - v / this.rtMax) * (this.H - t - b);
  }

  render(upto) {
    if (!this.data) return;
    this._upto = upto;
    const ctx = this.ctx;
    const d = this.data;
    const { l, r, t, b } = this.pad;
    ctx.clearRect(0, 0, this.W, this.H);

    // grid + left axis ticks
    ctx.strokeStyle = COL.grid;
    ctx.fillStyle = COL.text;
    ctx.lineWidth = 1;
    ctx.font = "11px 'IBM Plex Mono', monospace";
    ctx.textBaseline = "middle";
    const rows = 4;
    for (let i = 0; i <= rows; i++) {
      const v = (this.leftMax / rows) * i;
      const y = this._yLeft(v);
      ctx.beginPath();
      ctx.moveTo(l, y); ctx.lineTo(this.W - r, y);
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(fmtCompact(v), l - 8, y);
    }
    // right axis ticks (Rt)
    if (d.series.some((s) => s.axis === "right")) {
      ctx.textAlign = "left";
      for (let i = 0; i <= rows; i++) {
        const v = (this.rtMax / rows) * i;
        ctx.fillStyle = COL.text;
        ctx.fillText(v.toFixed(1), this.W - r + 8, this._yRight(v));
      }
    }

    // x axis labels
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const xticks = 5;
    for (let i = 0; i <= xticks; i++) {
      const day = (this.xMax / xticks) * i;
      ctx.fillStyle = COL.text;
      ctx.fillText("d" + Math.round(day), this._x(day), this.H - b + 8);
    }

    // reference lines (capacity)
    (d.hlines || []).forEach((line) => {
      const y = line.axis === "right" ? this._yRight(line.y) : this._yLeft(line.y);
      ctx.save();
      ctx.strokeStyle = line.color;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(this.W - r, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = line.color;
      ctx.textAlign = "left"; ctx.textBaseline = "bottom";
      ctx.fillText(line.label, l + 6, y - 3);
      ctx.restore();
    });

    // series
    const cut = Math.max(0, Math.min(upto, d.days.length - 1));
    d.series.forEach((s) => {
      if (s.hidden) return;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width || 2;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i <= cut; i++) {
        const x = this._x(d.days[i]);
        const y = s.axis === "right" ? this._yRight(s.values[i]) : this._yLeft(s.values[i]);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    // playhead
    if (cut < d.days.length - 1) {
      const x = this._x(d.days[cut]);
      ctx.strokeStyle = "rgba(236,231,221,0.5)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, this.H - b); ctx.stroke();
    }

    // axis frame
    ctx.strokeStyle = COL.axis;
    ctx.beginPath();
    ctx.moveTo(l, t); ctx.lineTo(l, this.H - b); ctx.lineTo(this.W - r, this.H - b);
    ctx.stroke();
  }
}

function fmtCompact(v) {
  if (v >= 1e9) return (v / 1e9).toFixed(1) + "B";
  if (v >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(0) + "k";
  return Math.round(v).toString();
}
