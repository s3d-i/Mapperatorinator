from __future__ import annotations


CANVAS_PREVIEW_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage 1 Mania Preview</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111318;
      --panel: #1b2028;
      --line: #394150;
      --text: #e9edf5;
      --muted: #9aa5b5;
      --accent: #2fb3a6;
      --tap: #f3d35c;
      --hold: #58a6ff;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text); font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { overflow: hidden; }
    .shell { display: grid; grid-template-rows: 48px 1fr 72px; height: 100vh; }
    header, footer { display: flex; align-items: center; gap: 16px; padding: 0 18px; background: var(--panel); border-color: var(--line); }
    header { border-bottom: 1px solid var(--line); }
    footer { border-top: 1px solid var(--line); }
    .title { font-weight: 650; white-space: nowrap; }
    .meta { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); font-size: 13px; }
    .stage { position: relative; min-height: 0; }
    canvas { width: 100%; height: 100%; display: block; background: #111318; }
    button {
      width: 44px;
      height: 40px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #242a34;
      color: var(--text);
      font-size: 18px;
      cursor: pointer;
    }
    button:disabled { color: #5c6676; cursor: default; }
    audio { width: min(520px, 55vw); height: 36px; }
    .status { color: var(--muted); font-size: 13px; min-width: 120px; }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="title">Stage 1 Preview</div>
      <div class="meta" id="meta">connecting</div>
    </header>
    <main class="stage">
      <canvas id="preview" width="900" height="1200"></canvas>
    </main>
    <footer>
      <button id="play" aria-label="Play or pause" title="Play or pause">▶</button>
      <audio id="audio" src="/audio" preload="auto"></audio>
      <div class="status" id="status">startup</div>
    </footer>
  </div>
  <script>
    const canvas = document.getElementById("preview");
    const ctx = canvas.getContext("2d");
    const audio = document.getElementById("audio");
    const play = document.getElementById("play");
    const meta = document.getElementById("meta");
    const statusEl = document.getElementById("status");
    const notes = [];
    let committedThrough = 0;
    let generatedThrough = 0;
    let ready = false;
    let buffering = true;
    let userWantsPlay = false;
    let targetBufferMs = 2000;
    let rebufferFloorMs = 750;
    let durationMs = 0;

    function resize() {
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    }
    window.addEventListener("resize", resize);
    resize();

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function syncBufferState() {
      const playhead = audio.currentTime * 1000;
      const future = committedThrough - playhead;
      if (!ready) {
        buffering = true;
        if (!audio.paused) audio.pause();
        setStatus("startup");
        return;
      }
      if (!audio.paused && future < rebufferFloorMs && generatedThrough < durationMs) {
        buffering = true;
        audio.pause();
        setStatus("buffering");
      } else if (buffering && future >= targetBufferMs) {
        buffering = false;
        setStatus("ready");
        if (userWantsPlay) audio.play().catch(() => {});
      } else if (!buffering) {
        setStatus("ready");
      }
    }

    play.addEventListener("click", () => {
      userWantsPlay = !userWantsPlay;
      play.textContent = userWantsPlay ? "Ⅱ" : "▶";
      if (userWantsPlay && ready && !buffering) audio.play().catch(() => {});
      if (!userWantsPlay) audio.pause();
    });

    audio.addEventListener("play", () => {
      if (!ready || buffering) {
        audio.pause();
        return;
      }
      userWantsPlay = true;
      play.textContent = "Ⅱ";
    });

    audio.addEventListener("pause", () => {
      if (!buffering) userWantsPlay = false;
      if (!userWantsPlay) play.textContent = "▶";
    });

    function addBatch(batch) {
      committedThrough = Math.max(committedThrough, batch.end_ms);
      generatedThrough = Math.max(generatedThrough, batch.generated_through_ms);
      for (const note of batch.notes) notes.push(note);
      notes.sort((a, b) => a.time_ms - b.time_ms || a.lane - b.lane);
      syncBufferState();
    }

    const stream = new EventSource("/events");
    stream.addEventListener("metadata", event => {
      const data = JSON.parse(event.data);
      durationMs = data.audio_duration_ms || 0;
      targetBufferMs = data.target_buffer_ms || targetBufferMs;
      rebufferFloorMs = data.rebuffer_floor_ms || rebufferFloorMs;
      meta.textContent = `${data.checkpoint_name} · ${Number(data.difficulty).toFixed(2)}★`;
    });
    stream.addEventListener("ready", () => {
      ready = true;
      buffering = false;
      play.disabled = false;
      syncBufferState();
      if (userWantsPlay) audio.play().catch(() => {});
    });
    stream.addEventListener("batch", event => addBatch(JSON.parse(event.data)));
    stream.addEventListener("status", event => {
      const data = JSON.parse(event.data);
      committedThrough = Math.max(committedThrough, data.committed_through_ms || 0);
      generatedThrough = Math.max(generatedThrough, data.generated_through_ms || 0);
      syncBufferState();
    });
    stream.addEventListener("done", event => {
      const data = JSON.parse(event.data);
      committedThrough = Math.max(committedThrough, data.committed_through_ms || committedThrough);
      generatedThrough = Math.max(generatedThrough, data.generated_through_ms || generatedThrough);
      buffering = false;
      setStatus("done");
      stream.close();
    });
    stream.addEventListener("error", event => {
      if (!event.data) return;
      setStatus("error");
      ready = false;
      buffering = true;
      userWantsPlay = false;
      play.textContent = "▶";
      play.disabled = true;
      audio.pause();
      audio.controls = false;
      stream.close();
    });

    function draw() {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      const playhead = audio.currentTime * 1000;
      const laneW = Math.min(110, w / 5);
      const boardW = laneW * 4;
      const left = (w - boardW) / 2;
      const receptorY = h - 110;
      const leadMs = 2600;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#111318";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#151a21";
      ctx.fillRect(left, 0, boardW, h);
      for (let lane = 0; lane < 4; lane++) {
        const x = left + lane * laneW;
        ctx.fillStyle = lane % 2 === 0 ? "#191f28" : "#161b23";
        ctx.fillRect(x, 0, laneW, h);
        ctx.strokeStyle = "#394150";
        ctx.strokeRect(x, 0, laneW, h);
      }
      ctx.fillStyle = "#2fb3a6";
      ctx.fillRect(left, receptorY, boardW, 4);
      for (const note of notes) {
        const start = note.time_ms;
        const end = note.end_time_ms || note.time_ms;
        if (end < playhead - 300 || start > playhead + leadMs) continue;
        const x = left + note.lane * laneW + 8;
        const y = receptorY - ((start - playhead) / leadMs) * (receptorY - 30);
        const noteW = laneW - 16;
        if (note.kind === "hold") {
          const y2 = receptorY - ((end - playhead) / leadMs) * (receptorY - 30);
          ctx.fillStyle = "rgba(88, 166, 255, 0.35)";
          ctx.fillRect(x, Math.min(y, y2), noteW, Math.max(10, Math.abs(y2 - y)));
          ctx.fillStyle = "#58a6ff";
          ctx.fillRect(x, y - 5, noteW, 10);
          ctx.fillRect(x, y2 - 5, noteW, 10);
        } else {
          ctx.fillStyle = "#f3d35c";
          ctx.fillRect(x, y - 7, noteW, 14);
        }
      }
      requestAnimationFrame(draw);
    }
    audio.addEventListener("timeupdate", syncBufferState);
    play.disabled = true;
    draw();
  </script>
</body>
</html>
"""
