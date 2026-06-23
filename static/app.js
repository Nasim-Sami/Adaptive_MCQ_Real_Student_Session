const startedAt = performance.now();

function formatDuration(totalSeconds) {
  const seconds = Math.max(0, Math.floor(Number(totalSeconds) || 0));
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

document.querySelectorAll("[data-episode-timer]").forEach((timer) => {
  const output = timer.querySelector("strong");
  const initialSeconds = Number(timer.dataset.initialSeconds || "0");
  const localStart = performance.now();

  const render = () => {
    const elapsed = initialSeconds + (performance.now() - localStart) / 1000;
    if (output) output.textContent = formatDuration(elapsed);
  };

  render();
  setInterval(render, 1000);
});

function createAudioContext() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return null;
  return new AudioContextClass();
}

function playNoiseBurst(ctx, startTime, duration, gainValue) {
  const sampleRate = ctx.sampleRate;
  const buffer = ctx.createBuffer(1, Math.max(1, Math.floor(sampleRate * duration)), sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i += 1) {
    data[i] = (Math.random() * 2 - 1) * (1 - i / data.length);
  }

  const source = ctx.createBufferSource();
  source.buffer = buffer;

  const filter = ctx.createBiquadFilter();
  filter.type = "bandpass";
  filter.frequency.value = 1400;
  filter.Q.value = 0.9;

  const gain = ctx.createGain();
  gain.gain.setValueAtTime(0.001, startTime);
  gain.gain.exponentialRampToValueAtTime(gainValue, startTime + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.001, startTime + duration);

  source.connect(filter);
  filter.connect(gain);
  gain.connect(ctx.destination);
  source.start(startTime);
  source.stop(startTime + duration);
}

function playClapping(ctx) {
  const now = ctx.currentTime + 0.05;
  for (let i = 0; i < 18; i += 1) {
    playNoiseBurst(ctx, now + i * 0.105 + Math.random() * 0.035, 0.055, 0.16);
  }
}

function playCheer(ctx) {
  const now = ctx.currentTime + 0.08;
  const notes = [523.25, 659.25, 783.99, 1046.5];
  notes.forEach((freq, index) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "triangle";
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.001, now + index * 0.16);
    gain.gain.exponentialRampToValueAtTime(0.08, now + index * 0.16 + 0.03);
    gain.gain.exponentialRampToValueAtTime(0.001, now + index * 0.16 + 0.28);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now + index * 0.16);
    osc.stop(now + index * 0.16 + 0.30);
  });
}

async function playCelebration(type) {
  const ctx = createAudioContext();
  if (!ctx) return;
  if (ctx.state === "suspended") {
    await ctx.resume();
  }
  playClapping(ctx);
  if (type === "clap_cheer") {
    playCheer(ctx);
  }
}

document.querySelectorAll("[data-celebration-button]").forEach((button) => {
  button.addEventListener("click", () => {
    playCelebration(button.dataset.celebrationButton || "clap");
  });
});

document.querySelectorAll("[data-celebration]").forEach((banner) => {
  const type = banner.dataset.celebration;
  if (!type) return;
  setTimeout(() => {
    playCelebration(type).catch(() => {});
  }, 450);
});

document.querySelectorAll("[data-timer-form]").forEach((form) => {
  form.addEventListener("submit", () => {
    const elapsedInput = form.querySelector("[data-elapsed-input]");
    if (elapsedInput) {
      elapsedInput.value = String(Math.round(performance.now() - startedAt));
    }

    form.querySelectorAll("button").forEach((button) => {
      button.disabled = true;
      button.textContent = button.type === "submit" ? "Submitting..." : button.textContent;
    });
  });
});

document.querySelectorAll("[data-option-card]").forEach((card) => {
  const input = card.querySelector("input[type='radio']");
  card.addEventListener("click", () => {
    document.querySelectorAll("[data-option-card]").forEach((item) => {
      item.classList.remove("selected");
    });
    card.classList.add("selected");
    if (input) {
      input.checked = true;
    }
  });
});

// "Why this topic?" XAI panel - fetched on demand, toggled open/closed.
document.querySelectorAll("[data-xai-button]").forEach((button) => {
  const container = document.querySelector("[data-xai-container]");
  if (!container) return;
  const defaultLabel = button.textContent;

  button.addEventListener("click", async () => {
    if (!container.hidden) {
      container.hidden = true;
      return;
    }
    if (container.dataset.loaded === "1") {
      container.hidden = false;
      return;
    }

    button.disabled = true;
    button.textContent = "Loading...";
    try {
      const response = await fetch(button.dataset.xaiUrl);
      container.innerHTML = await response.text();
      container.dataset.loaded = "1";
      container.hidden = false;
    } catch (err) {
      container.innerHTML = "<p class=\"empty-state\">XAI is unavailable right now.</p>";
      container.hidden = false;
    } finally {
      button.disabled = false;
      button.textContent = defaultLabel;
    }
  });
});
