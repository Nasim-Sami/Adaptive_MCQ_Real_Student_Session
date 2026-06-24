const startedAt = performance.now();

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
