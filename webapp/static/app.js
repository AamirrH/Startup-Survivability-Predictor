const form = document.querySelector("#startup-form");
const button = document.querySelector("#predict-button");
const errors = document.querySelector("#form-errors");
const emptyResult = document.querySelector("#empty-result");
const predictionResult = document.querySelector("#prediction-result");
const warnings = document.querySelector("#prediction-warnings");
let requestNumber = 0;
let controller;

function clearErrors() {
  errors.hidden = true;
  errors.replaceChildren();
  form.querySelectorAll(".field-error").forEach(element => element.textContent = "");
  form.querySelectorAll("[aria-invalid]").forEach(element => element.removeAttribute("aria-invalid"));
}

function showErrors(messages) {
  errors.hidden = false;
  const heading = document.createElement("strong");
  heading.textContent = "Check these details";
  const list = document.createElement("ul");
  let firstField;
  for (const [field, message] of Object.entries(messages)) {
    const item = document.createElement("li");
    item.textContent = message;
    list.append(item);
    const input = form.elements.namedItem(field);
    if (input) {
      input.setAttribute("aria-invalid", "true");
      document.getElementById(`${field}-error`).textContent = message;
      firstField ||= input;
    }
  }
  errors.replaceChildren(heading, list);
  if (firstField) firstField.focus();
  else errors.scrollIntoView({block: "center"});
}

function resetResult() {
  requestNumber += 1;
  controller?.abort();
  emptyResult.hidden = false;
  predictionResult.hidden = true;
  warnings.hidden = true;
  warnings.replaceChildren();
  button.disabled = false;
  button.innerHTML = 'Check startup outlook <span aria-hidden="true">→</span>';
  form.removeAttribute("aria-busy");
}

function renderResult(data) {
  document.querySelector("#result-company").textContent = data.startup_name;
  document.querySelector("#outcome").textContent = data.outcome;
  document.querySelector("#outcome-description").textContent = data.prediction === 1
    ? "This profile more closely resembles companies that were acquired."
    : "This profile more closely resembles companies that closed.";
  for (const outcome of ["acquired", "closed"]) {
    const value = data[`${outcome}_score`] * 100;
    document.querySelector(`#${outcome}-value`).textContent = `${value.toFixed(1)}%`;
    document.querySelector(`#${outcome}-bar`).value = value;
  }
  document.querySelector("#score-context").textContent = Math.abs(data.acquired_score - data.closed_score) < 0.1
    ? "The two scores are close. The model has no strong preference."
    : `${data.known_features} of 7 model inputs supplied. Unknown values use training-based defaults.`;
  warnings.replaceChildren();
  warnings.hidden = data.warnings.length === 0;
  for (const warning of data.warnings) {
    const paragraph = document.createElement("p");
    paragraph.textContent = warning;
    warnings.append(paragraph);
  }
  emptyResult.hidden = true;
  predictionResult.hidden = false;
  document.querySelector("#outcome").focus({preventScroll: true});
  if (window.matchMedia("(max-width: 760px)").matches) {
    predictionResult.scrollIntoView({block: "start", behavior: "auto"});
  }
}

form.querySelectorAll("input, select").forEach(input => {
  const error = document.getElementById(`${input.name}-error`);
  if (error) input.setAttribute("aria-describedby", error.id);
});

form.addEventListener("submit", async event => {
  event.preventDefault();
  clearErrors();
  resetResult();
  const currentRequest = requestNumber;
  controller = new AbortController();
  const requestController = controller;
  button.disabled = true;
  button.textContent = "Checking outlook…";
  form.setAttribute("aria-busy", "true");
  const timeout = window.setTimeout(() => requestController.abort(), 15000);
  try {
    const response = await fetch(form.action, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(Object.fromEntries(new FormData(form))),
      signal: requestController.signal
    });
    const data = await response.json();
    if (currentRequest !== requestNumber) return;
    if (!response.ok) {
      showErrors(data.errors || {form: "The prediction could not be completed. Please try again."});
      return;
    }
    renderResult(data);
  } catch (error) {
    if (currentRequest === requestNumber) {
      showErrors({form: "Could not reach the model. Check your connection and try again."});
    }
  } finally {
    window.clearTimeout(timeout);
    if (currentRequest === requestNumber) {
      button.disabled = false;
      button.innerHTML = 'Check startup outlook <span aria-hidden="true">→</span>';
      form.removeAttribute("aria-busy");
    }
  }
});

form.addEventListener("input", () => { clearErrors(); resetResult(); });
form.addEventListener("reset", () => { clearErrors(); resetResult(); });
const examples = {
  northstar: {
    startup_name: "Northstar Labs (example)", country_code: "USA", first_category: "Software",
    funding_total_usd: "5000000", funding_rounds: "3", founded_at_year: "2008",
    first_funding_at_year: "2009", last_funding_at_year: "2014"
  },
  harbor: {
    startup_name: "Harbor Cart (example)", country_code: "CAN", first_category: "E-Commerce",
    funding_total_usd: "250000", funding_rounds: "1", founded_at_year: "2013",
    first_funding_at_year: "2014", last_funding_at_year: "2014"
  },
  seedling: {
    startup_name: "Seedling Health (example)", country_code: "IND", first_category: "Health Care",
    funding_total_usd: "1500000", funding_rounds: "2", founded_at_year: "2011",
    first_funding_at_year: "2012", last_funding_at_year: "2013"
  }
};

document.querySelectorAll(".example-option").forEach(option => {
  option.addEventListener("click", () => {
    form.reset();
    const example = examples[option.dataset.example];
    if (example) {
      for (const [field, value] of Object.entries(example)) form.elements.namedItem(field).value = value;
    }
    option.closest("details").open = false;
    clearErrors();
    resetResult();
    form.elements.namedItem("startup_name").focus();
  });
});
