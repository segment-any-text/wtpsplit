import Worker from './model.worker.js';

const inputElement = document.querySelector("#to-split");
const outputElement = document.querySelector("#output");
const languageButtons = document.querySelectorAll("input[name=language]");

const worker = new Worker();
let timeout = null;
let language = Array.from(languageButtons).filter((x) => x.checked)[0].value;

function startInference() {
    worker.postMessage({ "text": inputElement.value, "language": language });
}

startInference();

inputElement.addEventListener("input", (e) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => {
        startInference();
    }, 150);
});

languageButtons.forEach((button) => {
    button.addEventListener("click", (e) => {
        language = button.value;
        startInference();
    });
})

worker.addEventListener("message", (e) => {
    outputElement.textContent = JSON.stringify(e.data, null, 2);
});