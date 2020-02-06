import Worker from './model.worker.js';

const inputElement = document.querySelector("#to-split");
const outputElement = document.querySelector("#output");

const worker = new Worker();
let timeout = null;

inputElement.addEventListener("input", (e) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => {
        worker.postMessage({ "text": inputElement.value });
    }, 100);
});

worker.addEventListener("message", (e) => {
    outputElement.textContent = JSON.stringify(e.data, null, "\t");
});