import("./index.js")
    .catch(e => console.error("Error importing `index.js`:", e))
    .then((module) => module.benchmark());
