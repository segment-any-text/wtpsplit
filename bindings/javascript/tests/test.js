const playwright = require('playwright');

(async () => {
    for (const browserType of ['chromium', 'firefox', 'webkit']) {
        const browser = await playwright[browserType].launch();
        const context = await browser.newContext();
        const page = await context.newPage();

        await page.goto('http://0.0.0.0:8080');

        page.on('console', msg => {
            for (let i = 0; i < msg.args().length; i++) {
                console.log(`${i}: ${msg.args()[i]}`);
            }
        });

        await page.waitForEvent("console", (msg) => {
            if (msg.type() == "error") {
                process.exit(1);
            }

            return msg.text() == "playwright:success";
        });
        await browser.close();
    }
})();