let fail_spy;
let success_spy;

Cypress.on('window:before:load', (win) => {
    fail_spy = cy.spy(win.console, "error");
    success_spy = cy.spy(win.console, "log").withArgs("cypress:success");
})

it('runs the javascript without errors and logs "cypress:success"', () => {
    cy.visit("/");
    // TODO: add early exit
    cy.wait(3000).then(() => {
        expect(fail_spy).not.to.be.called;
        expect(success_spy).to.be.called;
    });
});
