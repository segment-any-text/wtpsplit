const path = require("path");

module.exports = {
    entry: "./example/index.js",
    mode: "development",
    module: {
        rules: [
            {
                test: /\.worker\.js$/,
                use: { loader: "worker-loader" }
            }
        ]
    },
    output: {
        path: path.resolve(__dirname, "example"),
        filename: "main.js"
    },
    devServer: {
        contentBase: path.join(__dirname, 'example'),
    }
}