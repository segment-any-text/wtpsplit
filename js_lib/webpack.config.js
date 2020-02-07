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
        publicPath: "./dist/",
        path: path.resolve(__dirname, "example", "dist"),
        filename: "main.js"
    },
    devServer: {
        publicPath: "/dist/",
        contentBase: path.join(__dirname, 'example'),
    }
}