module.exports = {
  "transpileDependencies": [
    "vuetify"
  ],
  // see https://github.com/vuejs/vue-cli/issues/2948#issuecomment-438589725
  publicPath: process.env.NODE_ENV === 'production' ? '/nnsplit/' : '/',
  chainWebpack: config => {
    config.resolve.symlinks(false);
    config.module.rule('raw')
      .test(/\.md$/)
      .use('raw-loader')
      .loader('raw-loader')
      .end();
  }
}