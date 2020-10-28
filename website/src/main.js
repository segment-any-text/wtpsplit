import Vue from 'vue'
import MarkdownItVue from 'markdown-it-vue'

import App from './App.vue'
import vuetify from './plugins/vuetify'

Vue.use(MarkdownItVue)
Vue.config.productionTip = false

new Vue({
  vuetify,
  render: h => h(App)
}).$mount('#app')
