<template>
  <v-app>
    <h1 class="text-h2">NNSplit</h1>
    <v-main>
      <h2 class="text-h3">Demo</h2>
      <Splitter />
      <h2 class="text-h3">Usage</h2>
      <h2 class="text-h3">Metrics</h2>
      <Metrics />
      <h2 class="text-h3">Benchmarks</h2>
    </v-main>
  </v-app>
</template>

<script>
import Metrics from "./components/Metrics";

export default {
  name: "App",

  components: {
    Splitter: () => import("./components/Splitter"),
    Metrics,
  },

  data: () => ({
    //
  }),
};
</script>

<style lang="scss">
.container {
  padding: 0 2em 2em 2em;
  max-width: 80rem;
}

.v-application {
  padding: 2em 4em;
}
</style>