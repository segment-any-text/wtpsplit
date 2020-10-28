<template>
  <v-app>
    <h1 class="text-h2">NNSplit</h1>
    <v-main>
      <h2 class="text-h3 ml-10 mt-15 mb-5">Demo</h2>
      <v-container class="splitter-container elevation-4 rounded-lg">
        <Splitter />
      </v-container>

      <h2 class="text-h3 ml-10 mt-15 mb-5">Usage</h2>
      <Usage />
      <h2 class="text-h3 ml-10 mt-15 mb-5">Metrics</h2>
      <Metrics />
      <h2 class="text-h3 ml-10 mt-15 mb-5">Benchmarks</h2>
    </v-main>
  </v-app>
</template>

<script>
import Metrics from "./components/Metrics";
import Usage from "./components/Usage";
import "highlight.js/styles/github.css";

export default {
  name: "App",

  components: {
    Splitter: () => import("./components/Splitter"),
    Metrics,
    Usage,
  },

  data: () => ({
    //
  }),
};
</script>

<style lang="scss">
.splitter-container {
  height: 35rem;
}

.container {
  padding: 0 2em 2em 2em;
  max-width: 80rem;
}

.v-application {
  padding: 2em 4em;
}

.v-application code {
  all: unset;
}

h2 {
  font-family: "Roboto", sans-serif !important;
  font-weight: 500;
  letter-spacing: normal !important;
}

.v-data-table thead th {
  font-size: 1rem !important;
}

.v-data-table tbody td {
  font-size: 1rem !important;
}
</style>
