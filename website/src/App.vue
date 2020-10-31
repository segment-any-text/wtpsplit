<template>
  <v-app>
    <v-main class="pa-4 px-md-12">
      <h1 class="text-h2">NNSplit</h1>
      <iframe
        class="mt-1 mb-4"
        src="https://ghbtns.com/github-btn.html?user=bminixhofer&repo=nnsplit&type=star&count=true&size=large"
        frameborder="0"
        scrolling="0"
        width="170"
        height="30"
        title="GitHub"
      ></iframe>
      <p class="body-1">
        NNSplit is a tool to split text using a neural network. The main
        application is <strong>sentence boundary detection</strong>, but e. g.
        <strong>compound splitting for German</strong> is also supported.
        NNSplit's focus is on being <strong>fast</strong> and
        <strong>not reliant on correct punctuation, spelling and case</strong>.
      </p>
      <p>
        NNSplit uses a byte-level LSTM so the models are small (&lt; 4MB) and
        models can be trained for every unicode encodable language.
      </p>
      <p>
        NNSplit is written in Rust, with bindings for Python and Javascript
        (Browser and Node.js) and currently has models for 7 different
        languages.
      </p>
      <h2 id="demo" class="text-h3 mt-15 mb-5">Demo</h2>
      <p class="body-1">
        Try NNSplit! This demo runs powered by
        <a href="https://github.com/bminixhofer/tractjs">tractjs</a> in your
        browser so no internet connection is required once a model is loaded.
      </p>

      <v-container class="elevation-2 rounded-lg">
        <Splitter />
      </v-container>

      <h2 id="usage" class="text-h3 mt-15 mb-5">Usage</h2>
      <p class="body-1">
        NNSplit has a simple API. Calling <code>.split(...)</code> returns
        <code>Split</code> objects. Iterating over a <code>Split</code> yields
        lower-level <code>Split</code> objects until the lowest level, where a
        string is returned. Some useful links:
      </p>
      <ul class="mb-5 body-1">
        <li>
          <a href="https://github.com/bminixhofer/nnsplit/"
            >NNSplit on Github</a
          >
        </li>
        <li>
          <a href="https://www.npmjs.com/package/nnsplit"
            >NNSplit NPM package</a
          >
        </li>
        <li>
          <a href="https://pypi.org/project/nnsplit/">NNSplit PyPI package</a>
        </li>
        <li>
          <a href="https://crates.io/crates/nnsplit"
            >NNSplit crates.io package</a
          >
        </li>
        <li>
          <a href="https://docs.rs/nnsplit/"
            >NNSplit for Rust documentation on docs.rs</a
          >
        </li>
      </ul>
      <Usage />
      <h2 id="metrics" class="text-h3 mt-15 mb-5">Metrics</h2>
      <p class="body-1">
        Sentence boundary detection of NNSplit models was evaluated on the OPUS
        Open Subtitles dataset by concatenating 2 - 4 sentences and measuring
        the number of concatenations which are split completely correctly vs.
        the total number of concatenations in percent.
      </p>
      <Metrics />
      <h2 id="benchmark" class="text-h3 mt-15 mb-5">Benchmark</h2>
      <p class="body-1">
        This benchmark measures the time it takes to split 1000 texts from the
        German Wikipedia. The German NNSplit model is compared with the
        <code>de_core_news_sm</code> Spacy model and the German Spacy
        sentencizer. NNSplit Python bindings were used to make it as fair as
        possible and times averaged over 10 runs. The model architecture of
        NNSplit is the same for all languages. Some details:
      </p>
      <pre class="caption" style="font-family: monospace !important">
    Python version: 3.8.5
    NNSplit version: 0.5.2
    Spacy version: 2.3.2
    GPU: RTX 2080 TI
    CPU: GeForce RTX 2080 Ti
    CUDA/cuDNN version: 10.2.89 / 7.6.5
      </pre>
      <Benchmark />
    </v-main>
  </v-app>
</template>

<script>
import Metrics from "./components/Metrics";
import Usage from "./components/Usage";
import Benchmark from "./components/Benchmark";
import "highlight.js/styles/github.css";

export default {
  name: "App",

  components: {
    Splitter: () => import("./components/Splitter"),
    Metrics,
    Usage,
    Benchmark,
  },

  data: () => ({
    //
  }),
};
</script>

<style lang="scss">
.container {
  max-width: 80rem;
}

.container code {
  all: unset;
}

h2 {
  font-family: "Roboto", sans-serif !important;
  font-weight: 500;
  letter-spacing: normal !important;
}

pre {
  white-space: pre-wrap;
}

.v-data-table thead th {
  font-size: 1rem !important;
}

.v-data-table tbody td {
  font-size: 1rem !important;
}
</style>
