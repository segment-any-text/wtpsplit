<template>
  <v-container>
    <BenchmarkChart :chartdata="data" :options="options" :styles="styles" />
  </v-container>
</template>

<script>
import BenchmarkChart from "./BenchmarkChart";
import ChartJS from "chart.js";

ChartJS.defaults.global.defaultFontSize = 16;

export default {
  name: "Benchmark",
  data() {
    const batchSizes = [1, 4, 16, 64, 256, 1024];
    const scores = {
      "NNSplit (CPU)": [
        32.329824566841125,
        19.54384229183197,
        10.234382343292236,
        8.453002309799194,
        8.057734060287476,
        8.148843455314637,
      ],
      "NNSplit (GPU)": [
        20.889695167541504,
        5.528475999832153,
        1.7125168085098266,
        0.8738419771194458,
        0.6267955780029297,
        0.6094059944152832,
      ],
      "Spacy (Sentencizer)": [
        1.312553095817566,
        1.290270161628723,
        1.2882855892181397,
        1.286806845664978,
        1.2958542585372925,
        1.3005393505096436,
      ],
      "Spacy (Tagger) (CPU)": [
        6.851607370376587,
        5.483753633499146,
        4.571131563186645,
        3.8925987243652345,
        4.013920450210572,
        4.4459668636322025,
      ],
      "Spacy (Tagger) (GPU)": [
        7.207011651992798,
        4.695266628265381,
        3.4098012447357178,
        2.3435105323791503,
        2.0651674032211305,
        2.032839560508728,
      ],
    };
    const colors = [
      "#81D4FA",
      "#81D4FA",
      "rgba(0, 0, 0, 0.7)",
      "rgba(0, 0, 0, 0.7)",
      "rgba(0, 0, 0, 0.7)",
    ];
    const dashes = [[5, 5], [], [25, 5], [5, 5], []];

    const datasets = Object.entries(scores).map(([name, series], i) => {
      let dataset = {};
      dataset.data = series;
      dataset.label = name;
      dataset.fill = false;
      dataset.borderColor = colors[i];
      dataset.borderDash = dashes[i];

      return dataset;
    });

    return {
      data: {
        labels: batchSizes,
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          xAxes: [
            {
              scaleLabel: {
                display: true,
                labelString: "Batch Size",
              },
            },
          ],
          yAxes: [
            {
              scaleLabel: {
                display: true,
                labelString: "Time (seconds)",
              },
            },
          ],
        },
        tooltips: {
          mode: "index",
          callbacks: {
            label: (tooltipItem, data) => {
              var label = data.datasets[tooltipItem.datasetIndex].label || "";

              if (label) {
                label += ": ";
              }
              label += Math.round(tooltipItem.yLabel * 100) / 100;
              return label + "s";
            },
          },
        },
      },
      styles: {
        height: `100%`,
        position: "relative",
      },
    };
  },
  components: {
    BenchmarkChart,
  },
};
</script>

<style lang="scss" scoped>
.container {
  height: 40rem;
  padding: 0;
}
</style>