<template>
  <v-container class="elevation-4 rounded-lg">
    <v-row>
      <v-spacer />
      <v-col md="2" class="pb-0">
        <v-select
          height="3em"
          :items="Object.keys(scores)"
          label="Model"
          v-model="selected"
        ></v-select>
      </v-col>
    </v-row>
    <v-simple-table
      v-for="lang in Object.keys(scores)"
      v-show="selected == lang"
      :key="lang"
    >
      <template v-slot:default>
        <thead>
          <tr>
            <th></th>
            <th
              v-for="model in Object.keys(scores[lang])"
              class="text-left"
              :key="model"
            >
              {{ model }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="key in Object.keys(
              scores[lang][Object.keys(scores[lang])[0]]
            )"
            :key="key"
          >
            <td>{{ key }}</td>
            <td
              v-for="model in Object.keys(scores[lang])"
              :key="model"
              :class="
                scores[lang][model][key] >=
                Math.max(
                  ...Object.keys(scores[lang])
                    .map((x) => scores[lang][x][key])
                    .filter((x) => !Number.isNaN(x))
                )
                  ? 'font-weight-bold'
                  : ''
              "
            >
              {{
                Number.isNaN(scores[lang][model][key])
                  ? "-"
                  : Math.round(scores[lang][model][key] * 100 * 100) / 100
              }}
            </td>
          </tr>
        </tbody>
      </template>
    </v-simple-table>
  </v-container>
</template>

<script>
export default {
  name: "Metrics",
  data() {
    return {
      selected: "German",
      scores: {
        German: {
          NNSplit: {
            Clean: 0.807723,
            "Partial punctuation": 0.516424,
            "Partial case": 0.822369,
            "Partial punctuation and case": 0.499656,
            "No punctuation and case": 0.256299,
          },
          "Spacy (Tagger)": {
            Clean: 0.833368,
            "Partial punctuation": 0.426458,
            "Partial case": 0.792839,
            "Partial punctuation and case": 0.377201,
            "No punctuation and case": 0.0952267,
          },
          "Spacy (Sentencizer)": {
            Clean: 0.878471,
            "Partial punctuation": 0.266312,
            "Partial case": 0.876678,
            "Partial punctuation and case": 0.26697,
            "No punctuation and case": 0.00756195,
          },
        },
        English: {
          NNSplit: {
            Clean: 0.763458,
            "Partial punctuation": 0.486206,
            "Partial case": 0.768897,
            "Partial punctuation and case": 0.438204,
            "No punctuation and case": 0.141555,
          },
          "Spacy (Tagger)": {
            Clean: 0.859013,
            "Partial punctuation": 0.486595,
            "Partial case": 0.831067,
            "Partial punctuation and case": 0.4339,
            "No punctuation and case": 0.151777,
          },
          "Spacy (Sentencizer)": {
            Clean: 0.820934,
            "Partial punctuation": 0.249753,
            "Partial case": 0.819679,
            "Partial punctuation and case": 0.249873,
            "No punctuation and case": 0.00463281,
          },
        },
        Norwegian: {
          NNSplit: {
            Clean: 0.850256,
            "Partial punctuation": 0.623128,
            "Partial case": 0.847655,
            "Partial punctuation and case": 0.526556,
            "No punctuation and case": 0.195445,
          },
          "Spacy (Tagger)": {
            Clean: 0.93792,
            "Partial punctuation": 0.442299,
            "Partial case": 0.910273,
            "Partial punctuation and case": 0.377141,
            "No punctuation and case": 0.060107,
          },
          "Spacy (Sentencizer)": {
            Clean: 0.878859,
            "Partial punctuation": 0.263921,
            "Partial case": 0.877395,
            "Partial punctuation and case": 0.26413,
            "No punctuation and case": 0.00472248,
          },
        },
        Swedish: {
          NNSplit: {
            Clean: 0.831306,
            "Partial punctuation": 0.587172,
            "Partial case": 0.836716,
            "Partial punctuation and case": 0.51484,
            "No punctuation and case": 0.206952,
          },
          "Spacy (Tagger)": {
            Clean: NaN,
            "Partial punctuation": NaN,
            "Partial case": NaN,
            "Partial punctuation and case": NaN,
            "No punctuation and case": NaN,
          },
          "Spacy (Sentencizer)": {
            Clean: 0.873121,
            "Partial punctuation": 0.262038,
            "Partial case": 0.87339,
            "Partial punctuation and case": 0.262217,
            "No punctuation and case": 0.00352692,
          },
        },
        Turkish: {
          NNSplit: {
            Clean: 0.8733,
            "Partial punctuation": 0.632185,
            "Partial case": 0.877694,
            "Partial punctuation and case": 0.573482,
            "No punctuation and case": 0.243955,
          },
          "Spacy (Tagger)": {
            Clean: NaN,
            "Partial punctuation": NaN,
            "Partial case": NaN,
            "Partial punctuation and case": NaN,
            "No punctuation and case": NaN,
          },
          "Spacy (Sentencizer)": {
            Clean: 0.918164,
            "Partial punctuation": 0.274083,
            "Partial case": 0.917446,
            "Partial punctuation and case": 0.274352,
            "No punctuation and case": 0.00364647,
          },
        },
        French: {
          NNSplit: {
            Clean: 0.885584,
            "Partial punctuation": 0.66587,
            "Partial case": 0.887438,
            "Partial punctuation and case": 0.580686,
            "No punctuation and case": 0.251696,
          },
          "Spacy (Tagger)": {
            Clean: 0.903697,
            "Partial punctuation": 0.382312,
            "Partial case": 0.876797,
            "Partial punctuation and case": 0.34492,
            "No punctuation and case": 0.0473742,
          },
          "Spacy (Sentencizer)": {
            Clean: 0.896942,
            "Partial punctuation": 0.267478,
            "Partial case": 0.897211,
            "Partial punctuation and case": 0.267926,
            "No punctuation and case": 0.00298891,
          },
        },
        "Simplified Chinese": {
          NNSplit: {
            Clean: 0.328601,
            "Partial punctuation": 0.238784,
            "Partial case": 0.330125,
            "Partial punctuation and case": 0.240488,
            "No punctuation and case": 0.160774,
          },
          "Spacy (Tagger)": {
            Clean: 0.236004,
            "Partial punctuation": 0.153929,
            "Partial case": 0.235706,
            "Partial punctuation and case": 0.154108,
            "No punctuation and case": 0.109514,
          },
          "Spacy (Sentencizer)": {
            Clean: 0.186478,
            "Partial punctuation": 0.067579,
            "Partial case": 0.186568,
            "Partial punctuation and case": 0.067639,
            "No punctuation and case": 0.00275,
          },
        },
        Russian: {
          NNSplit: {
            Clean: 0.833488,
            "Partial punctuation": 0.523926,
            "Partial case": 0.831545,
            "Partial punctuation and case": 0.432047,
            "No punctuation and case": 0.100696,
          },
          "Spacy (Tagger)": {
            Clean: 0.903279,
            "Partial punctuation": 0.335027,
            "Partial case": 0.803927,
            "Partial punctuation and case": 0.316974,
            "No punctuation and case": 0.186239,
          },
          "Spacy (Sentencizer)": {
            Clean: 0.858714,
            "Partial punctuation": 0.260454,
            "Partial case": 0.858894,
            "Partial punctuation and case": 0.260842,
            "No punctuation and case": 0.005081,
          },
        },
        Ukrainian: {
          NNSplit: {
            Clean: 0.700152,
            "Partial punctuation": 0.408763,
            "Partial case": 0.6928,
            "Partial punctuation and case": 0.368264,
            "No punctuation and case": 0.111576,
          },
          "Spacy (Tagger)": {
            Clean: NaN,
            "Partial punctuation": NaN,
            "Partial case": NaN,
            "Partial punctuation and case": NaN,
            "No punctuation and case": NaN,
          },
          "Spacy (Sentencizer)": {
            Clean: 0.825657,
            "Partial punctuation": 0.25615,
            "Partial case": 0.826225,
            "Partial punctuation and case": 0.256329,
            "No punctuation and case": 0.00807,
          },
        },
      },
    };
  },
};
</script>

<style>
</style>