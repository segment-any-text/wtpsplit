<template>
  <v-container class="elevation-4 rounded-lg">
    <v-row>
      <v-col cols="10">
        <markdown-it-vue :content="usage[selected]['code']" />
      </v-col>
      <v-col cols="2">
        <v-select
          height="3em"
          :items="Object.keys(usage)"
          label="Programming Language"
          v-model="selected"
        >
          <template slot="item" slot-scope="data">
            <img class="language-icon" :src="usage[data.item].icon" />
            {{ data.item }} </template
          ><template slot="selection" slot-scope="data">
            <img class="language-icon" :src="usage[data.item].icon" />
            {{ data.item }}
          </template></v-select
        >
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import pythonUsage from "@/assets/python_usage.md";
import rustUsage from "@/assets/rust_usage.md";
import jsUsage from "@/assets/js_usage.md";

export default {
  name: "Usage",
  data() {
    return {
      selected: "Python",
      usage: {
        Python: {
          icon: "/assets/python.svg",
          code: pythonUsage,
        },
        Rust: {
          icon: "/assets/rust.svg",
          code: rustUsage,
        },
        JavaScript: {
          icon: "/assets/javascript.svg",
          code: jsUsage,
        },
      },
    };
  },
};
</script>

<style lang="scss">
.language-icon {
  width: 1.4em;
  margin-right: 0.3em;
}
</style>

