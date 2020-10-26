<template>
  <v-app>
    <v-main>
      <Splitter />
    </v-main>
  </v-app>
</template>

<script>
export default {
  name: "App",

  components: {
    Splitter: () => import("./components/Splitter"),
  },

  data: () => ({
    //
  }),
};
</script>
