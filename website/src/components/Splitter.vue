<template>
  <v-container class="elevation-4 rounded-lg">
    <v-row>
      <v-col>
        <v-select
          height="3em"
          v-for="model in models"
          :key="model.code"
          v-model="selectedLevels[model.code]"
          :items="model.levels"
          v-show="selected == model.code"
          attach
          chips
          label="Levels"
          multiple
        >
          <template #selection="{ item }">
            <v-chip
              class="font-weight-bold"
              text-color="white"
              :color="colors[model.levels.indexOf(item)]"
              >{{ item }}</v-chip
            >
          </template></v-select
        >
      </v-col>
      <v-col cols="2">
        <v-select
          height="3em"
          :items="models"
          item-text="name"
          item-value="code"
          label="Model"
          v-model="selected"
        ></v-select>
      </v-col>
    </v-row>
    <v-row>
      <div id="editor" spellcheck="false"></div>
    </v-row>
  </v-container>
</template>

<script>
import Quill from "quill";
import * as nnsplit from "nnsplit";
import "@/style/quill.snow.reduced.css";

// see https://stackoverflow.com/a/2117523
function getGUID() {
  return ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, (c) =>
    (
      c ^
      (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (c / 4)))
    ).toString(16)
  );
}

// see https://stackoverflow.com/a/14480366
function getPosition(string, subString, index) {
  return string.split(subString, index).join(subString).length;
}

const Inline = Quill.import("blots/inline");

const splitBlots = [];

function makeBlotIfDoesntExist(name) {
  if (splitBlots.includes(name)) {
    return;
  }

  class SplitBlot extends Inline {
    static blotName = name;
    static tagName = "span";

    constructor(domNode, value) {
      super(domNode, value);

      if (typeof value === "object") {
        this.domNode.dataset.num = value.num;
        this.domNode.dataset.guid = value.guid;

        this.domNode.style.setProperty("--bottom", `${5 + 8 * value.num}px`);
        this.domNode.style.setProperty("--color", value.color);
        this.domNode.style.setProperty(
          "--leftcap",
          `url(assets/left_cap_${value.level + 1}.svg)`
        );
        this.domNode.style.setProperty(
          "--rightcap",
          `url(assets/right_cap_${value.level + 1}.svg)`
        );
      }
    }

    static create() {
      const node = super.create();
      node.classList.add("split");

      return node;
    }

    static formats(node) {
      return node.dataset.guid;
    }
  }
  Quill.register(SplitBlot);
  splitBlots.push(name);
}

export default {
  name: "Splitter",

  data() {
    let colors = ["#448AFF", "#76FF03", "#FF3D00", "#614793"];

    let models = [
      { code: "de", name: "German", samplePage: "Künstliches neuronales Netz" },
      { code: "en", name: "English", samplePage: "Artificial neural network" },
      { code: "no", name: "Norwegian", samplePage: "Kunstig nevralt nettverk" },
      { code: "sv", name: "Swedish", samplePage: "Artificiellt neuronnät" },
      { code: "tr", name: "Turkish", samplePage: "Yapay sinir ağları" },
      {
        code: "fr",
        name: "French",
        samplePage: "Réseau de neurones artificiels",
      },
      { code: "zh", name: "Simplified Chinese", samplePage: "人工神经网络" },
    ];

    let splits = {};
    let selectedLevels = {};

    models.forEach((model) => {
      model.levels = [];
      model.splitter = null;
      model.text = null;

      selectedLevels[model.code] = [];
      splits[model.code] = null;
    });

    return {
      selected: models[0].code,
      selectedLevels,
      models,
      colors,
      splits,
    };
  },
  async mounted() {
    this.quill = new Quill("#editor", {
      modules: {
        toolbar: false,
        history: {
          userOnly: true,
        },
      },
    });

    // see https://github.com/quilljs/quill/issues/110#issuecomment-461591218
    delete this.quill.getModule("keyboard").bindings["9"];

    // see https://github.com/quilljs/quill/issues/1184#issuecomment-384935594
    this.quill.clipboard.addMatcher(Node.ELEMENT_NODE, (node, delta) => {
      const ops = [];
      delta.ops.forEach((op) => {
        if (op.insert && typeof op.insert === "string") {
          ops.push({
            insert: op.insert,
          });
        }
      });
      delta.ops = ops;
      return delta;
    });

    this.quill.on("text-change", async (delta, oldDelta, user) => {
      if (user === "api") {
        return;
      }

      clearTimeout(this.splitTimeout);
      let model = this.models.find((x) => x.code == this.selected);
      model.text = this.getText();

      this.clear();
      this.splitTimeout = setTimeout(() => {
        this.split();
      }, 300);
    });
  },
  watch: {
    splits: {
      handler: function () {
        this.render();
      },
      deep: true,
    },
    selectedLevels: {
      handler: function () {
        setTimeout(() => {
          this.render();
        }, 0);
      },
      deep: true,
    },
    selected: {
      handler: async function () {
        let model = this.models.find((x) => x.code == this.selected);

        if (model.text !== null) {
          this.quill.setText(model.text, "user");
        }

        if (model.splitter === null) {
          model.splitter = await nnsplit.NNSplit.new(
            `models/${model.code}/model.onnx`
          );
          model.levels = model.splitter
            .getLevels()
            .map((x) => (x.startsWith("_") ? x.substring(1) : x));
          this.selectedLevels[
            this.selected
          ] = model.splitter.getLevels().filter((x) => !x.startsWith("_"));

          model.text = await this.fetchSampleText(model.code, model.samplePage);
          this.quill.setText(model.text, "user");
        }
      },
      immediate: true,
    },
  },

  methods: {
    async fetchSampleText(code, samplePage) {
      let response = await fetch(
        `https://${code}.wikipedia.org/w/api.php?` +
          new URLSearchParams({
            format: "json",
            action: "query",
            prop: "extracts",
            exintro: "1",
            explaintext: "1",
            titles: samplePage,
            origin: "*",
          })
      );
      let data = await response.json();
      let pageId = Object.keys(data.query.pages);
      let text = data.query.pages[pageId].extract;
      let punctLimit = 5;

      text = text.slice(0, getPosition(text, ".", punctLimit) + 1);
      text = text.slice(0, getPosition(text, "。 ", punctLimit) + 1);

      return text;
    },

    async split() {
      let model = this.models.find((x) => x.code == this.selected);

      let text = this.getText();
      let splits = await model.splitter.split([text]);
      splits = splits[0];

      this.splits[this.selected] = [splits, text];
    },
    getText() {
      return Array.from(this.quill.container.querySelectorAll("p"))
        .map((el) => el.textContent)
        .join("\n");
    },
    clear() {
      this.quill.removeFormat(0, Infinity);
      this.quill.formatText(0, Infinity, { bold: true });
    },
    render() {
      let model = this.models.find((x) => x.code == this.selected);

      if (model === null || this.splits[this.selected] === null) {
        return;
      }
      let [splits, text] = this.splits[this.selected];

      if (text != this.getText()) {
        // split is not valid anymore
        return;
      }

      this.clear();

      let levelMask = model.levels.map((x) =>
        this.selectedLevels[this.selected].includes(x)
      );

      const traverse = (parts, offset, level) => {
        parts.forEach((part) => {
          let length = part.text ? part.text.length : part.length;
          let remaining = levelMask.slice(level + 1).reduce((a, b) => a + b, 0);

          if (levelMask[level]) {
            let name = `split${remaining}`;
            makeBlotIfDoesntExist(name);

            let guid = getGUID();
            this.quill.formatText(offset, length, {
              [name]: {
                level: level,
                color: this.colors[level],
                num: remaining,
                guid,
              },
            });
          }

          if (typeof part !== "string") {
            traverse(part.parts, offset, level + 1);
          }
          offset += length;
        });
      };

      traverse(splits.parts, 0, 0);
    },
  },
};
</script>

<style lang="scss">
:root {
  --max_offset: 30px;
  --cap_width: 2;
  --underline-width: 1;
  --underline-intrinsic-width: 3;
  --underline-cap-width: 6px;
}

#editor span,
#editor strong {
  font-weight: normal;
  font-size: 1.3rem;
  line-height: calc(1.3rem + var(--max_offset));
  padding-bottom: var(--max_offset);
}

.split {
  --underline-width-scale: calc(
    var(--underline-width) / var(--underline-intrinsic-width)
  );

  background-image: linear-gradient(180deg, var(--color), var(--color)),
    var(--leftcap), var(--rightcap);
  background-position-x: calc(
      var(--underline-cap-width) * var(--underline-width-scale) + 2px
    ),
    0, 100%;
  background-position-y: calc(100% - var(--max_offset) + var(--bottom));
  background-size: calc(100% - 8px) calc(var(--underline-width) * 2px),
    auto calc(var(--underline-width) * 10px),
    auto calc(var(--underline-width) * 10px);
}
</style>

<style lang="scss" scoped >
#editor {
  font-size: 1rem;
  width: 100%;
  height: 30rem;
}
</style>