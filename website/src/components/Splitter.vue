<template>
  <div class="splitter-div d-flex flex-column">
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
    <div id="editor" spellcheck="false"></div>
  </div>
</template>

<script>
import Quill from "quill";
import * as nnsplit from "nnsplit";
import "@/style/quill.snow.reduced.css";

// see https://stackoverflow.com/a/14480366
function getPosition(string, subString, index) {
  return string.split(subString, index).join(subString).length;
}

const Inline = Quill.import("blots/inline");

const splitBlots = [];

let prevName = null;
let allBlots = [];

function makeBlotIfDoesntExist(name) {
  if (splitBlots.includes(name)) {
    return;
  }

  class SplitBlot extends Inline {
    static blotName = name;
    static tagName = "span";

    constructor(domNode, value) {
      super(domNode, value);

      allBlots = allBlots.filter((x) => document.body.contains(x));
      allBlots.push(domNode);

      if (typeof value === "object") {
        this.domNode.dataset.guid = value.guid;
        this.domNode.dataset.level = value.level;

        this.domNode.style.setProperty("--color", value.color);
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
  // see https://github.com/quilljs/quill/issues/2312#issuecomment-426221880
  Inline.order.splice(Inline.order.indexOf(prevName || "bold"), 0, name);
  splitBlots.push(name);

  prevName = name;
}

export default {
  name: "Splitter",

  data() {
    let colors = ["#81D4FA", "#EF9A9A", "#80CBC4", "#B39DDB"];

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
      this.quill.formatText(0, model.text.length, { bold: true });

      this.splitTimeout = setTimeout(() => {
        this.split();
      }, 500);
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
        this.styleLevels();
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
    styleLevels() {
      let model = this.models.find((x) => x.code == this.selected);
      if (model === null) {
        return;
      }

      let levelMask = model.levels.map((x) =>
        this.selectedLevels[this.selected].includes(x)
      );

      allBlots.forEach((x) => {
        let level = parseInt(x.dataset.level);
        let num = levelMask.slice(level + 1).reduce((a, b) => {
          return a + b;
        }, 0);
        x.style.setProperty("--bottom", `${2 + 8 * num}`);

        if (levelMask[level]) {
          x.classList.remove("invisible");
        } else {
          x.classList.add("invisible");
        }
      });
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

      let clearData = { bold: false };
      splitBlots.forEach((x) => {
        clearData[x] = false;
      });
      this.quill.formatText(0, text.length, clearData);
      let whitespaceRegex = /^\s+$/;

      const traverse = (parts, offset, level) => {
        parts.forEach((part) => {
          let length = part.text ? part.text.length : part.length;

          let name = `split${level}`;
          makeBlotIfDoesntExist(name);

          // only whitespace is not interesting
          if (!whitespaceRegex.test(text.slice(offset, offset + length))) {
            this.quill.formatText(offset, length, {
              [name]: {
                level,
                color: this.colors[level],
                guid: `${level}_${offset}_${length}`,
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
      this.styleLevels();
    },
  },
};
</script>

<style lang="scss">
:root {
  --max_offset: 30;
  --cap_width: 2;
  --underline-width: 1;
  --underline-intrinsic-width: 3;
  --underline-cap-width: 6px;
}

#editor .ql-editor {
  padding: 0;
  max-height: 100%;
}

#editor span,
#editor strong {
  font-weight: normal;
  font-size: 1rem;
  line-height: calc(1rem + var(--max_offset) * 2px);
  padding-bottom: calc(var(--max_offset) * 1px);
  padding-top: calc(var(--max_offset) * 1px);
}

strong .split {
  all: unset;
}

.split {
  border: 2px solid var(--color);
  z-index: calc(var(--bottom) * -1);
  padding: calc(var(--bottom) * 1px) 0 !important;
  box-shadow: 3px 3px 5px 1px rgba(51, 51, 51, 0.2);
  margin: calc((var(--max_offset) - var(--bottom)) * 1px) 0;

  &.invisible {
    all: unset;
  }
}

.ql-editor p {
  padding-left: 1px !important;
}
</style>

<style lang="scss" scoped >
.splitter-div {
  height: 100%;
}

#editor {
  font-size: 1rem;
  width: 100%;
  height: calc(100% - 6em);
}
</style>