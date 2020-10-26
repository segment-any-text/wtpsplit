<template>
  <div id="editor" spellcheck="false"></div>
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

        this.domNode.style.setProperty("--bottom", `-${3 + 6 * value.num}px`);
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

  async mounted() {
    this.splitter = await nnsplit.NNSplit.new("models/de/model.onnx");
    this.levels = this.splitter.getLevels();
    this.levelMask = this.levels.map((x) => !x.startsWith("_"));

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
      this.quill.removeFormat(0, Infinity);
      this.splitTimeout = setTimeout(() => {
        this.split();
      }, 300);
    });
  },

  methods: {
    async split() {
      let splits = await this.splitter.split([this.getText()]);
      splits = splits[0];

      const traverse = (parts, offset, level) => {
        parts.forEach((part) => {
          let length = part.text ? part.text.length : part.length;
          let remaining = this.levelMask
            .slice(level + 1)
            .reduce((a, b) => a + b, 0);

          if (this.levelMask[level]) {
            let name = `split${remaining}`;
            makeBlotIfDoesntExist(name);

            this.quill.formatText(offset, length, {
              [name]: { num: remaining, guid: getGUID() },
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

    getText() {
      return Array.from(this.quill.container.querySelectorAll("p"))
        .map((el) => el.textContent)
        .join("\n");
    },
  },
};
</script>

<style lang="scss">
.split {
  position: relative;
}

.split::after {
  content: "";
  height: 6px;
  left: 0;
  bottom: var(--bottom);
  width: 100%;
  position: absolute;
  border: 1px solid black;
  border-top: none;
  border-bottom-left-radius: 3px;
  border-bottom-right-radius: 3px;
}
</style>

<style lang="scss" scoped >
#editor {
  height: 100%;
  font-size: 1rem;
  width: 100%;
}
</style>