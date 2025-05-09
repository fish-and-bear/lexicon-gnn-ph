import __buffer_polyfill from 'vite-plugin-node-polyfills/shims/buffer'
globalThis.Buffer = globalThis.Buffer || __buffer_polyfill
import __global_polyfill from 'vite-plugin-node-polyfills/shims/global'
globalThis.global = globalThis.global || __global_polyfill
import __process_polyfill from 'vite-plugin-node-polyfills/shims/process'
globalThis.process = globalThis.process || __process_polyfill

import {
  require_react
} from "./chunk-CJEDDXZD.js";
import {
  __toESM,
  require_dist,
  require_dist2,
  require_dist3
} from "./chunk-GJFZQ5ET.js";

// node_modules/@mui/material/Accordion/AccordionContext.js
var import_dist = __toESM(require_dist());
var import_dist2 = __toESM(require_dist2());
var import_dist3 = __toESM(require_dist3());
var React = __toESM(require_react());
var AccordionContext = React.createContext({});
if (process.env.NODE_ENV !== "production") {
  AccordionContext.displayName = "AccordionContext";
}
var AccordionContext_default = AccordionContext;

export {
  AccordionContext_default
};
//# sourceMappingURL=chunk-Q26JKWRY.js.map
