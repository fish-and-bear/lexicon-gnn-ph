import __buffer_polyfill from 'vite-plugin-node-polyfills/shims/buffer'
globalThis.Buffer = globalThis.Buffer || __buffer_polyfill
import __global_polyfill from 'vite-plugin-node-polyfills/shims/global'
globalThis.global = globalThis.global || __global_polyfill
import __process_polyfill from 'vite-plugin-node-polyfills/shims/process'
globalThis.process = globalThis.process || __process_polyfill

import {
  createChainedFunction,
  deprecatedPropType,
  setRef
} from "./chunk-O4QBWVDY.js";
import {
  ClassNameGenerator_default
} from "./chunk-ALJKGM6N.js";
import {
  __toESM,
  require_dist,
  require_dist2,
  require_dist3
} from "./chunk-GJFZQ5ET.js";

// node_modules/@mui/material/utils/createChainedFunction.js
var import_dist = __toESM(require_dist());
var import_dist2 = __toESM(require_dist2());
var import_dist3 = __toESM(require_dist3());
var createChainedFunction_default = createChainedFunction;

// node_modules/@mui/material/utils/deprecatedPropType.js
var import_dist4 = __toESM(require_dist());
var import_dist5 = __toESM(require_dist2());
var import_dist6 = __toESM(require_dist3());
var deprecatedPropType_default = deprecatedPropType;

// node_modules/@mui/material/utils/setRef.js
var import_dist7 = __toESM(require_dist());
var import_dist8 = __toESM(require_dist2());
var import_dist9 = __toESM(require_dist3());
var setRef_default = setRef;

// node_modules/@mui/material/utils/index.js
var import_dist10 = __toESM(require_dist());
var import_dist11 = __toESM(require_dist2());
var import_dist12 = __toESM(require_dist3());
var unstable_ClassNameGenerator = {
  configure: (generator) => {
    if (process.env.NODE_ENV !== "production") {
      console.warn(["MUI: `ClassNameGenerator` import from `@mui/material/utils` is outdated and might cause unexpected issues.", "", "You should use `import { unstable_ClassNameGenerator } from '@mui/material/className'` instead", "", "The detail of the issue: https://github.com/mui/material-ui/issues/30011#issuecomment-1024993401", "", "The updated documentation: https://mui.com/guides/classname-generator/"].join("\n"));
    }
    ClassNameGenerator_default.configure(generator);
  }
};

export {
  createChainedFunction_default,
  deprecatedPropType_default,
  setRef_default,
  unstable_ClassNameGenerator
};
//# sourceMappingURL=chunk-PZVDDBEV.js.map
