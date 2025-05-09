import __buffer_polyfill from 'vite-plugin-node-polyfills/shims/buffer'
globalThis.Buffer = globalThis.Buffer || __buffer_polyfill
import __global_polyfill from 'vite-plugin-node-polyfills/shims/global'
globalThis.global = globalThis.global || __global_polyfill
import __process_polyfill from 'vite-plugin-node-polyfills/shims/process'
globalThis.process = globalThis.process || __process_polyfill

"use client";
import {
  Typography_default,
  getTypographyUtilityClass,
  typographyClasses_default
} from "./chunk-2GDUN635.js";
import "./chunk-5PJKUT6B.js";
import "./chunk-UG57NDXK.js";
import "./chunk-ALJKGM6N.js";
import "./chunk-L2U4KOIB.js";
import "./chunk-62FJX24Z.js";
import "./chunk-CJEDDXZD.js";
import "./chunk-GJFZQ5ET.js";
export {
  Typography_default as default,
  getTypographyUtilityClass,
  typographyClasses_default as typographyClasses
};
