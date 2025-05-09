import __buffer_polyfill from 'vite-plugin-node-polyfills/shims/buffer'
globalThis.Buffer = globalThis.Buffer || __buffer_polyfill
import __global_polyfill from 'vite-plugin-node-polyfills/shims/global'
globalThis.global = globalThis.global || __global_polyfill
import __process_polyfill from 'vite-plugin-node-polyfills/shims/process'
globalThis.process = globalThis.process || __process_polyfill

"use client";
import {
  FormControl_default,
  formControlClasses_default,
  getFormControlUtilityClasses
} from "./chunk-7M57LFIW.js";
import "./chunk-PMV6P5QU.js";
import {
  useFormControl
} from "./chunk-QXCH2KLT.js";
import "./chunk-R3U6FVON.js";
import "./chunk-5PJKUT6B.js";
import "./chunk-UG57NDXK.js";
import "./chunk-QFPGFLDC.js";
import "./chunk-ALJKGM6N.js";
import "./chunk-L2U4KOIB.js";
import "./chunk-62FJX24Z.js";
import "./chunk-CJEDDXZD.js";
import "./chunk-GJFZQ5ET.js";
export {
  FormControl_default as default,
  formControlClasses_default as formControlClasses,
  getFormControlUtilityClasses,
  useFormControl
};
