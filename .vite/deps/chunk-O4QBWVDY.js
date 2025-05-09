import __buffer_polyfill from 'vite-plugin-node-polyfills/shims/buffer'
globalThis.Buffer = globalThis.Buffer || __buffer_polyfill
import __global_polyfill from 'vite-plugin-node-polyfills/shims/global'
globalThis.global = globalThis.global || __global_polyfill
import __process_polyfill from 'vite-plugin-node-polyfills/shims/process'
globalThis.process = globalThis.process || __process_polyfill

import {
  useEnhancedEffect_default
} from "./chunk-BZBXATFF.js";
import {
  clsx_default,
  init_clamp,
  init_getDisplayName
} from "./chunk-ALJKGM6N.js";
import {
  _extends,
  _objectWithoutPropertiesLoose,
  init_capitalize,
  init_deepmerge,
  init_extends,
  init_formatMuiErrorMessage,
  require_prop_types
} from "./chunk-L2U4KOIB.js";
import {
  require_jsx_runtime
} from "./chunk-62FJX24Z.js";
import {
  require_react
} from "./chunk-CJEDDXZD.js";
import {
  __toESM,
  require_dist,
  require_dist2,
  require_dist3
} from "./chunk-GJFZQ5ET.js";

// node_modules/@mui/utils/esm/index.js
var import_dist226 = __toESM(require_dist());
var import_dist227 = __toESM(require_dist2());
var import_dist228 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/chainPropTypes/index.js
var import_dist4 = __toESM(require_dist());
var import_dist5 = __toESM(require_dist2());
var import_dist6 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/chainPropTypes/chainPropTypes.js
var import_dist = __toESM(require_dist());
var import_dist2 = __toESM(require_dist2());
var import_dist3 = __toESM(require_dist3());
function chainPropTypes(propType1, propType2) {
  if (process.env.NODE_ENV === "production") {
    return () => null;
  }
  return function validate(...args) {
    return propType1(...args) || propType2(...args);
  };
}

// node_modules/@mui/utils/esm/index.js
init_deepmerge();
init_deepmerge();

// node_modules/@mui/utils/esm/elementAcceptingRef/index.js
var import_dist10 = __toESM(require_dist());
var import_dist11 = __toESM(require_dist2());
var import_dist12 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/elementAcceptingRef/elementAcceptingRef.js
var import_dist7 = __toESM(require_dist());
var import_dist8 = __toESM(require_dist2());
var import_dist9 = __toESM(require_dist3());
var import_prop_types = __toESM(require_prop_types());
function isClassComponent(elementType) {
  const {
    prototype = {}
  } = elementType;
  return Boolean(prototype.isReactComponent);
}
function acceptingRef(props, propName, componentName, location, propFullName) {
  const element = props[propName];
  const safePropName = propFullName || propName;
  if (element == null || // When server-side rendering React doesn't warn either.
  // This is not an accurate check for SSR.
  // This is only in place for Emotion compat.
  // TODO: Revisit once https://github.com/facebook/react/issues/20047 is resolved.
  typeof window === "undefined") {
    return null;
  }
  let warningHint;
  const elementType = element.type;
  if (typeof elementType === "function" && !isClassComponent(elementType)) {
    warningHint = "Did you accidentally use a plain function component for an element instead?";
  }
  if (warningHint !== void 0) {
    return new Error(`Invalid ${location} \`${safePropName}\` supplied to \`${componentName}\`. Expected an element that can hold a ref. ${warningHint} For more information see https://mui.com/r/caveat-with-refs-guide`);
  }
  return null;
}
var elementAcceptingRef = chainPropTypes(import_prop_types.default.element, acceptingRef);
elementAcceptingRef.isRequired = chainPropTypes(import_prop_types.default.element.isRequired, acceptingRef);
var elementAcceptingRef_default = elementAcceptingRef;

// node_modules/@mui/utils/esm/elementTypeAcceptingRef/index.js
var import_dist16 = __toESM(require_dist());
var import_dist17 = __toESM(require_dist2());
var import_dist18 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/elementTypeAcceptingRef/elementTypeAcceptingRef.js
var import_dist13 = __toESM(require_dist());
var import_dist14 = __toESM(require_dist2());
var import_dist15 = __toESM(require_dist3());
var import_prop_types2 = __toESM(require_prop_types());
function isClassComponent2(elementType) {
  const {
    prototype = {}
  } = elementType;
  return Boolean(prototype.isReactComponent);
}
function elementTypeAcceptingRef(props, propName, componentName, location, propFullName) {
  const propValue = props[propName];
  const safePropName = propFullName || propName;
  if (propValue == null || // When server-side rendering React doesn't warn either.
  // This is not an accurate check for SSR.
  // This is only in place for emotion compat.
  // TODO: Revisit once https://github.com/facebook/react/issues/20047 is resolved.
  typeof window === "undefined") {
    return null;
  }
  let warningHint;
  if (typeof propValue === "function" && !isClassComponent2(propValue)) {
    warningHint = "Did you accidentally provide a plain function component instead?";
  }
  if (warningHint !== void 0) {
    return new Error(`Invalid ${location} \`${safePropName}\` supplied to \`${componentName}\`. Expected an element type that can hold a ref. ${warningHint} For more information see https://mui.com/r/caveat-with-refs-guide`);
  }
  return null;
}
var elementTypeAcceptingRef_default = chainPropTypes(import_prop_types2.default.elementType, elementTypeAcceptingRef);

// node_modules/@mui/utils/esm/exactProp/index.js
var import_dist22 = __toESM(require_dist());
var import_dist23 = __toESM(require_dist2());
var import_dist24 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/exactProp/exactProp.js
var import_dist19 = __toESM(require_dist());
var import_dist20 = __toESM(require_dist2());
var import_dist21 = __toESM(require_dist3());
init_extends();
var specialProperty = "exact-prop: ​";
function exactProp(propTypes) {
  if (process.env.NODE_ENV === "production") {
    return propTypes;
  }
  return _extends({}, propTypes, {
    [specialProperty]: (props) => {
      const unsupportedProps = Object.keys(props).filter((prop) => !propTypes.hasOwnProperty(prop));
      if (unsupportedProps.length > 0) {
        return new Error(`The following props are not supported: ${unsupportedProps.map((prop) => `\`${prop}\``).join(", ")}. Please remove them.`);
      }
      return null;
    }
  });
}

// node_modules/@mui/utils/esm/index.js
init_formatMuiErrorMessage();
init_getDisplayName();

// node_modules/@mui/utils/esm/HTMLElementType/index.js
var import_dist28 = __toESM(require_dist());
var import_dist29 = __toESM(require_dist2());
var import_dist30 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/HTMLElementType/HTMLElementType.js
var import_dist25 = __toESM(require_dist());
var import_dist26 = __toESM(require_dist2());
var import_dist27 = __toESM(require_dist3());
function HTMLElementType(props, propName, componentName, location, propFullName) {
  if (process.env.NODE_ENV === "production") {
    return null;
  }
  const propValue = props[propName];
  const safePropName = propFullName || propName;
  if (propValue == null) {
    return null;
  }
  if (propValue && propValue.nodeType !== 1) {
    return new Error(`Invalid ${location} \`${safePropName}\` supplied to \`${componentName}\`. Expected an HTMLElement.`);
  }
  return null;
}

// node_modules/@mui/utils/esm/ponyfillGlobal/index.js
var import_dist34 = __toESM(require_dist());
var import_dist35 = __toESM(require_dist2());
var import_dist36 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/ponyfillGlobal/ponyfillGlobal.js
var import_dist31 = __toESM(require_dist());
var import_dist32 = __toESM(require_dist2());
var import_dist33 = __toESM(require_dist3());
var ponyfillGlobal_default = typeof window != "undefined" && window.Math == Math ? window : typeof self != "undefined" && self.Math == Math ? self : Function("return this")();

// node_modules/@mui/utils/esm/refType/index.js
var import_dist40 = __toESM(require_dist());
var import_dist41 = __toESM(require_dist2());
var import_dist42 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/refType/refType.js
var import_dist37 = __toESM(require_dist());
var import_dist38 = __toESM(require_dist2());
var import_dist39 = __toESM(require_dist3());
var import_prop_types3 = __toESM(require_prop_types());
var refType = import_prop_types3.default.oneOfType([import_prop_types3.default.func, import_prop_types3.default.object]);
var refType_default = refType;

// node_modules/@mui/utils/esm/index.js
init_capitalize();

// node_modules/@mui/utils/esm/createChainedFunction/index.js
var import_dist46 = __toESM(require_dist());
var import_dist47 = __toESM(require_dist2());
var import_dist48 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/createChainedFunction/createChainedFunction.js
var import_dist43 = __toESM(require_dist());
var import_dist44 = __toESM(require_dist2());
var import_dist45 = __toESM(require_dist3());
function createChainedFunction(...funcs) {
  return funcs.reduce((acc, func) => {
    if (func == null) {
      return acc;
    }
    return function chainedFunction(...args) {
      acc.apply(this, args);
      func.apply(this, args);
    };
  }, () => {
  });
}

// node_modules/@mui/utils/esm/debounce/index.js
var import_dist52 = __toESM(require_dist());
var import_dist53 = __toESM(require_dist2());
var import_dist54 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/debounce/debounce.js
var import_dist49 = __toESM(require_dist());
var import_dist50 = __toESM(require_dist2());
var import_dist51 = __toESM(require_dist3());
function debounce(func, wait = 166) {
  let timeout;
  function debounced(...args) {
    const later = () => {
      func.apply(this, args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  }
  debounced.clear = () => {
    clearTimeout(timeout);
  };
  return debounced;
}

// node_modules/@mui/utils/esm/deprecatedPropType/index.js
var import_dist58 = __toESM(require_dist());
var import_dist59 = __toESM(require_dist2());
var import_dist60 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/deprecatedPropType/deprecatedPropType.js
var import_dist55 = __toESM(require_dist());
var import_dist56 = __toESM(require_dist2());
var import_dist57 = __toESM(require_dist3());
function deprecatedPropType(validator2, reason) {
  if (process.env.NODE_ENV === "production") {
    return () => null;
  }
  return (props, propName, componentName, location, propFullName) => {
    const componentNameSafe = componentName || "<<anonymous>>";
    const propFullNameSafe = propFullName || propName;
    if (typeof props[propName] !== "undefined") {
      return new Error(`The ${location} \`${propFullNameSafe}\` of \`${componentNameSafe}\` is deprecated. ${reason}`);
    }
    return null;
  };
}

// node_modules/@mui/utils/esm/ownerDocument/index.js
var import_dist64 = __toESM(require_dist());
var import_dist65 = __toESM(require_dist2());
var import_dist66 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/ownerDocument/ownerDocument.js
var import_dist61 = __toESM(require_dist());
var import_dist62 = __toESM(require_dist2());
var import_dist63 = __toESM(require_dist3());
function ownerDocument(node) {
  return node && node.ownerDocument || document;
}

// node_modules/@mui/utils/esm/ownerWindow/index.js
var import_dist70 = __toESM(require_dist());
var import_dist71 = __toESM(require_dist2());
var import_dist72 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/ownerWindow/ownerWindow.js
var import_dist67 = __toESM(require_dist());
var import_dist68 = __toESM(require_dist2());
var import_dist69 = __toESM(require_dist3());
function ownerWindow(node) {
  const doc = ownerDocument(node);
  return doc.defaultView || window;
}

// node_modules/@mui/utils/esm/requirePropFactory/index.js
var import_dist76 = __toESM(require_dist());
var import_dist77 = __toESM(require_dist2());
var import_dist78 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/requirePropFactory/requirePropFactory.js
var import_dist73 = __toESM(require_dist());
var import_dist74 = __toESM(require_dist2());
var import_dist75 = __toESM(require_dist3());
init_extends();
function requirePropFactory(componentNameInError, Component) {
  if (process.env.NODE_ENV === "production") {
    return () => null;
  }
  const prevPropTypes = Component ? _extends({}, Component.propTypes) : null;
  const requireProp = (requiredProp) => (props, propName, componentName, location, propFullName, ...args) => {
    const propFullNameSafe = propFullName || propName;
    const defaultTypeChecker = prevPropTypes == null ? void 0 : prevPropTypes[propFullNameSafe];
    if (defaultTypeChecker) {
      const typeCheckerResult = defaultTypeChecker(props, propName, componentName, location, propFullName, ...args);
      if (typeCheckerResult) {
        return typeCheckerResult;
      }
    }
    if (typeof props[propName] !== "undefined" && !props[requiredProp]) {
      return new Error(`The prop \`${propFullNameSafe}\` of \`${componentNameInError}\` can only be used together with the \`${requiredProp}\` prop.`);
    }
    return null;
  };
  return requireProp;
}

// node_modules/@mui/utils/esm/setRef/index.js
var import_dist82 = __toESM(require_dist());
var import_dist83 = __toESM(require_dist2());
var import_dist84 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/setRef/setRef.js
var import_dist79 = __toESM(require_dist());
var import_dist80 = __toESM(require_dist2());
var import_dist81 = __toESM(require_dist3());
function setRef(ref, value) {
  if (typeof ref === "function") {
    ref(value);
  } else if (ref) {
    ref.current = value;
  }
}

// node_modules/@mui/utils/esm/useId/index.js
var import_dist88 = __toESM(require_dist());
var import_dist89 = __toESM(require_dist2());
var import_dist90 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/useId/useId.js
var import_dist85 = __toESM(require_dist());
var import_dist86 = __toESM(require_dist2());
var import_dist87 = __toESM(require_dist3());
var React = __toESM(require_react());
var globalId = 0;
function useGlobalId(idOverride) {
  const [defaultId, setDefaultId] = React.useState(idOverride);
  const id = idOverride || defaultId;
  React.useEffect(() => {
    if (defaultId == null) {
      globalId += 1;
      setDefaultId(`mui-${globalId}`);
    }
  }, [defaultId]);
  return id;
}
var maybeReactUseId = React["useId".toString()];
function useId(idOverride) {
  if (maybeReactUseId !== void 0) {
    const reactId = maybeReactUseId();
    return idOverride != null ? idOverride : reactId;
  }
  return useGlobalId(idOverride);
}

// node_modules/@mui/utils/esm/unsupportedProp/index.js
var import_dist94 = __toESM(require_dist());
var import_dist95 = __toESM(require_dist2());
var import_dist96 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/unsupportedProp/unsupportedProp.js
var import_dist91 = __toESM(require_dist());
var import_dist92 = __toESM(require_dist2());
var import_dist93 = __toESM(require_dist3());
function unsupportedProp(props, propName, componentName, location, propFullName) {
  if (process.env.NODE_ENV === "production") {
    return null;
  }
  const propFullNameSafe = propFullName || propName;
  if (typeof props[propName] !== "undefined") {
    return new Error(`The prop \`${propFullNameSafe}\` is not supported. Please remove it.`);
  }
  return null;
}

// node_modules/@mui/utils/esm/useControlled/index.js
var import_dist100 = __toESM(require_dist());
var import_dist101 = __toESM(require_dist2());
var import_dist102 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/useControlled/useControlled.js
var import_dist97 = __toESM(require_dist());
var import_dist98 = __toESM(require_dist2());
var import_dist99 = __toESM(require_dist3());
var React2 = __toESM(require_react());
function useControlled({
  controlled,
  default: defaultProp,
  name,
  state = "value"
}) {
  const {
    current: isControlled
  } = React2.useRef(controlled !== void 0);
  const [valueState, setValue] = React2.useState(defaultProp);
  const value = isControlled ? controlled : valueState;
  if (process.env.NODE_ENV !== "production") {
    React2.useEffect(() => {
      if (isControlled !== (controlled !== void 0)) {
        console.error([`MUI: A component is changing the ${isControlled ? "" : "un"}controlled ${state} state of ${name} to be ${isControlled ? "un" : ""}controlled.`, "Elements should not switch from uncontrolled to controlled (or vice versa).", `Decide between using a controlled or uncontrolled ${name} element for the lifetime of the component.`, "The nature of the state is determined during the first render. It's considered controlled if the value is not `undefined`.", "More info: https://fb.me/react-controlled-components"].join("\n"));
      }
    }, [state, name, controlled]);
    const {
      current: defaultValue
    } = React2.useRef(defaultProp);
    React2.useEffect(() => {
      if (!isControlled && !Object.is(defaultValue, defaultProp)) {
        console.error([`MUI: A component is changing the default ${state} state of an uncontrolled ${name} after being initialized. To suppress this warning opt to use a controlled ${name}.`].join("\n"));
      }
    }, [JSON.stringify(defaultProp)]);
  }
  const setValueIfUncontrolled = React2.useCallback((newValue) => {
    if (!isControlled) {
      setValue(newValue);
    }
  }, []);
  return [value, setValueIfUncontrolled];
}

// node_modules/@mui/utils/esm/useEventCallback/index.js
var import_dist106 = __toESM(require_dist());
var import_dist107 = __toESM(require_dist2());
var import_dist108 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/useEventCallback/useEventCallback.js
var import_dist103 = __toESM(require_dist());
var import_dist104 = __toESM(require_dist2());
var import_dist105 = __toESM(require_dist3());
var React3 = __toESM(require_react());
function useEventCallback(fn) {
  const ref = React3.useRef(fn);
  useEnhancedEffect_default(() => {
    ref.current = fn;
  });
  return React3.useRef((...args) => (
    // @ts-expect-error hide `this`
    (0, ref.current)(...args)
  )).current;
}
var useEventCallback_default = useEventCallback;

// node_modules/@mui/utils/esm/useForkRef/index.js
var import_dist112 = __toESM(require_dist());
var import_dist113 = __toESM(require_dist2());
var import_dist114 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/useForkRef/useForkRef.js
var import_dist109 = __toESM(require_dist());
var import_dist110 = __toESM(require_dist2());
var import_dist111 = __toESM(require_dist3());
var React4 = __toESM(require_react());
function useForkRef(...refs) {
  return React4.useMemo(() => {
    if (refs.every((ref) => ref == null)) {
      return null;
    }
    return (instance) => {
      refs.forEach((ref) => {
        setRef(ref, instance);
      });
    };
  }, refs);
}

// node_modules/@mui/utils/esm/useLazyRef/index.js
var import_dist118 = __toESM(require_dist());
var import_dist119 = __toESM(require_dist2());
var import_dist120 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/useLazyRef/useLazyRef.js
var import_dist115 = __toESM(require_dist());
var import_dist116 = __toESM(require_dist2());
var import_dist117 = __toESM(require_dist3());
var React5 = __toESM(require_react());
var UNINITIALIZED = {};
function useLazyRef(init, initArg) {
  const ref = React5.useRef(UNINITIALIZED);
  if (ref.current === UNINITIALIZED) {
    ref.current = init(initArg);
  }
  return ref;
}

// node_modules/@mui/utils/esm/useTimeout/index.js
var import_dist127 = __toESM(require_dist());
var import_dist128 = __toESM(require_dist2());
var import_dist129 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/useTimeout/useTimeout.js
var import_dist124 = __toESM(require_dist());
var import_dist125 = __toESM(require_dist2());
var import_dist126 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/useOnMount/useOnMount.js
var import_dist121 = __toESM(require_dist());
var import_dist122 = __toESM(require_dist2());
var import_dist123 = __toESM(require_dist3());
var React6 = __toESM(require_react());
var EMPTY = [];
function useOnMount(fn) {
  React6.useEffect(fn, EMPTY);
}

// node_modules/@mui/utils/esm/useTimeout/useTimeout.js
var Timeout = class _Timeout {
  constructor() {
    this.currentId = null;
    this.clear = () => {
      if (this.currentId !== null) {
        clearTimeout(this.currentId);
        this.currentId = null;
      }
    };
    this.disposeEffect = () => {
      return this.clear;
    };
  }
  static create() {
    return new _Timeout();
  }
  /**
   * Executes `fn` after `delay`, clearing any previously scheduled call.
   */
  start(delay, fn) {
    this.clear();
    this.currentId = setTimeout(() => {
      this.currentId = null;
      fn();
    }, delay);
  }
};
function useTimeout() {
  const timeout = useLazyRef(Timeout.create).current;
  useOnMount(timeout.disposeEffect);
  return timeout;
}

// node_modules/@mui/utils/esm/useOnMount/index.js
var import_dist130 = __toESM(require_dist());
var import_dist131 = __toESM(require_dist2());
var import_dist132 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/useIsFocusVisible/index.js
var import_dist136 = __toESM(require_dist());
var import_dist137 = __toESM(require_dist2());
var import_dist138 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/useIsFocusVisible/useIsFocusVisible.js
var import_dist133 = __toESM(require_dist());
var import_dist134 = __toESM(require_dist2());
var import_dist135 = __toESM(require_dist3());
var React7 = __toESM(require_react());
var hadKeyboardEvent = true;
var hadFocusVisibleRecently = false;
var hadFocusVisibleRecentlyTimeout = new Timeout();
var inputTypesWhitelist = {
  text: true,
  search: true,
  url: true,
  tel: true,
  email: true,
  password: true,
  number: true,
  date: true,
  month: true,
  week: true,
  time: true,
  datetime: true,
  "datetime-local": true
};
function focusTriggersKeyboardModality(node) {
  const {
    type,
    tagName
  } = node;
  if (tagName === "INPUT" && inputTypesWhitelist[type] && !node.readOnly) {
    return true;
  }
  if (tagName === "TEXTAREA" && !node.readOnly) {
    return true;
  }
  if (node.isContentEditable) {
    return true;
  }
  return false;
}
function handleKeyDown(event) {
  if (event.metaKey || event.altKey || event.ctrlKey) {
    return;
  }
  hadKeyboardEvent = true;
}
function handlePointerDown() {
  hadKeyboardEvent = false;
}
function handleVisibilityChange() {
  if (this.visibilityState === "hidden") {
    if (hadFocusVisibleRecently) {
      hadKeyboardEvent = true;
    }
  }
}
function prepare(doc) {
  doc.addEventListener("keydown", handleKeyDown, true);
  doc.addEventListener("mousedown", handlePointerDown, true);
  doc.addEventListener("pointerdown", handlePointerDown, true);
  doc.addEventListener("touchstart", handlePointerDown, true);
  doc.addEventListener("visibilitychange", handleVisibilityChange, true);
}
function isFocusVisible(event) {
  const {
    target
  } = event;
  try {
    return target.matches(":focus-visible");
  } catch (error) {
  }
  return hadKeyboardEvent || focusTriggersKeyboardModality(target);
}
function useIsFocusVisible() {
  const ref = React7.useCallback((node) => {
    if (node != null) {
      prepare(node.ownerDocument);
    }
  }, []);
  const isFocusVisibleRef = React7.useRef(false);
  function handleBlurVisible() {
    if (isFocusVisibleRef.current) {
      hadFocusVisibleRecently = true;
      hadFocusVisibleRecentlyTimeout.start(100, () => {
        hadFocusVisibleRecently = false;
      });
      isFocusVisibleRef.current = false;
      return true;
    }
    return false;
  }
  function handleFocusVisible(event) {
    if (isFocusVisible(event)) {
      isFocusVisibleRef.current = true;
      return true;
    }
    return false;
  }
  return {
    isFocusVisibleRef,
    onFocus: handleFocusVisible,
    onBlur: handleBlurVisible,
    ref
  };
}

// node_modules/@mui/utils/esm/getScrollbarSize/index.js
var import_dist142 = __toESM(require_dist());
var import_dist143 = __toESM(require_dist2());
var import_dist144 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/getScrollbarSize/getScrollbarSize.js
var import_dist139 = __toESM(require_dist());
var import_dist140 = __toESM(require_dist2());
var import_dist141 = __toESM(require_dist3());
function getScrollbarSize(doc) {
  const documentWidth = doc.documentElement.clientWidth;
  return Math.abs(window.innerWidth - documentWidth);
}

// node_modules/@mui/utils/esm/scrollLeft/index.js
var import_dist148 = __toESM(require_dist());
var import_dist149 = __toESM(require_dist2());
var import_dist150 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/scrollLeft/scrollLeft.js
var import_dist145 = __toESM(require_dist());
var import_dist146 = __toESM(require_dist2());
var import_dist147 = __toESM(require_dist3());
var cachedType;
function detectScrollType() {
  if (cachedType) {
    return cachedType;
  }
  const dummy = document.createElement("div");
  const container = document.createElement("div");
  container.style.width = "10px";
  container.style.height = "1px";
  dummy.appendChild(container);
  dummy.dir = "rtl";
  dummy.style.fontSize = "14px";
  dummy.style.width = "4px";
  dummy.style.height = "1px";
  dummy.style.position = "absolute";
  dummy.style.top = "-1000px";
  dummy.style.overflow = "scroll";
  document.body.appendChild(dummy);
  cachedType = "reverse";
  if (dummy.scrollLeft > 0) {
    cachedType = "default";
  } else {
    dummy.scrollLeft = 1;
    if (dummy.scrollLeft === 0) {
      cachedType = "negative";
    }
  }
  document.body.removeChild(dummy);
  return cachedType;
}
function getNormalizedScrollLeft(element, direction) {
  const scrollLeft = element.scrollLeft;
  if (direction !== "rtl") {
    return scrollLeft;
  }
  const type = detectScrollType();
  switch (type) {
    case "negative":
      return element.scrollWidth - element.clientWidth + scrollLeft;
    case "reverse":
      return element.scrollWidth - element.clientWidth - scrollLeft;
    default:
      return scrollLeft;
  }
}

// node_modules/@mui/utils/esm/usePreviousProps/index.js
var import_dist154 = __toESM(require_dist());
var import_dist155 = __toESM(require_dist2());
var import_dist156 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/usePreviousProps/usePreviousProps.js
var import_dist151 = __toESM(require_dist());
var import_dist152 = __toESM(require_dist2());
var import_dist153 = __toESM(require_dist3());
var React8 = __toESM(require_react());
var usePreviousProps = (value) => {
  const ref = React8.useRef({});
  React8.useEffect(() => {
    ref.current = value;
  });
  return ref.current;
};
var usePreviousProps_default = usePreviousProps;

// node_modules/@mui/utils/esm/getValidReactChildren/index.js
var import_dist160 = __toESM(require_dist());
var import_dist161 = __toESM(require_dist2());
var import_dist162 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/getValidReactChildren/getValidReactChildren.js
var import_dist157 = __toESM(require_dist());
var import_dist158 = __toESM(require_dist2());
var import_dist159 = __toESM(require_dist3());
var React9 = __toESM(require_react());
function getValidReactChildren(children) {
  return React9.Children.toArray(children).filter((child) => React9.isValidElement(child));
}

// node_modules/@mui/utils/esm/visuallyHidden/index.js
var import_dist166 = __toESM(require_dist());
var import_dist167 = __toESM(require_dist2());
var import_dist168 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/visuallyHidden/visuallyHidden.js
var import_dist163 = __toESM(require_dist());
var import_dist164 = __toESM(require_dist2());
var import_dist165 = __toESM(require_dist3());
var visuallyHidden = {
  border: 0,
  clip: "rect(0 0 0 0)",
  height: "1px",
  margin: "-1px",
  overflow: "hidden",
  padding: 0,
  position: "absolute",
  whiteSpace: "nowrap",
  width: "1px"
};
var visuallyHidden_default = visuallyHidden;

// node_modules/@mui/utils/esm/integerPropType/index.js
var import_dist172 = __toESM(require_dist());
var import_dist173 = __toESM(require_dist2());
var import_dist174 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/integerPropType/integerPropType.js
var import_dist169 = __toESM(require_dist());
var import_dist170 = __toESM(require_dist2());
var import_dist171 = __toESM(require_dist3());
function getTypeByValue(value) {
  const valueType = typeof value;
  switch (valueType) {
    case "number":
      if (Number.isNaN(value)) {
        return "NaN";
      }
      if (!Number.isFinite(value)) {
        return "Infinity";
      }
      if (value !== Math.floor(value)) {
        return "float";
      }
      return "number";
    case "object":
      if (value === null) {
        return "null";
      }
      return value.constructor.name;
    default:
      return valueType;
  }
}
function ponyfillIsInteger(x) {
  return typeof x === "number" && isFinite(x) && Math.floor(x) === x;
}
var isInteger = Number.isInteger || ponyfillIsInteger;
function requiredInteger(props, propName, componentName, location) {
  const propValue = props[propName];
  if (propValue == null || !isInteger(propValue)) {
    const propType = getTypeByValue(propValue);
    return new RangeError(`Invalid ${location} \`${propName}\` of type \`${propType}\` supplied to \`${componentName}\`, expected \`integer\`.`);
  }
  return null;
}
function validator(props, propName, ...other) {
  const propValue = props[propName];
  if (propValue === void 0) {
    return null;
  }
  return requiredInteger(props, propName, ...other);
}
function validatorNoop() {
  return null;
}
validator.isRequired = requiredInteger;
validatorNoop.isRequired = validatorNoop;
var integerPropType_default = process.env.NODE_ENV === "production" ? validatorNoop : validator;

// node_modules/@mui/utils/esm/index.js
init_clamp();

// node_modules/@mui/utils/esm/useSlotProps/index.js
var import_dist214 = __toESM(require_dist());
var import_dist215 = __toESM(require_dist2());
var import_dist216 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/useSlotProps/useSlotProps.js
var import_dist211 = __toESM(require_dist());
var import_dist212 = __toESM(require_dist2());
var import_dist213 = __toESM(require_dist3());
init_extends();

// node_modules/@mui/utils/esm/appendOwnerState/index.js
var import_dist184 = __toESM(require_dist());
var import_dist185 = __toESM(require_dist2());
var import_dist186 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/appendOwnerState/appendOwnerState.js
var import_dist181 = __toESM(require_dist());
var import_dist182 = __toESM(require_dist2());
var import_dist183 = __toESM(require_dist3());
init_extends();

// node_modules/@mui/utils/esm/isHostComponent/index.js
var import_dist178 = __toESM(require_dist());
var import_dist179 = __toESM(require_dist2());
var import_dist180 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/isHostComponent/isHostComponent.js
var import_dist175 = __toESM(require_dist());
var import_dist176 = __toESM(require_dist2());
var import_dist177 = __toESM(require_dist3());
function isHostComponent(element) {
  return typeof element === "string";
}
var isHostComponent_default = isHostComponent;

// node_modules/@mui/utils/esm/appendOwnerState/appendOwnerState.js
function appendOwnerState(elementType, otherProps, ownerState) {
  if (elementType === void 0 || isHostComponent_default(elementType)) {
    return otherProps;
  }
  return _extends({}, otherProps, {
    ownerState: _extends({}, otherProps.ownerState, ownerState)
  });
}
var appendOwnerState_default = appendOwnerState;

// node_modules/@mui/utils/esm/mergeSlotProps/index.js
var import_dist202 = __toESM(require_dist());
var import_dist203 = __toESM(require_dist2());
var import_dist204 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/mergeSlotProps/mergeSlotProps.js
var import_dist199 = __toESM(require_dist());
var import_dist200 = __toESM(require_dist2());
var import_dist201 = __toESM(require_dist3());
init_extends();

// node_modules/@mui/utils/esm/extractEventHandlers/index.js
var import_dist190 = __toESM(require_dist());
var import_dist191 = __toESM(require_dist2());
var import_dist192 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/extractEventHandlers/extractEventHandlers.js
var import_dist187 = __toESM(require_dist());
var import_dist188 = __toESM(require_dist2());
var import_dist189 = __toESM(require_dist3());
function extractEventHandlers(object, excludeKeys = []) {
  if (object === void 0) {
    return {};
  }
  const result = {};
  Object.keys(object).filter((prop) => prop.match(/^on[A-Z]/) && typeof object[prop] === "function" && !excludeKeys.includes(prop)).forEach((prop) => {
    result[prop] = object[prop];
  });
  return result;
}
var extractEventHandlers_default = extractEventHandlers;

// node_modules/@mui/utils/esm/omitEventHandlers/index.js
var import_dist196 = __toESM(require_dist());
var import_dist197 = __toESM(require_dist2());
var import_dist198 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/omitEventHandlers/omitEventHandlers.js
var import_dist193 = __toESM(require_dist());
var import_dist194 = __toESM(require_dist2());
var import_dist195 = __toESM(require_dist3());
function omitEventHandlers(object) {
  if (object === void 0) {
    return {};
  }
  const result = {};
  Object.keys(object).filter((prop) => !(prop.match(/^on[A-Z]/) && typeof object[prop] === "function")).forEach((prop) => {
    result[prop] = object[prop];
  });
  return result;
}
var omitEventHandlers_default = omitEventHandlers;

// node_modules/@mui/utils/esm/mergeSlotProps/mergeSlotProps.js
function mergeSlotProps(parameters) {
  const {
    getSlotProps,
    additionalProps,
    externalSlotProps,
    externalForwardedProps,
    className
  } = parameters;
  if (!getSlotProps) {
    const joinedClasses2 = clsx_default(additionalProps == null ? void 0 : additionalProps.className, className, externalForwardedProps == null ? void 0 : externalForwardedProps.className, externalSlotProps == null ? void 0 : externalSlotProps.className);
    const mergedStyle2 = _extends({}, additionalProps == null ? void 0 : additionalProps.style, externalForwardedProps == null ? void 0 : externalForwardedProps.style, externalSlotProps == null ? void 0 : externalSlotProps.style);
    const props2 = _extends({}, additionalProps, externalForwardedProps, externalSlotProps);
    if (joinedClasses2.length > 0) {
      props2.className = joinedClasses2;
    }
    if (Object.keys(mergedStyle2).length > 0) {
      props2.style = mergedStyle2;
    }
    return {
      props: props2,
      internalRef: void 0
    };
  }
  const eventHandlers = extractEventHandlers_default(_extends({}, externalForwardedProps, externalSlotProps));
  const componentsPropsWithoutEventHandlers = omitEventHandlers_default(externalSlotProps);
  const otherPropsWithoutEventHandlers = omitEventHandlers_default(externalForwardedProps);
  const internalSlotProps = getSlotProps(eventHandlers);
  const joinedClasses = clsx_default(internalSlotProps == null ? void 0 : internalSlotProps.className, additionalProps == null ? void 0 : additionalProps.className, className, externalForwardedProps == null ? void 0 : externalForwardedProps.className, externalSlotProps == null ? void 0 : externalSlotProps.className);
  const mergedStyle = _extends({}, internalSlotProps == null ? void 0 : internalSlotProps.style, additionalProps == null ? void 0 : additionalProps.style, externalForwardedProps == null ? void 0 : externalForwardedProps.style, externalSlotProps == null ? void 0 : externalSlotProps.style);
  const props = _extends({}, internalSlotProps, additionalProps, otherPropsWithoutEventHandlers, componentsPropsWithoutEventHandlers);
  if (joinedClasses.length > 0) {
    props.className = joinedClasses;
  }
  if (Object.keys(mergedStyle).length > 0) {
    props.style = mergedStyle;
  }
  return {
    props,
    internalRef: internalSlotProps.ref
  };
}
var mergeSlotProps_default = mergeSlotProps;

// node_modules/@mui/utils/esm/resolveComponentProps/index.js
var import_dist208 = __toESM(require_dist());
var import_dist209 = __toESM(require_dist2());
var import_dist210 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/resolveComponentProps/resolveComponentProps.js
var import_dist205 = __toESM(require_dist());
var import_dist206 = __toESM(require_dist2());
var import_dist207 = __toESM(require_dist3());
function resolveComponentProps(componentProps, ownerState, slotState) {
  if (typeof componentProps === "function") {
    return componentProps(ownerState, slotState);
  }
  return componentProps;
}
var resolveComponentProps_default = resolveComponentProps;

// node_modules/@mui/utils/esm/useSlotProps/useSlotProps.js
var _excluded = ["elementType", "externalSlotProps", "ownerState", "skipResolvingSlotProps"];
function useSlotProps(parameters) {
  var _parameters$additiona;
  const {
    elementType,
    externalSlotProps,
    ownerState,
    skipResolvingSlotProps = false
  } = parameters, rest = _objectWithoutPropertiesLoose(parameters, _excluded);
  const resolvedComponentsProps = skipResolvingSlotProps ? {} : resolveComponentProps_default(externalSlotProps, ownerState);
  const {
    props: mergedProps,
    internalRef
  } = mergeSlotProps_default(_extends({}, rest, {
    externalSlotProps: resolvedComponentsProps
  }));
  const ref = useForkRef(internalRef, resolvedComponentsProps == null ? void 0 : resolvedComponentsProps.ref, (_parameters$additiona = parameters.additionalProps) == null ? void 0 : _parameters$additiona.ref);
  const props = appendOwnerState_default(elementType, _extends({}, mergedProps, {
    ref
  }), ownerState);
  return props;
}
var useSlotProps_default = useSlotProps;

// node_modules/@mui/utils/esm/getReactElementRef/index.js
var import_dist220 = __toESM(require_dist());
var import_dist221 = __toESM(require_dist2());
var import_dist222 = __toESM(require_dist3());

// node_modules/@mui/utils/esm/getReactElementRef/getReactElementRef.js
var import_dist217 = __toESM(require_dist());
var import_dist218 = __toESM(require_dist2());
var import_dist219 = __toESM(require_dist3());
var React10 = __toESM(require_react());
function getReactElementRef(element) {
  if (parseInt(React10.version, 10) >= 19) {
    var _element$props;
    return (element == null || (_element$props = element.props) == null ? void 0 : _element$props.ref) || null;
  }
  return (element == null ? void 0 : element.ref) || null;
}

// node_modules/@mui/utils/esm/types.js
var import_dist223 = __toESM(require_dist());
var import_dist224 = __toESM(require_dist2());
var import_dist225 = __toESM(require_dist3());

// node_modules/@mui/system/esm/RtlProvider/index.js
var import_dist229 = __toESM(require_dist());
var import_dist230 = __toESM(require_dist2());
var import_dist231 = __toESM(require_dist3());
init_extends();
var React11 = __toESM(require_react());
var import_prop_types4 = __toESM(require_prop_types());
var import_jsx_runtime = __toESM(require_jsx_runtime());
var _excluded2 = ["value"];
var RtlContext = React11.createContext();
function RtlProvider(_ref) {
  let {
    value
  } = _ref, props = _objectWithoutPropertiesLoose(_ref, _excluded2);
  return (0, import_jsx_runtime.jsx)(RtlContext.Provider, _extends({
    value: value != null ? value : true
  }, props));
}
process.env.NODE_ENV !== "production" ? RtlProvider.propTypes = {
  children: import_prop_types4.default.node,
  value: import_prop_types4.default.bool
} : void 0;
var useRtl = () => {
  const value = React11.useContext(RtlContext);
  return value != null ? value : false;
};
var RtlProvider_default = RtlProvider;

export {
  chainPropTypes,
  Timeout,
  useTimeout,
  elementTypeAcceptingRef_default,
  elementAcceptingRef_default,
  exactProp,
  HTMLElementType,
  refType_default,
  createChainedFunction,
  debounce,
  deprecatedPropType,
  ownerDocument,
  ownerWindow,
  requirePropFactory,
  setRef,
  useId,
  unsupportedProp,
  useControlled,
  useEventCallback_default,
  useForkRef,
  useIsFocusVisible,
  getScrollbarSize,
  detectScrollType,
  getNormalizedScrollLeft,
  usePreviousProps_default,
  getValidReactChildren,
  visuallyHidden_default,
  integerPropType_default,
  isHostComponent_default,
  appendOwnerState_default,
  extractEventHandlers_default,
  mergeSlotProps_default,
  resolveComponentProps_default,
  useSlotProps_default,
  getReactElementRef,
  useRtl,
  RtlProvider_default
};
//# sourceMappingURL=chunk-O4QBWVDY.js.map
