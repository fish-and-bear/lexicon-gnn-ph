import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { nodePolyfills } from 'vite-plugin-node-polyfills';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    nodePolyfills({
      // To exclude specific polyfills, add them to this list.
      // For example, if you don't need fs polyfill:
      // exclude: ['fs'],
      // Whether to polyfill `node:` protocol imports.
      protocolImports: true,
    }),
  ],
  define: {
    // By default, Vite doesn't include shims for NodeJS/CJS globals.
    // Necessary for packages like 'process' which require these globals.
    // Ensure process is available for packages like 'process/browser.js'
    'process.env': {},
    'global': {}
  },
  // Optional: If you need to alias Buffer similar to Webpack ProvidePlugin
  // resolve: {
  //   alias: {
  //     Buffer: 'buffer/'
  //   }
  // }
}); 