const webpack = require('webpack');

module.exports = function override(config, env) {

  // --- Temporarily disable ALL overrides for diagnostics ---
  /*
  // Restore fallback configuration for Webpack 5 polyfills
  config.resolve.fallback = {
    ...config.resolve.fallback, // Spread existing fallbacks if any
    "https": require.resolve("https-browserify"),
    "http": require.resolve("stream-http"),
    "stream": require.resolve("stream-browserify"),
    "crypto": require.resolve("crypto-browserify"),
    "zlib": require.resolve("browserify-zlib"),
    "path": require.resolve("path-browserify"),
    "process": require.resolve("process/browser.js"), // Explicitly add process
    "buffer": require.resolve("buffer/"),         // Explicitly add buffer
    "fs": false,
    "net": false,
    "tls": false
  };
  config.plugins.push(
    new webpack.ProvidePlugin({
      process: 'process/browser.js',
      Buffer: ['buffer', 'Buffer'],
    })
  );
  */
  // --- End temporary disabling ---

  // --- Temporarily comment out MJS rule for diagnostics ---
  /*
  // Add rule for .mjs files
  config.module.rules.push({
    test: /\.m?js/,
    resolve: {
      fullySpecified: false,
    },
  });
  */
  // --- End temporary comment out ---

  return config;
}