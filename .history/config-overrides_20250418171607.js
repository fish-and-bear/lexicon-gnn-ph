const webpack = require('webpack');

module.exports = function override(config, env) {
  console.log("Applying webpack config overrides...");
  
  // Initialize config objects if they don't exist
  config.resolve = config.resolve || {};
  config.plugins = config.plugins || [];
  config.module = config.module || {};
  config.module.rules = config.module.rules || [];

  // Add fallbacks for Node.js core modules
  config.resolve.fallback = {
    ...config.resolve.fallback,
    "https": require.resolve("https-browserify"),
    "http": require.resolve("stream-http"),
    "stream": require.resolve("stream-browserify"),
    "crypto": require.resolve("crypto-browserify"),
    "zlib": require.resolve("browserify-zlib"),
    "path": require.resolve("path-browserify"),
    "url": require.resolve("url/"),
    "buffer": require.resolve("buffer/"),
    "process": require.resolve("process/browser"),
    "fs": false,
    "net": false,
    "tls": false,
    "child_process": false,
    "util": require.resolve("util"),
    "asset": require.resolve("assert")
  };

  // Ensure ProvidePlugin isn't added multiple times if script is re-run
  config.plugins = config.plugins.filter(
    (plugin) => !(plugin.constructor && plugin.constructor.name === 'ProvidePlugin')
  );
  // Add ProvidePlugin for process and Buffer
  config.plugins.push(
    new webpack.ProvidePlugin({
      process: 'process/browser.js',
      Buffer: ['buffer', 'Buffer'],
    })
  );
  
  // Add rule to handle .mjs files and disable fullySpecified requirement
  config.module.rules.push({
    test: /\.m?js/,
    resolve: { 
      fullySpecified: false 
    }
  });

  // Ignore specific warnings from source-map-loader
  config.ignoreWarnings = [
    ...(config.ignoreWarnings || []),
    function ignoreSourcemapsloaderWarnings(warning) {
      return (
        warning.module &&
        warning.module.resource.includes('node_modules') &&
        warning.message.includes('Failed to parse source map')
      );
    },
  ];

  console.log("Webpack config overrides applied.");
  return config;
}