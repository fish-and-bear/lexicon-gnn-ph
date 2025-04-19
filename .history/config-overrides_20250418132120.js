const webpack = require('webpack');

module.exports = function override(config, env) {
  console.log("Applying webpack config overrides...");
  config.resolve = config.resolve || {};
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
  };

  config.plugins = config.plugins || [];
  
  config.plugins = config.plugins.filter(
    (plugin) => !(plugin.constructor && plugin.constructor.name === 'ProvidePlugin')
  );
  config.plugins.push(
    new webpack.ProvidePlugin({
      process: 'process/browser',
      Buffer: ['buffer', 'Buffer'],
    })
  );

  config.ignoreWarnings = [
    ...(config.ignoreWarnings || []),
    /Failed to parse source map/,
  ];

  console.log("Webpack config overrides applied.");
  return config;
}