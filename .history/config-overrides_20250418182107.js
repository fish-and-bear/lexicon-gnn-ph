const webpack = require('webpack');

module.exports = function override(config, env) {
  // Removed fallback configuration to resolve Webpack schema error
  // config.resolve.fallback = {
  //   ...config.resolve.fallback,
  //   "https": require.resolve("https-browserify"),
  //   "http": require.resolve("stream-http"),
  //   "stream": require.resolve("stream-browserify"),
  //   "crypto": require.resolve("crypto-browserify"),
  //   "zlib": require.resolve("browserify-zlib"),
  //   "path": require.resolve("path-browserify"),
  //   "fs": false,
  //   "net": false,
  //   "tls": false
  // };
  config.plugins.push(
    new webpack.ProvidePlugin({
      process: 'process/browser',
      Buffer: ['buffer', 'Buffer'],
    })
  );
  return config;
}