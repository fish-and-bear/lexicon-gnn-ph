const webpack = require('webpack');

module.exports = {
  webpack: {
    configure: (webpackConfig, { env, paths }) => {
      // Polyfill Node.js core modules for Webpack 5+
      webpackConfig.resolve.fallback = {
        ...webpackConfig.resolve.fallback, // Spread existing fallbacks if any
        "https": require.resolve("https-browserify"),
        "http": require.resolve("stream-http"),
        "stream": require.resolve("stream-browserify"),
        "crypto": require.resolve("crypto-browserify"),
        "zlib": require.resolve("browserify-zlib"),
        "path": require.resolve("path-browserify"),
        "process": require.resolve("process/browser.js"), // Keep .js extension as required before
        "buffer": require.resolve("buffer/"),
        "fs": false,
        "net": false,
        "tls": false
      };

      // Provide fallbacks for process and Buffer
      webpackConfig.plugins = (
        webpackConfig.plugins || []
      ).concat([
        new webpack.ProvidePlugin({
          process: 'process/browser.js', // Keep .js extension as required before
          Buffer: ['buffer', 'Buffer'],
        })
      ]);

      // --- MJS rule remains commented out for now ---
      /*
      webpackConfig.module.rules.push({
        test: /\.m?js/,
        resolve: {
          fullySpecified: false,
        },
      });
      */
     
      // Ignore source map warnings from specific packages if needed
      webpackConfig.ignoreWarnings = [...(webpackConfig.ignoreWarnings || []), /Failed to parse source map/];

      return webpackConfig;
    },
  },
}; 