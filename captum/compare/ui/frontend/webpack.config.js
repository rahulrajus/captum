var path = require('path');

// Custom webpack rules are generally the same for all webpack bundles, hence
// stored in a separate local variable.
var rules = [
  {
    test: /\.module.css$/, use: [
      'style-loader',
      {
        loader: 'css-loader',
        options: {
          modules: true
        }
      },
    ]
  },
  { test: /\.css$/, use: ['style-loader', 'css-loader'] },
  {
    test: /\.(js|ts|tsx)$/,
    exclude: /node_modules/,
    loaders: 'babel-loader',
    options: {
      presets: ['@babel/preset-react', ['@babel/preset-env', { "targets": { "esmodules": true } }], '@babel/preset-typescript'],
      plugins: [
        "@babel/plugin-proposal-class-properties"
      ],
    },
  },
  {
    test: /\.svg$/,
    use: [
      {
        loader: '@svgr/webpack',
        options: { ref: true },
      },
      'url-loader',
    ],
  }
]

var extensions = ['.js', '.ts', '.tsx']

const config = {
  mode: 'production',
  entry: './src/index.tsx',
  devtool: 'source-map',
  module: {
    rules: rules,
  },
  resolveLoader: {
    modules: ['./node_modules']
  },
  resolve: {
    modules: ['./node_modules'],
    extensions: extensions
  },
  externals: ['@jupyter-widgets/base'],
}

module.exports = [
  {
    ...config,
    entry: "./src/index.webapp.tsx",
    output: {
      filename: 'index.js',
      path: path.resolve(__dirname, 'build'),
      library: 'lenses'
    },
  },
  {
    ...config,
    entry: "./src/index.widget.tsx",
    devtool: 'inline-source-map',
    output: {
      filename: 'widget.js',
      path: path.resolve(__dirname, '..', 'widget', 'build'),
      libraryTarget: 'umd'
    },
  }
];
