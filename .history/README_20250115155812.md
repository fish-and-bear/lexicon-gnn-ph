# Filipino Root Word Explorer

An interactive web application for exploring Filipino words, their etymologies, and relationships.

![Filipino Root Word Explorer Screenshot](screenshot.png)

## Overview

Filipino Root Word Explorer is a sophisticated tool designed for linguists, language enthusiasts, and students. It offers an intuitive interface to discover the intricate connections between Filipino words, their roots, and related terms.

## Key Features

- **Interactive Word Network**: Visualize word relationships through a dynamic, force-directed graph.
- **Comprehensive Word Details**: Access in-depth information including definitions, etymologies, and usage examples.
- **Customizable Exploration**: Adjust graph depth and breadth to tailor your linguistic journey.
- **Bilingual Support**: Seamlessly switch between Filipino and English interfaces.
- **Dark Mode**: Enhanced reading comfort with a thoughtfully designed dark theme.

## Technology Stack

- **Frontend**: React with TypeScript
- **Visualization**: D3.js
- **Styling**: CSS Modules with Material-UI components
- **API Integration**: Axios with rate limiting
- **State Management**: React Hooks and Context API

## Getting Started

### Prerequisites

- Node.js (v14.0.0 or later)
- npm (v6.0.0 or later)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-organization/filipino-root-word-explorer.git
   cd filipino-root-word-explorer
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add:
   ```
   REACT_APP_API_BASE_URL=https://api.example.com/v1
   ```

4. Start the development server:
   ```
   npm start
   ```

Visit `http://localhost:3000` to explore the application.

## Architecture

The application follows a modular architecture with key components:

- `WordExplorer`: The main component orchestrating the exploration experience.
- `WordGraph`: Renders the interactive network visualization using D3.js.
- `WordDetails`: Displays comprehensive information about selected words.

State management is handled through React Hooks and Context, ensuring efficient updates and a smooth user experience.

## API Integration

The application interfaces with a custom API for word data. Requests are managed in `src/api/wordApi.ts`, implementing caching and rate limiting for optimal performance.

## Styling Philosophy

The UI is crafted with a focus on clarity and aesthetics, utilizing a carefully chosen color palette that adapts seamlessly between light and dark modes. CSS Modules ensure style encapsulation and maintainability.

## Contributing

We welcome contributions that enhance the Filipino Root Word Explorer. Please review our [Contribution Guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For detailed documentation, please visit our [Wiki](https://github.com/your-organization/filipino-root-word-explorer/wiki).