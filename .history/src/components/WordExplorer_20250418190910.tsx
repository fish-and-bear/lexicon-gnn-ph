import React from "react";
// import WordGraph from "./WordGraph";
// import WordDetails from "./WordDetails";
// import { useTheme } from "../contexts/ThemeContext";
import "./WordExplorer.css";
// import { WordNetwork, WordInfo, SearchResult, SearchOptions, EtymologyTree, Statistics, Definition, SearchWordResult, Relation } from "../types";
// import unidecode from "unidecode";
// import { 
//   fetchWordNetwork, 
//   fetchWordDetails, 
//   searchWords, 
//   getRandomWord,
//   testApiConnection,
//   resetCircuitBreaker,
//   getPartsOfSpeech,
//   getStatistics,
//   getBaybayinWords,
//   getAffixes,
//   getRelations,
//   getAllWords,
//   getEtymologyTree,
//   fetchWordRelations
// } from "../api/wordApi";
// import axios from 'axios';
// import DOMPurify from 'dompurify';
// import { debounce } from "lodash";
// import Tabs from '@mui/material/Tabs';
// import Tab from '@mui/material/Tab';
// import Box from '@mui/material/Box';
// import CircularProgress from '@mui/material/CircularProgress';
// import { styled } from '@mui/material/styles';
// import NetworkControls from './NetworkControls';
// import { Typography, Button } from "@mui/material";

// // Import Resizable Panels components
// import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
// import useMediaQuery from '@mui/material/useMediaQuery'; // Reuse useMediaQuery

// // Add a shuffle function utility near the top, outside the component
// function shuffleArray(array: any[]) {
//   for (let i = array.length - 1; i > 0; i--) {
//     const j = Math.floor(Math.random() * (i + 1));
//     [array[i], array[j]] = [array[j], array[i]];
//   }
//   return array;
// }

// // Custom TabPanel component for displaying tab content
// interface TabPanelProps {
//   children?: React.ReactNode;
//   index: number;
//   value: number;
// }

// const TabPanel = (props: TabPanelProps) => {
//   const { children, value, index, ...other } = props;

//   return (
//     <div
//       role="tabpanel"
//       hidden={value !== index}
//       id={`simple-tabpanel-${index}`}
//       aria-labelledby={`simple-tab-${index}`}
//       {...other}
//       style={{
//         display: value === index ? 'block' : 'none',
//         height: 'calc(100vh - 160px)',
//         overflow: 'hidden'
//       }}
//     >
//       {value === index && children}
//     </div>
//   );
// };

const WordExplorer: React.FC = () => {
  // Comment out all state, effects, callbacks, refs, etc.
  // const [searchTerm, setSearchTerm] = useState<string>("");
  // ... all other state variables ...
  // const { theme, toggleTheme } = useTheme();
  // const detailsContainerRef = useRef<HTMLDivElement>(null);
  // ... all useEffect hooks ...
  // ... all useCallback hooks (fetchers, handlers) ...
  // ... all render functions ...

  return (
    <div className={`word-explorer`}>
      <header className="header-content">
        <h1>Filipino Root Word Explorer - Simplified Test</h1>
      </header>
      <main>
        <p>If you see this, the basic build and render pipeline is working.</p>
      </main>
      <footer className="footer">
        © {new Date().getFullYear()} Filipino Root Word Explorer. All Rights Reserved.
      </footer>
    </div>
  );
};

export default WordExplorer;
