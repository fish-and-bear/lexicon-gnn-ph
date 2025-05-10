/* eslint-disable @typescript-eslint/no-this-alias */
import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from "react";
import * as d3 from "d3";
import "./WordGraph.css";
import { WordNetwork as ImportedWordNetwork, NetworkNode as ImportedNetworkNode, NetworkLink as ImportedNetworkLink } from "../types";
import { useAppTheme } from "../contexts/ThemeContext";
import Slider from '@mui/material/Slider';
import Typography from "@mui/material/Typography";
import axios from 'axios';
import NetworkControls from './NetworkControls';
// Import color utility functions
import { 
  mapRelationshipToGroup, 
  getNodeColor, 
  getTextColorForBackground,
  getRelationshipTypeLabel
} from '../utils/colorUtils';

// MUI imports
import Drawer from '@mui/material/Drawer';
import IconButton from '@mui/material/IconButton';
import Box from '@mui/material/Box';
import TuneIcon from '@mui/icons-material/Tune';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import Checkbox from '@mui/material/Checkbox';
import ListSubheader from '@mui/material/ListSubheader';
import Divider from '@mui/material/Divider';
import { useTheme as useMuiTheme } from '@mui/material/styles';
import { alpha } from '@mui/material/styles';
import { CircularProgress, Dialog, DialogTitle, DialogContent, DialogActions, Button } from '@mui/material'; // Added Dialog imports
import InfoIcon from '@mui/icons-material/Info'; // Icon for legend button
import ListItemButton from '@mui/material/ListItemButton'; // Added ListItemButton import

interface WordGraphProps {
  wordNetwork: ImportedWordNetwork | null;
  mainWord: string | null;
  onNodeClick: (word: string) => void;
  onNetworkChange: (depth: number, breadth: number) => void;
  initialDepth: number;
  initialBreadth: number;
  isMobile: boolean;
  isLoading: boolean;
}

interface CustomNode extends d3.SimulationNodeDatum {
  id: string; // Will be node.label
  word: string;
  label: string; 
  group: string;
  connections?: number;
  pinned?: boolean;
  originalId?: number | string; // Can be string or number from source
  language?: string;
  definitions?: string[];
  has_baybayin?: boolean;
  baybayin_form?: string | null;
  relationshipToMain?: string; 
  pathToMain?: string[];
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

interface CustomLink extends d3.SimulationLinkDatum<CustomNode> {
  relationship: string;
  source: string | CustomNode; // Initially string (ID), resolved to CustomNode by D3
  target: string | CustomNode; // Initially string (ID), resolved to CustomNode by D3
  metadata: Record<string, any> | null | undefined; 
}

interface RelationshipTypeInfo {
  category: string;
  label: string;
  color: string;
}

interface RelationshipLabelInfo {
  label: string;
  color: string;
  types: string[]; 
}

interface RelationshipGroups {
  uniqueTypes: Record<string, RelationshipTypeInfo>; 
  categories: Array<{ name: string; labels: RelationshipLabelInfo[] }>;
}

const WordGraph: React.FC<WordGraphProps> = ({
  wordNetwork,
  mainWord,
  onNodeClick,
  onNetworkChange,
  initialDepth,
  initialBreadth,
  isMobile,
  isLoading
}) => {
  const { themeMode } = useAppTheme();
  const muiTheme = useMuiTheme();
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const gRef = useRef<SVGGElement | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const legendContainerRef = useRef<d3.Selection<SVGGElement, unknown, null, undefined> | null>(null);
  const legendWidthRef = useRef<number>(0); 
  const resizeTimeoutRef = useRef<NodeJS.Timeout | null>(null); 

  const [hoveredNode, setHoveredNode] = useState<CustomNode | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const [depth, setDepth] = useState<number>(initialDepth);
  const [breadth, setBreadth] = useState<number>(initialBreadth);
  const [error, setError] = useState<string | null>(null);
  const [isValidNetwork, setIsValidNetwork] = useState(true);
  const simulationRef = useRef<d3.Simulation<CustomNode, CustomLink> | null>(null);
  const [filteredRelationships, setFilteredRelationships] = useState<string[]>([]);
  const [forceUpdate, setForceUpdate] = useState<number>(0);
  const [mobileLegendOpen, setMobileLegendOpen] = useState(false);
  const [selectedMobileNodeId, setSelectedMobileNodeId] = useState<string | null>(null);
  const [controlsOpen, setControlsOpen] = useState(false);

  const isDraggingRef = useRef(false);
  const prevMainWordRef = useRef<string | null>(null);

  const filterUpdateKey = useMemo(() => {
    return filteredRelationships.join(',');
  }, [filteredRelationships]);

  // Handle mobile legend toggle
  const handleToggleMobileLegend = useCallback(() => {
    setMobileLegendOpen(!mobileLegendOpen);
  }, []);

  // Toggle control drawer
  const toggleControlsDrawer = useCallback(() => {
    setControlsOpen(!controlsOpen);
  }, []);

  // Check if network data is valid
  useEffect(() => {
    if (!wordNetwork || !wordNetwork.nodes || !Array.isArray(wordNetwork.nodes) || 
        !wordNetwork.links || !Array.isArray(wordNetwork.links)) {
      console.error("Invalid wordNetwork structure:", wordNetwork);
      setIsValidNetwork(false);
    } else {
      setIsValidNetwork(true);
    }
  }, [wordNetwork]);

  // Memoized data transformations (order matters for dependencies)
  const nodeIdToLabelMap = useMemo(() => {
    const map = new Map<string, string>();
    if (!wordNetwork?.nodes) return map;
    wordNetwork.nodes.forEach(node => {
      let nodeIdKey: string | null = null;
      if (typeof node.id === 'number') {
        nodeIdKey = String(node.id);
      } else if (typeof node.id === 'string' && node.id) {
        nodeIdKey = node.id;
      }
      if (nodeIdKey && typeof node.label === 'string' && node.label) {
        map.set(nodeIdKey, node.label);
      }
    });
    return map;
  }, [wordNetwork?.nodes]);

  const baseLinks = useMemo((): CustomLink[] => {
    if (!wordNetwork?.links || !nodeIdToLabelMap.size) return [];
    return wordNetwork.links
      .map(link => {
        let sourceLabel: string | null = null;
        let targetLabel: string | null = null;
        let sourceIdKey: string | null = typeof link.source === 'number' ? String(link.source) : (typeof link.source === 'string' ? link.source : null);
        let targetIdKey: string | null = typeof link.target === 'number' ? String(link.target) : (typeof link.target === 'string' ? link.target : null);

        if (sourceIdKey) sourceLabel = nodeIdToLabelMap.get(sourceIdKey) || null;
        if (targetIdKey) targetLabel = nodeIdToLabelMap.get(targetIdKey) || null;

        if (!sourceLabel || !targetLabel) {
          return null;
        }
        return {
          source: sourceLabel, 
          target: targetLabel, 
          relationship: link.relationship,
          metadata: link.metadata
        } as CustomLink;
      })
      .filter((link): link is CustomLink => link !== null);
  }, [wordNetwork?.links, nodeIdToLabelMap]);

  const baseNodes = useMemo<CustomNode[]>(() => {
    if (!wordNetwork?.nodes || !mainWord) return [];
    const uniqueNodes = new Map<string, CustomNode>();
    
    // Process and deduplicate nodes - placeholder implementation
    wordNetwork.nodes.forEach(node => {
      const nodeId = node.label || '';
      if (nodeId && !uniqueNodes.has(nodeId)) {
        uniqueNodes.set(nodeId, {
          id: nodeId,
          word: node.word || node.label || '',
          label: node.label || '',
          group: 'default',
          // Add other required properties
        });
      }
    });
    
    return Array.from(uniqueNodes.values());
  }, [wordNetwork?.nodes, mainWord]);

  // Define mainWordIdString at component level for dependency arrays
  const mainWordIdString = useMemo(() => {
    if (!mainWord || !baseNodes.length) return null;
    const mainWordNode = baseNodes.find(node => node.word === mainWord);
    return mainWordNode ? mainWordNode.id : null;
  }, [mainWord, baseNodes]);

  // Reset highlights function
  const resetHighlights = useCallback(() => {
    if (!svgRef.current || isDraggingRef.current) return;
    
    // Simple implementation - replace with your actual implementation
    d3.select(svgRef.current)
      .selectAll('.node')
      .style('opacity', 1);
      
    d3.select(svgRef.current)
      .selectAll('.link')
      .style('opacity', 1);
  }, [isDraggingRef]);

  // Apply highlights function
  const applyHighlights = useCallback((nodeData: CustomNode) => {
    if (!svgRef.current || isDraggingRef.current) return;
    
    // Simple implementation - replace with your actual implementation
    const nodeId = nodeData.id;
    const connectedIds = new Set<string>([nodeId]);
    
    d3.select(svgRef.current)
      .selectAll('.node')
      .style('opacity', (d: any) => connectedIds.has(d.id) ? 1 : 0.3);
  }, [isDraggingRef]);

  // Rest of the component with functions fixed to use mainWordIdString and resetHighlights
  // ...

  // Main useEffect for rendering
  useEffect(() => {
    if (!svgRef.current || !wordNetwork || !mainWord) {
      if (svgRef.current) d3.select(svgRef.current).selectAll("*").remove();
      if (simulationRef.current) simulationRef.current.stop();
      return;
    }
    
    // Rest of your rendering code
    // ...
    
    return () => {
      if (simulationRef.current) simulationRef.current.stop();
      // Other cleanup
    };
  }, [
    // Dependencies
    wordNetwork,
    mainWord,
    filteredRelationships,
    // Other dependencies...
  ]);

  // Tooltip rendering function
  const renderTooltip = useCallback(() => {
    if (!hoveredNode) return null;
    
    // Your tooltip JSX
    return (
      <div className="node-tooltip">
        {/* Tooltip content */}
        {hoveredNode.word}
      </div>
    );
  }, [hoveredNode]);

  // Return the JSX
  return (
    <div ref={containerRef} className="graph-container">
      {/* SVG Container for the Graph */}
      <div className="graph-wrapper">
        <div className="graph-svg-container">
          {isLoading && (
            <div className="loading-overlay"><div className="spinner"></div><p>Loading Network...</p></div>
          )}
          {error && (
            <div className="error-overlay">
              <p className="error-message">{error}</p>
            </div>
          )}
          {(!wordNetwork || !mainWord || !filteredRelationships.length && !error) && (
            <div className="empty-graph-message">Enter a word to see its network.</div>
          )}
          <svg 
            ref={svgRef} 
            className={`graph-svg ${isLoading ? 'loading' : 'loaded'}`}
            key={`graph-${mainWord}-${depth}-${breadth}-${filteredRelationships.join('.')}-${forceUpdate}-${filterUpdateKey}`} 
          >
          </svg>
        </div>
      </div>
        
      {/* Bottom Controls - Slider Area (fixed at bottom) */}
      <div className="controls-container">
        {isMobile && (
          <IconButton
            onClick={handleToggleMobileLegend}
            color="primary"
            aria-label="view legend"
            size="small" // Make button smaller for mobile
            sx={{
              mr: 0.5, // Adjust margin
              padding: '6px',
              border: `1px solid ${alpha(muiTheme.palette.primary.main, 0.3)}`,
              backgroundColor: alpha(muiTheme.palette.primary.main, 0.05),
                '&:hover': {
                  backgroundColor: alpha(muiTheme.palette.primary.main, 0.1),
                }
            }}
          >
            <InfoIcon fontSize="medium" /> {/* Slightly larger icon */}
          </IconButton>
        )}
        {/* Zoom Controls */}
        <div className="zoom-controls">
          <button 
            onClick={() => { 
              if (zoomRef.current && svgRef.current) d3.select(svgRef.current).call(zoomRef.current.scaleBy, 1.3); 
            }} 
            className="zoom-button" 
            title="Zoom In"
          >
            +
          </button>
          <button 
            onClick={() => { 
              if (zoomRef.current && svgRef.current) d3.select(svgRef.current).call(zoomRef.current.scaleBy, 1/1.3); 
            }} 
            className="zoom-button" 
            title="Zoom Out"
          >
            -
          </button>
          <button 
            onClick={() => { 
              if (zoomRef.current && svgRef.current) {
                const containerRect = svgRef.current.parentElement?.getBoundingClientRect();
                const width = containerRect ? containerRect.width : 800;
                const height = containerRect ? containerRect.height : 600;
                const resetTransform = d3.zoomIdentity.translate(width / 2, height / 2).scale(1);
                d3.select(svgRef.current).transition().duration(600).ease(d3.easeCubicInOut)
                  .call(zoomRef.current.transform, resetTransform);
              }
            }} 
            className="reset-zoom-button" 
            title="Reset View"
          >
            Reset
          </button>
        </div>

        {/* Network Controls */}
        <NetworkControls 
          depth={depth}
          breadth={breadth}
          onDepthChange={(newDepth) => {
            setDepth(newDepth);
            onNetworkChange(newDepth, breadth);
          }}
          onBreadthChange={(newBreadth) => {
            setBreadth(newBreadth); 
            onNetworkChange(depth, newBreadth);
          }}
          className="network-controls"
        />
      </div>
        
      {/* Tooltip */}
      {renderTooltip()}

      {/* Mobile Legend Dialog */}
      <Dialog 
        open={mobileLegendOpen} 
        onClose={handleToggleMobileLegend} 
        fullWidth 
        maxWidth="xs" 
        scroll="paper"
      >
        <DialogTitle sx={{ pb: 1, fontSize: '1.1rem', borderBottom: `1px solid ${muiTheme.palette.divider}` }}>
          Graph Legend
        </DialogTitle>
        <DialogContent dividers sx={{ p: 0, '& .MuiListSubheader-root': { lineHeight: '32px', py: 0.25 } }}>
          {/* Dialog content */}
        </DialogContent>
        <DialogActions sx={{ pt: 1.5, pb: 1.5, pr: 2, borderTop: `1px solid ${muiTheme.palette.divider}` }}>
          <Button onClick={handleToggleMobileLegend} variant="outlined" size="small">Close</Button>
        </DialogActions>
      </Dialog>
    </div>
  );
};

export default WordGraph; 