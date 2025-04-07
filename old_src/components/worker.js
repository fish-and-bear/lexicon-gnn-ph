/* eslint-disable no-restricted-globals */

import * as d3 from 'd3';

self.onmessage = function (event) {
  const { nodes, links, width, height } = event.data;

  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(100))
    .force('charge', d3.forceManyBody().strength(-300))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collide', d3.forceCollide().radius(30).strength(0.7))
    .on('tick', () => {
      self.postMessage({ nodes, links });
    });
};
