/* eslint-disable no-restricted-globals */

import axios from 'axios';

self.onmessage = async function (event) {
  const { url, params } = event.data;
  try {
    const response = await axios.get(url, { params });
    self.postMessage({ data: response.data });
  } catch (error) {
    self.postMessage({ error: error.message });
  }
};
