import axios from 'axios';

const makeRequest = async (url: string): Promise<any> => {
  try {
    const data = await axios.get(url);
    return data.data
  } catch (e) {
    if (e.response) {
      throw new Error(e.response.data.message);
    } else {
      throw new Error('Please refresh page & try again.');
    }
  }
}

const getSampleImage = async (sample_id: number): Promise<any> => {
  return await makeRequest(`/image/${sample_id}`)
};

const getComparison = async (layer1: string, layer2: string): Promise<any> => {
  return await makeRequest(`/compare/${layer1}/${layer2}`)
};

export { makeRequest, getSampleImage, getComparison }
