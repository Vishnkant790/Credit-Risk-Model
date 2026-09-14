import axios from 'axios'

const API_URL = 'http://localhost:8000';

export const getPrediction = async(customerData) => {
    const response = await axios.post('${API_URL}/predict', customerData);
    return response.data;

};