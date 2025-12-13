/**
 * Express Server for SOP Automation Frontend
 */
const express = require('express');
const path = require('path');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8001';

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));

// Serve static files
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Proxy endpoint for SOP generation
app.post('/api/generate-sop', async (req, res) => {
    try {
        // This will be handled by frontend directly, but keeping for proxy option
        const response = await axios.post(`${BACKEND_URL}/api/generate-sop`, req.body, {
            headers: req.headers,
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });
        res.json(response.data);
    } catch (error) {
        console.error('Proxy error:', error.message);
        res.status(error.response?.status || 500).json({
            error: error.message,
            details: error.response?.data
        });
    }
});

// Proxy for PDF download
app.get('/api/download-pdf/:filename', async (req, res) => {
    try {
        const response = await axios.get(`${BACKEND_URL}/api/download-pdf/${req.params.filename}`, {
            responseType: 'stream'
        });
        res.setHeader('Content-Type', 'application/pdf');
        res.setHeader('Content-Disposition', `attachment; filename="${req.params.filename}"`);
        response.data.pipe(res);
    } catch (error) {
        console.error('PDF download error:', error.message);
        res.status(error.response?.status || 500).json({ error: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`Frontend server running on http://localhost:${PORT}`);
    console.log(`Backend URL: ${BACKEND_URL}`);
});

