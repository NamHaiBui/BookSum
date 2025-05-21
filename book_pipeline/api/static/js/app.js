// AudioBookSum Application JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const processingSection = document.getElementById('processing-section');
    const resultSection = document.getElementById('result-section');
    const errorMessage = document.getElementById('error-message');
    const statusMessage = document.getElementById('status-message');
    const progressBar = document.getElementById('progress-bar');
    const summaryPreview = document.getElementById('summary-preview');
    const downloadHighlighted = document.getElementById('download-highlighted');
    const downloadSummary = document.getElementById('download-summary');
    const downloadEvidence = document.getElementById('download-evidence');

    // Handle form submission
    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        // Reset UI
        errorMessage.textContent = '';
        uploadForm.classList.add('was-validated');
        
        if (uploadForm.checkValidity()) {
            // Get form data
            const formData = new FormData(uploadForm);
            
            // Show processing section
            processingSection.style.display = 'block';
            resultSection.style.display = 'none';
            
            // Start the upload and processing
            processFile(formData);
        }
    });

    // Process the file and handle API interaction
    function processFile(formData) {
        // Mock progress updates (can be replaced with WebSocket for real-time updates)
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 5;
            if (progress <= 90) {
                updateProgress(progress);
            }
        }, 2000);

        // Make API request
        fetch('/api/process-pdf', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            clearInterval(progressInterval);
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            updateProgress(100);
            return response.json();
        })
        .then(data => {
            // Processing complete
            setTimeout(() => {
                showResults(data);
            }, 1000);
        })
        .catch(error => {
            clearInterval(progressInterval);
            processingSection.style.display = 'none';
            errorMessage.textContent = `Error: ${error.message}`;
        });
    }

    // Update progress bar and status message
    function updateProgress(percent) {
        progressBar.style.width = `${percent}%`;
        progressBar.textContent = `${percent}%`;
        progressBar.setAttribute('aria-valuenow', percent);
        
        if (percent < 25) {
            statusMessage.textContent = 'Uploading PDF...';
        } else if (percent < 50) {
            statusMessage.textContent = 'Extracting text from PDF...';
        } else if (percent < 75) {
            statusMessage.textContent = 'Processing content...';
        } else if (percent < 100) {
            statusMessage.textContent = 'Generating summary and highlights...';
        } else {
            statusMessage.textContent = 'Processing complete!';
        }
    }

    // Show the processing results
    function showResults(data) {
        processingSection.style.display = 'none';
        resultSection.style.display = 'block';
        
        // Update summary preview (truncate if too long)
        const summary = data.summary || 'Summary not available.';
        summaryPreview.textContent = summary.length > 500 ? 
            summary.substring(0, 500) + '...' : summary;
        
        // Set download links
        if (data.highlighted_pdf_url) {
            downloadHighlighted.href = data.highlighted_pdf_url;
            downloadHighlighted.classList.remove('disabled');
        } else {
            downloadHighlighted.classList.add('disabled');
        }
        
        if (data.summary_url) {
            downloadSummary.href = data.summary_url;
            downloadSummary.classList.remove('disabled');
        } else {
            downloadSummary.classList.add('disabled');
        }
        
        if (data.evidence_url) {
            downloadEvidence.href = data.evidence_url;
            downloadEvidence.classList.remove('disabled');
        } else {
            downloadEvidence.classList.add('disabled');
        }
    }

    // Add file name display when a file is selected
    const fileInput = document.getElementById('file');
    fileInput.addEventListener('change', function() {
        const fileName = this.files[0]?.name;
        const fileLabel = this.nextElementSibling;
        if (fileName) {
            fileLabel.textContent = fileName;
        }
    });
});

// WebSocket connection for real-time updates (can be implemented later)
function setupWebSocket() {
    if (!window.WebSocket) {
        console.error('WebSocket not supported');
        return;
    }
    
    const ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onopen = function() {
        console.log('WebSocket connection established');
    };
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        console.log('WebSocket message received:', data);
        
        if (data.type === 'progress') {
            updateProgress(data.percent);
        } else if (data.type === 'status') {
            statusMessage.textContent = data.message;
        } else if (data.type === 'complete') {
            showResults(data.results);
        } else if (data.type === 'error') {
            errorMessage.textContent = data.message;
        }
    };
    
    ws.onclose = function() {
        console.log('WebSocket connection closed');
    };
    
    return ws;
}