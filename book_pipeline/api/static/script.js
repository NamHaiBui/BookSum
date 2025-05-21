// Global variables
let jobsData = [];
let activeSection = 'home';
let statusUpdateInterval;
let jobsUpdateInterval;
let currentJobModal = null;
let metricsCharts = {};

// Bootstrap modal instance
let jobDetailModal;

// DOM ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap components
    jobDetailModal = new bootstrap.Modal(document.getElementById('job-detail-modal'));
    
    // Set up range input displays
    document.getElementById('chunk-size').addEventListener('input', function() {
        document.getElementById('chunk-size-value').textContent = this.value;
    });
    
    document.getElementById('chunk-overlap').addEventListener('input', function() {
        document.getElementById('chunk-overlap-value').textContent = this.value;
    });
    
    // Set up navigation
    document.getElementById('nav-home').addEventListener('click', function(e) {
        e.preventDefault();
        showSection('home');
    });
    
    document.getElementById('nav-jobs').addEventListener('click', function(e) {
        e.preventDefault();
        showSection('jobs');
    });
    
    document.getElementById('nav-metrics').addEventListener('click', function(e) {
        e.preventDefault();
        showSection('metrics');
    });
    
    // Set up form submission
    document.getElementById('upload-form').addEventListener('submit', function(e) {
        e.preventDefault();
        uploadAndProcess();
    });
    
    // Initialize the app
    init();
});

// Initialize the application
function init() {
    // Load initial data
    fetchSystemStatus();
    fetchJobs();
    
    // Set up intervals for auto-updates
    statusUpdateInterval = setInterval(fetchSystemStatus, 10000);
    jobsUpdateInterval = setInterval(fetchJobs, 5000);
}

// Show the selected section
function showSection(section) {
    // Hide all sections
    document.getElementById('home-section').style.display = 'none';
    document.getElementById('jobs-section').style.display = 'none';
    document.getElementById('metrics-section').style.display = 'none';
    
    // Show the selected section
    document.getElementById(`${section}-section`).style.display = 'block';
    
    // Update active nav link
    document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
    document.getElementById(`nav-${section}`).classList.add('active');
    
    // Update active section
    activeSection = section;
    
    // Initialize metrics charts if showing metrics section
    if (section === 'metrics') {
        initMetricsCharts();
    }
}

// Upload and process a PDF
function uploadAndProcess() {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('pdf-file');
    
    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select a PDF file to upload');
        return;
    }
    
    // Create FormData to send the file and options
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    // Add form options
    formData.append('use_mistral', document.getElementById('use-mistral').checked);
    formData.append('use_chunking_v1', document.getElementById('use-chunking-v1').checked);
    formData.append('chunk_size', document.getElementById('chunk-size').value);
    formData.append('chunk_overlap', document.getElementById('chunk-overlap').value);
    formData.append('save_intermediate', document.getElementById('save-intermediate').checked);
    
    // Update the submit button to indicate processing
    const submitButton = form.querySelector('button[type="submit"]');
    const originalButtonText = submitButton.textContent;
    submitButton.disabled = true;
    submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Uploading...';
    
    // Send the request
    fetch('/api/process', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Reset the form
        form.reset();
        document.getElementById('chunk-size-value').textContent = '50000';
        document.getElementById('chunk-overlap-value').textContent = '100';
        
        // Show success message
        alert(`PDF uploaded successfully! Job ID: ${data.id}`);
        
        // Fetch updated jobs
        fetchJobs();
        
        // Optionally switch to jobs view
        showSection('jobs');
    })
    .catch(error => {
        console.error('Error uploading PDF:', error);
        alert('Error uploading PDF: ' + error.message);
    })
    .finally(() => {
        // Reset button state
        submitButton.disabled = false;
        submitButton.textContent = originalButtonText;
    });
}

// Fetch system status
function fetchSystemStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            updateSystemStatus(data);
        })
        .catch(error => {
            console.error('Error fetching system status:', error);
            document.getElementById('status-container').innerHTML = 
                '<div class="alert alert-danger">Error fetching system status</div>';
        });
}

// Update system status display
function updateSystemStatus(data) {
    const statusHtml = `
        <div class="mb-3">
            <p><strong>API Status:</strong> <span class="badge bg-success">${data.status}</span></p>
            <p><strong>Version:</strong> ${data.version}</p>
        </div>
        <div class="mb-3">
            <h6>Jobs</h6>
            <div class="row">
                <div class="col-md-6">
                    <p><strong>Active:</strong> ${data.active_jobs}</p>
                    <p><strong>Queued:</strong> ${data.queued_jobs}</p>
                </div>
                <div class="col-md-6">
                    <p><strong>Completed:</strong> ${data.completed_jobs}</p>
                    <p><strong>Failed:</strong> ${data.failed_jobs}</p>
                </div>
            </div>
        </div>
    `;
    
    document.getElementById('status-container').innerHTML = statusHtml;
}

// Fetch jobs
function fetchJobs() {
    fetch('/api/jobs')
        .then(response => response.json())
        .then(data => {
            jobsData = data.jobs;
            updateJobsTable();
            updateLatestJobs();
        })
        .catch(error => {
            console.error('Error fetching jobs:', error);
        });
}

// Update the jobs table
function updateJobsTable() {
    if (!jobsData || jobsData.length === 0) {
        document.getElementById('jobs-table-body').innerHTML = 
            '<tr><td colspan="7" class="text-center">No jobs found</td></tr>';
        return;
    }
    
    // Sort jobs by created_at (newest first)
    jobsData.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    
    let tableHtml = '';
    
    for (const job of jobsData) {
        const createdDate = new Date(job.created_at).toLocaleString();
        const completedDate = job.completed_at ? new Date(job.completed_at).toLocaleString() : '-';
        
        let progressHtml;
        if (job.status === 'processing') {
            progressHtml = `
                <div class="progress">
                    <div class="progress-bar progress-bar-striped progress-bar-animated" 
                         role="progressbar" style="width: ${job.progress}%" 
                         aria-valuenow="${job.progress}" aria-valuemin="0" aria-valuemax="100">
                        ${job.progress.toFixed(0)}%
                    </div>
                </div>
            `;
        } else if (job.status === 'completed') {
            progressHtml = `
                <div class="progress">
                    <div class="progress-bar bg-success" 
                         role="progressbar" style="width: 100%" 
                         aria-valuenow="100" aria-valuemin="0" aria-valuemax="100">
                        100%
                    </div>
                </div>
            `;
        } else if (job.status === 'failed') {
            progressHtml = `
                <div class="progress">
                    <div class="progress-bar bg-danger" 
                         role="progressbar" style="width: 100%" 
                         aria-valuenow="100" aria-valuemin="0" aria-valuemax="100">
                        Failed
                    </div>
                </div>
            `;
        } else {
            progressHtml = `
                <div class="progress">
                    <div class="progress-bar bg-secondary" 
                         role="progressbar" style="width: 0%" 
                         aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
                        Queued
                    </div>
                </div>
            `;
        }
        
        let actionsHtml = `
            <button class="btn btn-sm btn-info view-job" data-job-id="${job.id}">
                View
            </button>
        `;
        
        if (job.status !== 'processing') {
            actionsHtml += `
                <button class="btn btn-sm btn-danger delete-job ms-1" data-job-id="${job.id}">
                    Delete
                </button>
            `;
        }
        
        tableHtml += `
            <tr>
                <td>${job.id.substring(0, 8)}...</td>
                <td>${job.pdf_name}</td>
                <td>
                    <span class="badge status-${job.status}">
                        ${job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                    </span>
                </td>
                <td>${progressHtml}</td>
                <td>${createdDate}</td>
                <td>${completedDate}</td>
                <td>${actionsHtml}</td>
            </tr>
        `;
    }
    
    document.getElementById('jobs-table-body').innerHTML = tableHtml;
    
    // Add event listeners for view and delete buttons
    document.querySelectorAll('.view-job').forEach(button => {
        button.addEventListener('click', function() {
            const jobId = this.getAttribute('data-job-id');
            viewJobDetails(jobId);
        });
    });
    
    document.querySelectorAll('.delete-job').forEach(button => {
        button.addEventListener('click', function() {
            const jobId = this.getAttribute('data-job-id');
            deleteJob(jobId);
        });
    });
}

// Update the latest jobs display on the home page
function updateLatestJobs() {
    if (!jobsData || jobsData.length === 0) {
        document.getElementById('latest-jobs').innerHTML = 
            '<p>No jobs found</p>';
        return;
    }
    
    // Sort jobs by created_at (newest first)
    const sortedJobs = [...jobsData]
        .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
        .slice(0, 5); // Get only the 5 most recent jobs
    
    let jobsHtml = '<div class="list-group">';
    
    for (const job of sortedJobs) {
        const createdDate = new Date(job.created_at).toLocaleString();
        
        jobsHtml += `
            <a href="#" class="list-group-item list-group-item-action job-list-item" data-job-id="${job.id}">
                <div class="d-flex w-100 justify-content-between">
                    <h6 class="mb-1">${job.pdf_name}</h6>
                    <small class="text-muted">${createdDate}</small>
                </div>
                <p class="mb-1">
                    <span class="badge status-${job.status}">
                        ${job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                    </span>
                    ${job.status === 'processing' ? 
                        `<span class="ms-2">${job.progress.toFixed(0)}%</span>` : 
                        ''}
                </p>
            </a>
        `;
    }
    
    jobsHtml += '</div>';
    
    document.getElementById('latest-jobs').innerHTML = jobsHtml;
    
    // Add event listeners for job list items
    document.querySelectorAll('.job-list-item').forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            const jobId = this.getAttribute('data-job-id');
            viewJobDetails(jobId);
        });
    });
}

// View job details
function viewJobDetails(jobId) {
    // Find the job in jobsData
    const job = jobsData.find(j => j.id === jobId);
    
    if (!job) {
        console.error('Job not found:', jobId);
        return;
    }
    
    // Set modal title
    document.querySelector('#job-detail-modal .modal-title').textContent = 
        `Job Details: ${job.pdf_name}`;
    
    // Initialize content placeholders
    document.getElementById('job-summary').innerHTML = 
        '<div class="text-center"><div class="loading-spinner"></div> Loading summary...</div>';
    document.getElementById('job-evidence').innerHTML = 
        '<div class="text-center"><div class="loading-spinner"></div> Loading evidence points...</div>';
    document.getElementById('job-files').innerHTML = 
        '<div class="text-center"><div class="loading-spinner"></div> Loading files...</div>';
    
    // Show the modal
    jobDetailModal.show();
    
    // Set the current job modal
    currentJobModal = jobId;
    
    // If the job is not completed, don't try to fetch results
    if (job.status !== 'completed') {
        document.getElementById('job-summary').innerHTML = 
            `<div class="alert alert-info">Job is ${job.status}. Results will be available when the job completes.</div>`;
        document.getElementById('job-evidence').innerHTML = 
            `<div class="alert alert-info">Job is ${job.status}. Results will be available when the job completes.</div>`;
        
        // Still show output files if any
        updateJobFiles(job);
        return;
    }
    
    // Fetch job results
    fetch(`/api/jobs/${jobId}/result`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            updateJobDetails(data);
        })
        .catch(error => {
            console.error('Error fetching job results:', error);
            document.getElementById('job-summary').innerHTML = 
                '<div class="alert alert-danger">Error fetching job results</div>';
            document.getElementById('job-evidence').innerHTML = 
                '<div class="alert alert-danger">Error fetching job results</div>';
        });
    
    // Update output files
    updateJobFiles(job);
}

// Update job details in the modal
function updateJobDetails(data) {
    // Make sure this is still the current modal
    if (currentJobModal !== data.id) {
        return;
    }
    
    // Update summary
    if (data.summary) {
        document.getElementById('job-summary').innerHTML = data.summary;
    } else {
        document.getElementById('job-summary').innerHTML = 
            '<div class="alert alert-warning">No summary available</div>';
    }
    
    // Update evidence
    if (data.evidence && data.evidence.length > 0) {
        let evidenceHtml = '<ol>';
        for (const ev of data.evidence) {
            evidenceHtml += `<li>${ev.text}</li>`;
        }
        evidenceHtml += '</ol>';
        document.getElementById('job-evidence').innerHTML = evidenceHtml;
    } else {
        document.getElementById('job-evidence').innerHTML = 
            '<div class="alert alert-warning">No evidence points available</div>';
    }
}

// Update output files display
function updateJobFiles(job) {
    if (!job.output_files || job.output_files.length === 0) {
        document.getElementById('job-files').innerHTML = 
            '<div class="alert alert-warning">No output files available</div>';
        return;
    }
    
    let filesHtml = '';
    
    for (const file of job.output_files) {
        const fileName = file.split('/').pop();
        const fileType = fileName.split('.').pop().toLowerCase();
        
        let fileIcon = '';
        if (fileType === 'pdf') {
            fileIcon = '<i class="bi bi-file-pdf"></i>';
        } else if (fileType === 'txt') {
            fileIcon = '<i class="bi bi-file-text"></i>';
        } else if (fileType === 'json') {
            fileIcon = '<i class="bi bi-file-code"></i>';
        } else {
            fileIcon = '<i class="bi bi-file"></i>';
        }
        
        filesHtml += `
            <a href="/api/jobs/${job.id}/output/${fileName}" 
               target="_blank" class="file-link">
                ${fileIcon} ${fileName}
            </a>
        `;
    }
    
    document.getElementById('job-files').innerHTML = filesHtml;
}

// Delete a job
function deleteJob(jobId) {
    if (!confirm('Are you sure you want to delete this job? This will remove all job data and files.')) {
        return;
    }
    
    fetch(`/api/jobs/${jobId}`, {
        method: 'DELETE'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Fetch updated jobs
        fetchJobs();
        alert('Job deleted successfully');
    })
    .catch(error => {
        console.error('Error deleting job:', error);
        alert('Error deleting job: ' + error.message);
    });
}

// Initialize metrics charts
function initMetricsCharts() {
    // Processing Time Chart
    if (!metricsCharts.processingTime) {
        const processingTimeCtx = document.getElementById('processing-time-chart').getContext('2d');
        metricsCharts.processingTime = new Chart(processingTimeCtx, {
            type: 'bar',
            data: {
                labels: ['Extraction', 'Transformation', 'Loading', 'Total'],
                datasets: [{
                    label: 'Average Time (seconds)',
                    data: [3.2, 2.1, 5.4, 10.7],
                    backgroundColor: [
                        'rgba(75, 192, 192, 0.2)',
                        'rgba(54, 162, 235, 0.2)',
                        'rgba(153, 102, 255, 0.2)',
                        'rgba(255, 99, 132, 0.2)'
                    ],
                    borderColor: [
                        'rgba(75, 192, 192, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(153, 102, 255, 1)',
                        'rgba(255, 99, 132, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Time (seconds)'
                        }
                    }
                }
            }
        });
    }
    
    // LLM Requests Chart
    if (!metricsCharts.llmRequests) {
        const llmRequestsCtx = document.getElementById('llm-requests-chart').getContext('2d');
        metricsCharts.llmRequests = new Chart(llmRequestsCtx, {
            type: 'pie',
            data: {
                labels: ['Success', 'Cached', 'Failed'],
                datasets: [{
                    data: [65, 30, 5],
                    backgroundColor: [
                        'rgba(75, 192, 192, 0.2)',
                        'rgba(255, 206, 86, 0.2)',
                        'rgba(255, 99, 132, 0.2)'
                    ],
                    borderColor: [
                        'rgba(75, 192, 192, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(255, 99, 132, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }
    
    // Evidence Points Chart
    if (!metricsCharts.evidencePoints) {
        const evidencePointsCtx = document.getElementById('evidence-points-chart').getContext('2d');
        metricsCharts.evidencePoints = new Chart(evidencePointsCtx, {
            type: 'bar',
            data: {
                labels: ['Min', 'Average', 'Max'],
                datasets: [{
                    label: 'Evidence Points per PDF',
                    data: [5, 12, 25],
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Count'
                        }
                    }
                }
            }
        });
    }
    
    // Jobs Status Chart
    if (!metricsCharts.jobsStatus) {
        const jobsStatusCtx = document.getElementById('jobs-status-chart').getContext('2d');
        metricsCharts.jobsStatus = new Chart(jobsStatusCtx, {
            type: 'doughnut',
            data: {
                labels: ['Completed', 'Processing', 'Queued', 'Failed'],
                datasets: [{
                    data: [8, 2, 1, 1],
                    backgroundColor: [
                        'rgba(75, 192, 192, 0.2)',
                        'rgba(54, 162, 235, 0.2)',
                        'rgba(255, 206, 86, 0.2)',
                        'rgba(255, 99, 132, 0.2)'
                    ],
                    borderColor: [
                        'rgba(75, 192, 192, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(255, 99, 132, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }
    
    // In a real application, we would fetch actual metrics data from the server
    // and update these charts with real values
}