// Fixes for main.js (or create a new enhanced version)

// Safe element getter with null checking
function safeGetElement(id) {
    const element = document.getElementById(id);
    if (!element && window.console) {
        console.warn(`Element not found: ${id}`);
    }
    return element;
}

// Safe element initialization
function initializeElements() {
    // Get all elements with null checking
    const elements = {
        fileInput: safeGetElement('fileInput'),
        analyzeBtn: safeGetElement('analyzeBtn'),
        batchAnalyzeBtn: safeGetElement('batchAnalyzeBtn'),
        resultsDiv: safeGetElement('results'),
        loadingDiv: safeGetElement('loading'),
        errorDiv: safeGetElement('error'),
        progressBar: safeGetElement('progressBar'),
        progressText: safeGetElement('progressText'),
        
        // Optional elements that might not exist
        analysisMethodSelect: safeGetElement('analysisMethodSelect'),
        colorMethodsDiv: safeGetElement('colorMethodsDiv'),
        pixelRatioInput: safeGetElement('pixelRatio'),
        chlorophyllThresholdInput: safeGetElement('chlorophyllThreshold'),
        minCellAreaInput: safeGetElement('minCellArea'),
        maxCellAreaInput: safeGetElement('maxCellArea')
    };
    
    return elements;
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    const elements = initializeElements();
    
    // Only attach event listeners to elements that exist
    if (elements.fileInput) {
        elements.fileInput.addEventListener('change', function(e) {
            const files = e.target.files;
            if (files.length > 0) {
                updateFileInfo(files);
            }
        });
    }
    
    if (elements.analyzeBtn) {
        elements.analyzeBtn.addEventListener('click', performAnalysis);
    }
    
    if (elements.batchAnalyzeBtn) {
        elements.batchAnalyzeBtn.addEventListener('click', performBatchAnalysis);
    }
    
    // Check system health on load
    checkSystemHealth();
});

// Safe display functions
function showLoading(message = 'Processing...') {
    const loadingDiv = safeGetElement('loading');
    if (loadingDiv) {
        loadingDiv.innerHTML = `
            <div class="spinner-border text-primary" role="status">
                <span class="sr-only">Loading...</span>
            </div>
            <p class="mt-2">${message}</p>
        `;
        loadingDiv.style.display = 'block';
    }
}

function hideLoading() {
    const loadingDiv = safeGetElement('loading');
    if (loadingDiv) {
        loadingDiv.style.display = 'none';
    }
}

function showError(message) {
    const errorDiv = safeGetElement('error');
    if (errorDiv) {
        errorDiv.innerHTML = `
            <div class="alert alert-danger alert-dismissible fade show" role="alert">
                <strong>Error:</strong> ${message}
                <button type="button" class="close" data-dismiss="alert" aria-label="Close">
                    <span aria-hidden="true">&times;</span>
                </button>
            </div>
        `;
        errorDiv.style.display = 'block';
    }
}

function hideError() {
    const errorDiv = safeGetElement('error');
    if (errorDiv) {
        errorDiv.style.display = 'none';
    }
}

// Safe parameter extraction
function getAnalysisParameters() {
    const params = new FormData();
    
    // Add parameters only if elements exist
    const pixelRatio = safeGetElement('pixelRatio');
    if (pixelRatio) {
        params.append('pixel_ratio', pixelRatio.value || '1.0');
    }
    
    const chlorophyllThreshold = safeGetElement('chlorophyllThreshold');
    if (chlorophyllThreshold) {
        params.append('chlorophyll_threshold', chlorophyllThreshold.value || '0.6');
    }
    
    const minCellArea = safeGetElement('minCellArea');
    if (minCellArea) {
        params.append('min_cell_area', minCellArea.value || '30');
    }
    
    const maxCellArea = safeGetElement('maxCellArea');
    if (maxCellArea) {
        params.append('max_cell_area', maxCellArea.value || '8000');
    }
    
    const analysisMethod = safeGetElement('analysisMethodSelect');
    if (analysisMethod) {
        params.append('analysis_method', analysisMethod.value || 'auto');
    }
    
    const colorMethod = safeGetElement('colorMethod');
    if (colorMethod) {
        params.append('color_method', colorMethod.value || 'green_wolffia');
    }
    
    return params;
}

// Perform single image analysis
async function performAnalysis() {
    const fileInput = safeGetElement('fileInput');
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        showError('Please select an image file first.');
        return;
    }
    
    const formData = getAnalysisParameters();
    formData.append('image', fileInput.files[0]);
    
    hideError();
    showLoading('Analyzing image...');
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        hideLoading();
        
        if (response.ok && result.success) {
            displayResults(result);
        } else {
            showError(result.error || 'Analysis failed');
        }
    } catch (error) {
        hideLoading();
        showError('Network error: ' + error.message);
    }
}

// Display analysis results
function displayResults(result) {
    const resultsDiv = safeGetElement('results');
    if (!resultsDiv) return;
    
    let html = '<div class="card mt-3">';
    html += '<div class="card-header"><h5>Analysis Results</h5></div>';
    html += '<div class="card-body">';
    
    // Summary statistics
    if (result.summary) {
        html += '<h6>Summary Statistics</h6>';
        html += '<div class="row">';
        html += `<div class="col-md-3"><strong>Total Cells:</strong> ${result.total_cells || 0}</div>`;
        html += `<div class="col-md-3"><strong>Avg Area:</strong> ${(result.summary.avg_area || 0).toFixed(2)} μm²</div>`;
        html += `<div class="col-md-3"><strong>Chlorophyll Ratio:</strong> ${(result.summary.chlorophyll_ratio || 0).toFixed(1)}%</div>`;
        html += `<div class="col-md-3"><strong>Quality Score:</strong> ${(result.quality_score || 0).toFixed(3)}</div>`;
        html += '</div>';
    }
    
    // Visualizations
    if (result.visualizations) {
        html += '<div class="mt-3">';
        html += '<h6>Visualizations</h6>';
        html += '<div class="row">';
        
        if (result.visualizations.original_image) {
            html += '<div class="col-md-6">';
            html += '<img src="data:image/png;base64,' + result.visualizations.original_image + '" class="img-fluid" alt="Original">';
            html += '<p class="text-center">Original Image</p>';
            html += '</div>';
        }
        
        if (result.visualizations.segmentation) {
            html += '<div class="col-md-6">';
            html += '<img src="data:image/png;base64,' + result.visualizations.segmentation + '" class="img-fluid" alt="Segmentation">';
            html += '<p class="text-center">Cell Segmentation</p>';
            html += '</div>';
        }
        
        html += '</div>';
        html += '</div>';
    }
    
    // ML Enhancements
    if (result.ml_enhancements) {
        html += '<div class="mt-3">';
        html += '<h6>ML Analysis</h6>';
        
        if (result.ml_enhancements.ml_classifications) {
            const classifications = result.ml_enhancements.ml_classifications;
            html += `<p><strong>ML Classifications:</strong> ${classifications.predictions ? classifications.predictions.length : 0} cells classified</p>`;
            html += `<p><strong>High Confidence:</strong> ${classifications.high_confidence_predictions || 0} cells</p>`;
        }
        
        if (result.ml_enhancements.anomaly_detection) {
            const anomalies = result.ml_enhancements.anomaly_detection;
            html += `<p><strong>Anomalies Detected:</strong> ${anomalies.combined_anomalies || 0} cells</p>`;
        }
        
        html += '</div>';
    }
    
    // Processing info
    if (result.analysis_metadata) {
        html += '<div class="mt-3">';
        html += '<h6>Processing Information</h6>';
        html += `<p><strong>Processing Time:</strong> ${result.analysis_metadata.processing_time.toFixed(2)} seconds</p>`;
        html += `<p><strong>Analysis ID:</strong> ${result.analysis_metadata.analysis_id}</p>`;
        html += '</div>';
    }
    
    html += '</div></div>';
    
    resultsDiv.innerHTML = html;
    resultsDiv.style.display = 'block';
}

// Check system health
async function checkSystemHealth() {
    try {
        const response = await fetch('/api/health');
        const health = await response.json();
        
        const statusElement = safeGetElement('systemStatus');
        if (statusElement) {
            if (health.status === 'healthy') {
                statusElement.innerHTML = '<span class="badge badge-success">System Ready</span>';
            } else {
                statusElement.innerHTML = '<span class="badge badge-warning">System Degraded</span>';
            }
        }
        
        // Update UI based on available components
        if (health.components) {
            updateUIForComponents(health.components);
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Update UI based on available components
function updateUIForComponents(components) {
    // Show/hide ML training button if ML is available
    const mlTrainBtn = safeGetElement('mlTrainBtn');
    if (mlTrainBtn) {
        mlTrainBtn.style.display = components.ml_enhancement ? 'inline-block' : 'none';
    }
    
    // Show/hide batch processing if available
    const batchBtn = safeGetElement('batchAnalyzeBtn');
    if (batchBtn) {
        batchBtn.style.display = components.batch_processing ? 'inline-block' : 'none';
    }
}

// File info update
function updateFileInfo(files) {
    const fileInfoDiv = safeGetElement('fileInfo');
    if (fileInfoDiv) {
        if (files.length === 1) {
            fileInfoDiv.innerHTML = `Selected: ${files[0].name} (${(files[0].size / 1024 / 1024).toFixed(2)} MB)`;
        } else {
            fileInfoDiv.innerHTML = `Selected: ${files.length} files`;
        }
    }
}