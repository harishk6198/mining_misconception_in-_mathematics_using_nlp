// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Initialize dashboard pie chart if element exists
    const chartContainer = document.getElementById('pie-chart');
    if (chartContainer) {
        initPieChart();
    }
    
    // File upload custom styling
    const fileInput = document.getElementById('file-upload');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name || 'No file chosen';
            document.getElementById('file-name').textContent = fileName;
        });
    }
    
    // Generate test predictions button
    const predictBtn = document.getElementById('generate-predictions-btn');
    if (predictBtn) {
        predictBtn.addEventListener('click', generateTestPredictions);
    }
    
    // Auto-dismiss alerts after 5 seconds
    setTimeout(function() {
        const alerts = document.querySelectorAll('.alert.alert-dismissible');
        alerts.forEach(function(alert) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);
});

// Function to initialize pie chart
function initPieChart() {
    // Fetch data from the API
    fetch('/plot-data')
        .then(response => response.json())
        .then(data => {
            if (data.labels.length > 0) {
                const ctx = document.getElementById('pie-chart').getContext('2d');
                
                // Create chart
                new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: data.labels,
                        datasets: [{
                            data: data.values,
                            backgroundColor: ['#8b5cf6', '#c026d3'],
                            borderColor: 'transparent',
                            borderWidth: 0
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        cutout: '60%',
                        plugins: {
                            legend: {
                                position: 'bottom',
                                labels: {
                                    color: '#e0e7ff',
                                    font: {
                                        family: "'Poppins', sans-serif",
                                        size: 14
                                    },
                                    padding: 20
                                }
                            },
                            tooltip: {
                                backgroundColor: 'rgba(14, 13, 38, 0.8)',
                                titleFont: {
                                    family: "'Outfit', sans-serif",
                                    size: 16
                                },
                                bodyFont: {
                                    family: "'Poppins', sans-serif",
                                    size: 14
                                },
                                borderColor: 'rgba(124, 58, 237, 0.3)',
                                borderWidth: 1
                            }
                        }
                    }
                });
            } else {
                document.getElementById('chart-container').innerHTML = '<p class="text-center text-muted">No data available for visualization</p>';
            }
        })
        .catch(error => {
            console.error('Error fetching plot data:', error);
            document.getElementById('chart-container').innerHTML = '<p class="text-center text-danger">Error loading chart data</p>';
        });
}

// Function to train the model
function trainModel() {
    const trainBtn = document.getElementById('train-model-btn');
    const originalText = trainBtn.innerHTML;
    
    // Update button state
    trainBtn.disabled = true;
    trainBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Training...';
    
    // Show loading message
    showAlert('Training model. This may take a few minutes...', 'info', true);
    
    // Send request to train endpoint
    fetch('/train', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showAlert(data.message, 'success');
            // Update model status if element exists
            const statusElement = document.getElementById('model-status');
            if (statusElement) {
                statusElement.textContent = 'Trained and Ready ✅';
            }
        } else {
            showAlert(data.message, 'danger');
        }
    })
    .catch(error => {
        console.error('Error training model:', error);
        showAlert('An error occurred while training the model.', 'danger');
    })
    .finally(() => {
        // Reset button state
        trainBtn.disabled = false;
        trainBtn.innerHTML = originalText;
    });
}

// Function to generate test predictions
function generateTestPredictions() {
    const predictBtn = document.getElementById('generate-predictions-btn');
    const originalText = predictBtn.innerHTML;
    
    // Update button state
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Generating...';
    
    // Show loading message
    showAlert('Generating predictions. This may take a moment...', 'info', true);
    
    // Send request to generate predictions endpoint
    fetch('/generate-test-predictions', {
        method: 'POST'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to generate predictions');
        }
        return response.blob();
    })
    .then(blob => {
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'submission.csv';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        showAlert('Predictions generated successfully! Download started.', 'success');
    })
    .catch(error => {
        console.error('Error generating predictions:', error);
        showAlert('An error occurred while generating predictions.', 'danger');
    })
    .finally(() => {
        // Reset button state
        predictBtn.disabled = false;
        predictBtn.innerHTML = originalText;
    });
}

// Function to show an alert message
function showAlert(message, type, isLoading = false) {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert-container .alert');
    existingAlerts.forEach(alert => alert.remove());
    
    // Create alert element
    const alertContainer = document.querySelector('.alert-container');
    if (!alertContainer) return;
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.role = 'alert';
    
    // Add loading spinner if loading
    let alertContent = '';
    if (isLoading) {
        alertContent = `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>`;
    }
    
    alertContent += message;
    alert.innerHTML = alertContent + '<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>';
    
    alertContainer.appendChild(alert);
    
    // Auto-dismiss after 5 seconds unless it's a loading alert
    if (!isLoading) {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    }
}