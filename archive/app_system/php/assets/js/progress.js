/**
 * progress.js
 * Handles form submissions and progress tracking for evaluations
 */

// Toggle manual persona selection
document.addEventListener('DOMContentLoaded', function() {
    const personaMode = document.getElementById('persona-mode');
    if (personaMode) {
        personaMode.addEventListener('change', function() {
            const manualPersonas = document.getElementById('manual-personas');
            manualPersonas.style.display = this.value === 'manual' ? 'block' : 'none';
        });
    }

    // Handle referee form submission
    const refereeForm = document.getElementById('referee-form');
    if (refereeForm) {
        refereeForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleRefereeSubmit(new FormData(this));
        });
    }

    // Handle section evaluator form submission
    const sectionForm = document.getElementById('section-eval-form');
    if (sectionForm) {
        sectionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleSectionEvalSubmit(new FormData(this));
        });
    }
});

/**
 * Handle referee evaluation submission
 */
function handleRefereeSubmit(formData) {
    // Show progress container
    const progressContainer = document.getElementById('progress-container');
    const form = document.getElementById('referee-form');

    form.style.display = 'none';
    progressContainer.style.display = 'block';

    // Submit via fetch
    fetch('index.php?action=run_referee', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
            form.style.display = 'block';
            progressContainer.style.display = 'none';
        } else if (data.redirect) {
            // Start polling for progress
            pollProgress(() => {
                window.location.href = data.redirect;
            });
        }
    })
    .catch(error => {
        alert('Request failed: ' + error);
        form.style.display = 'block';
        progressContainer.style.display = 'none';
    });
}

/**
 * Handle section evaluator submission
 */
function handleSectionEvalSubmit(formData) {
    // Show progress container
    const progressContainer = document.getElementById('section-progress-container');
    const form = document.getElementById('section-eval-form');

    form.style.display = 'none';
    progressContainer.style.display = 'block';

    // Submit via fetch
    fetch('index.php?action=run_section_eval', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
            form.style.display = 'block';
            progressContainer.style.display = 'none';
        } else if (data.redirect) {
            // For section eval, redirect after short delay (no real-time progress tracking yet)
            updateSectionProgress(50, 'Evaluating sections...');
            setTimeout(() => {
                updateSectionProgress(100, 'Complete!');
                setTimeout(() => {
                    window.location.href = data.redirect;
                }, 500);
            }, 2000);
        }
    })
    .catch(error => {
        alert('Request failed: ' + error);
        form.style.display = 'block';
        progressContainer.style.display = 'none';
    });
}

/**
 * Poll for evaluation progress
 */
function pollProgress(onComplete) {
    const progressFill = document.getElementById('progress-fill');
    const progressMessage = document.getElementById('progress-message');

    let attempts = 0;
    const maxAttempts = 600; // 10 minutes max (600 * 1 second)

    const interval = setInterval(() => {
        attempts++;

        // Fetch progress
        fetch('index.php?action=get_progress')
            .then(response => response.json())
            .then(data => {
                const progress = data.progress || 0;
                const message = data.message || 'Processing...';

                // Update UI
                progressFill.style.width = (progress * 100) + '%';
                progressMessage.textContent = message;

                // Check if complete
                if (progress >= 1.0) {
                    clearInterval(interval);
                    if (onComplete) {
                        onComplete();
                    }
                }

                // Check timeout
                if (attempts >= maxAttempts) {
                    clearInterval(interval);
                    alert('Evaluation timed out. Please try again.');
                    window.location.reload();
                }
            })
            .catch(error => {
                console.error('Progress fetch error:', error);
            });
    }, 1000); // Poll every second
}

/**
 * Update section evaluator progress (simplified)
 */
function updateSectionProgress(percent, message) {
    const progressFill = document.getElementById('section-progress-fill');
    const progressMessage = document.getElementById('section-progress-message');

    if (progressFill) {
        progressFill.style.width = percent + '%';
    }

    if (progressMessage) {
        progressMessage.textContent = message;
    }
}

/**
 * Validate form before submission
 */
function validateForm(formData) {
    const selectedFile = formData.get('selected_file');
    if (!selectedFile) {
        alert('Please select a file');
        return false;
    }
    return true;
}
