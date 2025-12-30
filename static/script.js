document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const resetBtn = document.getElementById('resetBtn');
    const videoFeed = document.getElementById('videoFeed');
    const loadingOverlay = document.querySelector('.loading-overlay');
    const feedbackBox = document.getElementById('feedbackBox');
    const feedbackText = document.getElementById('feedbackText');
    const bicepCount = document.getElementById('bicepCount');
    const squatCount = document.getElementById('squatCount');
    const lateralCount = document.getElementById('lateralCount');
    const currentExercise = document.getElementById('currentExercise');
    const accuracyValue = document.getElementById('accuracyValue');

    // State variables
    let streamActive = false;
    let statsInterval = null;

    // Functions
    function startCamera() {
        if (streamActive) return;

        // Show loading state
        loadingOverlay.classList.add('active');
        loadingOverlay.querySelector('p').textContent = 'Starting camera...';

        // Call the API to start the camera
        fetch('/start_camera')
            .then(response => response.json())
            .then(data => {
                console.log(data.status);
                
                // Set video source with timestamp to prevent caching
                videoFeed.src = `/video_feed?t=${new Date().getTime()}`;
                
                // Hide loading overlay when the image starts loading
                videoFeed.onload = function() {
                    loadingOverlay.classList.remove('active');
                    streamActive = true;
                    
                    // Update UI state
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    
                    // Start stats polling
                    startStatsPolling();
                };
                
                // Handle loading errors
                videoFeed.onerror = function() {
                    loadingOverlay.querySelector('p').textContent = 'Error loading video feed';
                    console.error('Failed to load video feed');
                };
            })
            .catch(error => {
                console.error('Error starting camera:', error);
                loadingOverlay.querySelector('p').textContent = 'Failed to start camera';
            });
    }

    function stopCamera() {
        if (!streamActive) return;

        // Stop stats polling
        stopStatsPolling();
        
        // Call the API to stop the camera
        fetch('/stop_camera')
            .then(response => response.json())
            .then(data => {
                console.log(data.status);
                
                // Clear video source
                videoFeed.src = '';
                
                // Update UI
                loadingOverlay.classList.add('active');
                loadingOverlay.querySelector('p').textContent = 'Camera stopped. Click "Start" to begin tracking';
                streamActive = false;
                
                // Update button states
                startBtn.disabled = false;
                stopBtn.disabled = true;
            })
            .catch(error => {
                console.error('Error stopping camera:', error);
            });
    }

    function resetCounters() {
        // Call the API to reset counters
        fetch('/reset_counters')
            .then(response => response.json())
            .then(data => {
                console.log(data.status);
                
                // Reset UI counters
                bicepCount.textContent = '0 reps';
                squatCount.textContent = '0 reps';
                lateralCount.textContent = '0 reps';
                currentExercise.textContent = 'Current Exercise: None';
                feedbackText.textContent = 'Counters reset';
                feedbackBox.className = 'feedback correct';
            })
            .catch(error => {
                console.error('Error resetting counters:', error);
            });
    }

    function startStatsPolling() {
        // Clear any existing interval
        if (statsInterval) {
            clearInterval(statsInterval);
        }
        
        // Poll for stats every 500ms
        statsInterval = setInterval(updateStats, 500);
    }

    function stopStatsPolling() {
        if (statsInterval) {
            clearInterval(statsInterval);
            statsInterval = null;
        }
    }

    function updateStats() {
        fetch('/get_stats')
            .then(response => response.json())
            .then(data => {
                // Update rep counters
                bicepCount.textContent = `${data.bicep_count} reps`;
                squatCount.textContent = `${data.squat_count} reps`;
                lateralCount.textContent = `${data.lateral_count} reps`;
                
                // Update current exercise
                currentExercise.textContent = `Current Exercise: ${data.current_exercise}`;
                
                // Update accuracy
                accuracyValue.textContent = `Pose Accuracy: ${data.accuracy.toFixed(1)}%`;
                
                // Update feedback
                feedbackText.textContent = data.feedback;
                
                // Update feedback box style based on form correctness
                if (data.form_correct) {
                    feedbackBox.className = 'feedback correct';
                } else {
                    feedbackBox.className = 'feedback incorrect';
                }
            })
            .catch(error => {
                console.error('Error updating stats:', error);
            });
    }

    // Event Listeners
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    resetBtn.addEventListener('click', resetCounters);

    // Handle page unload to ensure camera is stopped
    window.addEventListener('beforeunload', function() {
        if (streamActive) {
            fetch('/stop_camera').catch(() => {});
        }
    });
});