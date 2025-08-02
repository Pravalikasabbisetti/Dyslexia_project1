
let currentStream = null;
let currentImage = null;

// Option selection
document.getElementById('upload-option').addEventListener('click', function() {
    selectOption('upload');
});

document.getElementById('capture-option').addEventListener('click', function() {
    selectOption('capture');
});

function selectOption(option) {
    // Reset all sections
    document.querySelectorAll('.option-card').forEach(card => card.classList.remove('active'));
    document.querySelectorAll('.upload-section, .camera-section, .preview-section').forEach(section => {
        section.classList.remove('active');
    });

    // Stop camera if it's running
    stopCamera();

    // Activate selected option
    if (option === 'upload') {
        document.getElementById('upload-option').classList.add('active');
        document.getElementById('upload-section').classList.add('active');
    } else if (option === 'capture') {
        document.getElementById('capture-option').classList.add('active');
        document.getElementById('camera-section').classList.add('active');
    }
}

// File upload handling
document.getElementById('file-input').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            alert('File size too large. Please choose a file smaller than 10MB.');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = function(e) {
            showPreview(e.target.result);
        };
        reader.readAsDataURL(file);
    }
});

// Camera handling
document.getElementById('start-camera').addEventListener('click', async function() {
    try {
        const constraints = {
            video: {
                facingMode: 'environment',
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        };

        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        const video = document.getElementById('video');
        video.srcObject = currentStream;
        
        // Wait for video to load
        video.onloadedmetadata = function() {
            video.play();
        };

        document.getElementById('start-camera').style.display = 'none';
        document.getElementById('capture-photo').style.display = 'inline-block';
        document.getElementById('stop-camera').style.display = 'inline-block';
    } catch (err) {
        console.error('Camera error:', err);
        alert('Error accessing camera: ' + err.message + '\nPlease ensure you have granted camera permissions.');
    }
});

document.getElementById('stop-camera').addEventListener('click', function() {
    stopCamera();
});

document.getElementById('capture-photo').addEventListener('click', function() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    if (video.videoWidth === 0 || video.videoHeight === 0) {
        alert('Camera not ready. Please wait a moment and try again.');
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const imageData = canvas.toDataURL('image/jpeg', 0.9);
    showPreview(imageData);
    stopCamera();
});

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => {
            track.stop();
        });
        currentStream = null;
    }
    
    // Reset camera controls
    document.getElementById('start-camera').style.display = 'inline-block';
    document.getElementById('capture-photo').style.display = 'none';
    document.getElementById('stop-camera').style.display = 'none';
    
    // Clear video
    const video = document.getElementById('video');
    video.srcObject = null;
}

function showPreview(imageSrc) {
    currentImage = imageSrc;
    document.getElementById('preview-image').src = imageSrc;
    document.getElementById('preview-section').classList.add('active');
}

// Submit handling
document.getElementById('submit-button').addEventListener('click', async function () {
    if (!currentImage) {
        alert("No image selected or captured.");
        return;
    }

    const formData = new FormData();

    // Convert base64 to a Blob
    const blob = dataURItoBlob(currentImage);
    formData.append("image", blob, "handwriting.jpg");

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        if (response.ok) {
            // Redirect to result page (Flask will return HTML)
            const resultHtml = await response.text();
            document.open();
            document.write(resultHtml);
            document.close();
        } else {
            const errorText = await response.text();
            alert("Error: " + errorText);
        }
    } catch (error) {
        console.error("Error during submission:", error);
        alert("Failed to analyze handwriting. Please try again.");
    }
});

// Reset functionality
document.getElementById('reset-button').addEventListener('click', function() {
    currentImage = null;
    document.getElementById('file-input').value = '';
    document.getElementById('preview-section').classList.remove('active');
    
    stopCamera();
    
    // Reset option selection
    document.querySelectorAll('.option-card').forEach(card => card.classList.remove('active'));
    document.querySelectorAll('.upload-section, .camera-section').forEach(section => {
        section.classList.remove('active');
    });
});

// Handle page unload
window.addEventListener('beforeunload', function() {
    stopCamera();
});
function dataURItoBlob(dataURI) {
    const byteString = atob(dataURI.split(',')[1]);
    const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];

    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }

    return new Blob([ab], { type: mimeString });
}

// Function to show results based on analysis
function showResults(hasDyslexia, correctedText = '', confidence = 75) {
    if (hasDyslexia) {
        document.getElementById('positiveResult').style.display = 'block';
        document.getElementById('negativeResult').style.display = 'none';
        
        // Update corrected text if provided
        if (correctedText) {
            document.getElementById('correctedText').textContent = correctedText;
        }
        
        // Update confidence
        document.getElementById('confidenceBar').style.width = confidence + '%';
        document.getElementById('confidenceLevel').textContent = confidence + '%';
        
        // Update confidence bar color based on level
        const confidenceBar = document.getElementById('confidenceBar');
        confidenceBar.className = 'confidence-fill ' + 
            (confidence >= 80 ? 'confidence-high' : 
             confidence >= 60 ? 'confidence-medium' : 'confidence-low');
    } else {
        document.getElementById('positiveResult').style.display = 'none';
        document.getElementById('negativeResult').style.display = 'block';
    }
}

// Navigation functions
function goToHome() {
    window.location.href = "{{ url_for('index') }}";
}

function downloadResults() {
    // Get the current result type
    const hasPositiveResult = document.getElementById('positiveResult').style.display !== 'none';
    
    let reportContent = 'DYSLEXIA ANALYSIS REPORT\n';
    reportContent += '=' .repeat(50) + '\n\n';
    reportContent += 'Date: ' + new Date().toLocaleDateString() + '\n';
    reportContent += 'Time: ' + new Date().toLocaleTimeString() + '\n\n';
    
    if (hasPositiveResult) {
        reportContent += 'RESULT: Dyslexia indicators detected\n';
        reportContent += 'Confidence Level: ' + document.getElementById('confidenceLevel').textContent + '\n\n';
        
        reportContent += 'CORRECTED TEXT:\n';
        reportContent += '-' .repeat(20) + '\n';
        reportContent += document.getElementById('correctedText').textContent + '\n\n';
        
        reportContent += 'RECOMMENDATIONS:\n';
        reportContent += '-' .repeat(20) + '\n';
        reportContent += '• Reading Techniques: Use a ruler or finger to track lines while reading. Consider using colored overlays or special fonts designed for dyslexia.\n\n';
        reportContent += '• Writing Support: Practice letter formation with large, spaced letters. Use graph paper to help with spacing and alignment.\n\n';
        reportContent += '• Professional Support: Consider consulting with a learning specialist or educational psychologist for comprehensive assessment and personalized strategies.\n\n';
        reportContent += '• Technology Tools: Explore text-to-speech software, spell checkers, and apps designed specifically for dyslexic learners.\n\n';
        
        reportContent += 'IMPORTANT NOTE:\n';
        reportContent += 'This is a preliminary assessment. Please consult with a healthcare professional for proper diagnosis and treatment.\n';
    } else {
        reportContent += 'RESULT: No dyslexia indicators detected\n';
        reportContent += 'Confidence Level: 90%\n\n';
        
        reportContent += 'ANALYZED TEXT:\n';
        reportContent += '-' .repeat(20) + '\n';
        reportContent += document.querySelector('#negativeResult .text-content').textContent + '\n\n';
        
        reportContent += 'RECOMMENDATIONS:\n';
        reportContent += '-' .repeat(20) + '\n';
        reportContent += '• Continue Good Practices: Keep practicing regular writing to maintain and improve your skills.\n\n';
        reportContent += '• Stay Aware: If you notice any changes in your reading or writing abilities, don\'t hesitate to seek professional advice.\n\n';
        
        reportContent += 'NOTE:\n';
        reportContent += 'If you continue to experience reading or writing difficulties, we recommend consulting with a healthcare professional.\n';
    }
    
    // Create and download the file with proper settings
    const blob = new Blob([reportContent], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'handwriting_analysis_report.txt';
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 100);
}

function newTest() {
    // Redirect to home page
    window.location.href = "{{ url_for('index') }}";
}

/* // Example usage - you can call this with your actual analysis results
// For demonstration, showing positive result
showResults(true, 'The quick brown fox jumps over the lazy dog. This corrected text shows proper spelling and formatting.', 75);

// To show negative result instead, uncomment the line below:
// showResults(false);*/


