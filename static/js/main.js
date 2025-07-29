let currentLogId = null;

// --- Main form submission ---
document.getElementById('spam-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const message = document.getElementById('message-input').value;
    const resultContainer = document.getElementById('result-container');
    const predictionResult = document.getElementById('prediction-result');
    const hamScore = document.getElementById('ham-score');
    const spamScore = document.getElementById('spam-score');

    predictionResult.textContent = '...';
    resultContainer.className = 'hidden';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        if (response.ok) {
            const prediction = data.prediction;
            predictionResult.textContent = prediction.toUpperCase();
            resultContainer.className = 'result-container';
            predictionResult.className = '';
            resultContainer.classList.add(prediction);
            predictionResult.classList.add(prediction);
            hamScore.textContent = `Ham: ${(data.confidence.ham * 100).toFixed(2)}%`;
            spamScore.textContent = `Spam: ${(data.confidence.spam * 100).toFixed(2)}%`;

            // Store the log ID and show feedback buttons
            currentLogId = data.log_id;
            document.getElementById('feedback-container').classList.remove('hidden');
            document.getElementById('feedback-thanks').classList.add('hidden');
        } else {
            predictionResult.textContent = `Error: ${data.error}`;
            resultContainer.className = 'result-container spam';
        }
    } catch (error) {
        predictionResult.textContent = 'An unexpected error occurred.';
        resultContainer.className = 'result-container spam';
    }
});

// --- Feedback button listeners (moved outside) ---
document.getElementById('correct-btn').addEventListener('click', () => sendFeedback(true));
document.getElementById('incorrect-btn').addEventListener('click', () => sendFeedback(false));

async function sendFeedback(isCorrect) {
    if (!currentLogId) return;

    await fetch('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ log_id: currentLogId, is_correct: isCorrect })
    });

    document.getElementById('feedback-container').classList.add('hidden');
    document.getElementById('feedback-thanks').classList.remove('hidden');
    currentLogId = null; // Reset log ID
}