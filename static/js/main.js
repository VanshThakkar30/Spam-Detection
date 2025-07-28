document.getElementById('spam-form').addEventListener('submit', async function(event) {
    event.preventDefault(); // Prevent the form from submitting the traditional way

    const message = document.getElementById('message-input').value;
    const resultContainer = document.getElementById('result-container');
    const predictionResult = document.getElementById('prediction-result');
    const hamScore = document.getElementById('ham-score');
    const spamScore = document.getElementById('spam-score');

    // Show a loading state if you want
    predictionResult.textContent = '...';
    resultContainer.className = 'hidden';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        if (response.ok) {
            const prediction = data.prediction;
            predictionResult.textContent = prediction.toUpperCase();

            // Style the result container based on the prediction
            resultContainer.className = 'result-container';
            predictionResult.className = '';
            resultContainer.classList.add(prediction);
            predictionResult.classList.add(prediction);

            // Display confidence scores
            hamScore.textContent = `Ham: ${(data.confidence.ham * 100).toFixed(2)}%`;
            spamScore.textContent = `Spam: ${(data.confidence.spam * 100).toFixed(2)}%`;

        } else {
            predictionResult.textContent = `Error: ${data.error}`;
            resultContainer.className = 'result-container spam';
        }
    } catch (error) {
        predictionResult.textContent = 'An unexpected error occurred.';
        resultContainer.className = 'result-container spam';
    }
});
