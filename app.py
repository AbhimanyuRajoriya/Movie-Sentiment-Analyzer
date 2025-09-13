from flask import Flask, request, jsonify, render_template_string
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizor.pkl")


@app.route("/")
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentiment Analyzer</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                background: linear-gradient(135deg, #1e3c72, #2a5298);
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .card {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 8px 20px rgba(0,0,0,0.3);
                width: 500px;
                max-width: 90%;
                text-align: center;
            }
            textarea {
                resize: none;
                border-radius: 10px;
                padding: 10px;
                width: 100%;
                border: none;
                outline: none;
                font-size: 16px;
            }
            button {
                border-radius: 30px;
                padding: 10px 25px;
                font-size: 16px;
                font-weight: bold;
                background: #ff9800;
                border: none;
                transition: 0.3s;
            }
            button:hover {
                background: #e68900;
                transform: scale(1.05);
            }
            #result {
                margin-top: 20px;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                border-radius: 8px;
                display: none;
            }
            .pos {
                background-color: rgba(76, 175, 80, 0.8);
            }
            .neg {
                background-color: rgba(244, 67, 54, 0.8);
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h2 class="mb-4">Movie Review Sentiment</h2>
            <textarea id="review" rows="5" placeholder="Type your review here..."></textarea>
            <br><br>
            <button onclick="analyze()">Analyze</button>
            <div id="result"></div>
        </div>

        <script>
            async function analyze() {
                let text = document.getElementById("review").value;
                let response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: text })
                });
                let data = await response.json();
                let resultDiv = document.getElementById("result");
                
                if(data.Sentiment){
                    resultDiv.style.display = "block";
                    resultDiv.innerText = "Sentiment: " + data.Sentiment.toUpperCase();
                    if(data.Sentiment === "pos"){
                        resultDiv.className = "pos";
                    } else {
                        resultDiv.className = "neg";
                    }
                }
            }
        </script>
    </body>
    </html>
    """)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    x = vectorizer.transform([text])
    pred = model.predict(x)[0]

    return jsonify({"Sentiment": pred})

if __name__ == "__main__":
    app.run(debug=True)
