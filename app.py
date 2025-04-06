from flask import Flask, render_template, redirect, url_for, request
import model
import os
from fileinput import filename

app = Flask(__name__)

@app.route('/', methods=["POST", "GET"])
def home():
    if request.method == "POST":
        audio = request.files["audio"]
        audio.save(audio.filename)

        isFart = model.predict_audio_class(audio.filename)
        os.remove(audio.filename)

        if isFart:
            return redirect(url_for('fart'))
        else:
            return redirect(url_for('art'))
        
    return render_template('index.html')

@app.route("/fart")
def fart():
    return render_template('fart.html')

@app.route("/art")
def art():
    return render_template('art.html')

if __name__ == "__main__":
    app.run(debug=True)