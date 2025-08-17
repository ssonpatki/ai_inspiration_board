# flask: run local web server
# render_template: renders HTML templates
# request: handle user input from frontend form
from flask import Flask, render_template, request

# pipline: import hugging face transformer pipeline (access to pretrained NLP models)
from transformers import pipeline

app = Flask(__name__)   # initialize flask app

# Load pretrained small GPT model locally (given prompt -> generate text)
    # GPT = generative pre-trained transformer (subset of LLM)
generator = pipeline("text-generation", model="distilgpt2")

# Define main route of web app (/)
# Accept GET (load page) and POST (submit form) requests
@app.route("/", methods=["GET", "POST"])   
def home():
    output = "" # empty string to store AI response
    if request.method == "POST":    # check if user submitted the form
        topic = request.form["topic"]   # get text from user/form input
        # Create prompt for LLM (prompt = ...)
        prompt = f"Give a short summary about {topic} and 3 suggested next steps:"
        # Send prompt to GPT model to generate response (generator(prompt...))
        result = generator(prompt, max_length=100, num_return_sequences=1)  # output limited to 100 char
        output = result[0]['generated_text']    # extract generated text from model's output
    # send response to index.html to display text on page
    return render_template("index.html", output=output) 

if __name__ == "__main__":
    app.run(debug=True)
