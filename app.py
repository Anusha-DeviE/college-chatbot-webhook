from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

data = {
    "fees": {
        "b.tech": 120000,
        "mba": 150000,
        "mca": 100000
    },
    "admission": "The admission process involves entrance exams and counseling.",
    "placement": "Our placement rate is over 90% with top companies visiting.",
    "duration": {
        "b.tech": "4 years",
        "mba": "2 years",
        "mca": "3 years"
    },
    "hostel": "Yes, hostel facilities are available for all programs."
}

classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")

intent_labels = ["fees", "admission", "placement", "duration", "hostel"]

@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json()
    intent = req.get("queryResult", {}).get("intent", {}).get("displayName", "")
    user_query = req.get("queryResult", {}).get("queryText", "").lower()
    parameters = req.get("queryResult", {}).get("parameters", {})
    course = parameters.get("course_name", "").lower()

    if intent == "ask_fees":
        fee = data["fees"].get(course)
        response = f"The fee for {course.upper()} is ₹{fee} per year." if fee else "Sorry, fee details not found."
    else:
        result = classifier(user_query, intent_labels)
        best_intent = result["labels"][0]

        if best_intent == "admission":
            response = data["admission"]
        elif best_intent == "placement":
            response = data["placement"]
        elif best_intent == "hostel":
            response = data["hostel"]
        elif best_intent == "duration":
            dur = data["duration"].get(course)
            response = f"The duration of {course.upper()} is {dur}." if dur else "Duration info not found."
        elif best_intent == "fees":
            fee = data["fees"].get(course)
            response = f"The fee for {course.upper()} is ₹{fee} per year." if fee else "Fee info not found."
        else:
            response = "Sorry, I couldn’t understand your question."

    return jsonify({"fulfillmentText": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
