import requests

url = "http://127.0.0.1:8000/healthcare-assist/"


files = [
    ('files', ('Cardiologist_Report.pdf', open('input/Cardiologist_Report.pdf', 'rb'), 'application/pdf')),
    ('files', ('Psychologist_Report.pdf', open('input/Psychologist_Report.pdf', 'rb'), 'application/pdf')),
    ('files', ('Pulmonologist_Report.pdf', open('input/Pulmonologist_Report.pdf', 'rb'), 'application/pdf')),
]

# The form data (symptoms or question and agent type)
data = {
    'query': 'Patient has been experiencing shortness of breath, chest tightness, and anxiety. What are the possible causes and next steps?',
    'option': 'MultidisciplinaryTeam'  # Could be Cardiologist, Psychologist, Pulmonologist, or MultidisciplinaryTeam
}

# Send the request
response = requests.post(url, files=files, data=data)

# Print the response
print(response.json())
