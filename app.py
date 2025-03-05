from flask import Flask, request, jsonify
import threading
from email_ai_assistant import authenticate_gmail, list_drafts, list_messages, generate_response, send_email

app = Flask(__name__)

# Function to handle reading emails and responding to them
def handle_emails():
    service = authenticate_gmail()
    if service:
        emails = list_messages(service)
        if not emails:
            print("No unread emails to process.")
        else:
            for email in emails:
                print(f"Processing email with Subject: {email['subject']}")
                response = generate_response(email['body'])
                # Send the response back to the sender (replace with actual sender email extraction)
                send_email(service, email['id'], response, 'sender@example.com')  # Replace with correct sender

# Route to trigger email processing
@app.route('/process_emails', methods=['GET'])
def process_emails():
    try:
        # Authenticate and get the Gmail service
        service = authenticate_gmail()
        if not service:
            return jsonify({'message': 'Authentication failed'}), 400
        
        # Fetch unread emails
        emails = list_messages(service)
        if not emails:
            return jsonify({'message': 'No new emails to process'}), 200
        
        # Process each unread email, generate AI responses, and save as drafts
        for email in emails:
            print(f"Processing email with Subject: {email['subject']}")
            
            # Generate AI response for the email
            response = generate_response(email['body'])
            print(f"AI Response for '{email['subject']}':\n{response}")
            
            # Save the response as a draft (don't send it)
            send_email(service, email['id'], response, sender_email='me')  # This will create a draft, not send
            
            print(f"Draft created for email with Subject: {email['subject']}")
        
        return jsonify({'message': 'Emails processed successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/get_email_responses', methods=['GET'])
def get_email_responses():
    # Authenticate and process emails
    service = authenticate_gmail()
    if service:
        emails = list_messages(service)
        email_responses = []
        for email in emails:
            response = generate_response(email['body'])
            email_responses.append({
                'id': email['id'],
                'subject': email['subject'],
                'body': email['body'],
                'response': response
            })
        return jsonify({'email_responses': email_responses}), 200
    else:
        return jsonify({'message': 'Authentication failed'}), 400


@app.route('/list_drafts', methods=['GET'])
def list_drafts_route():
    try:
        # Authenticate and get the Gmail service
        service = authenticate_gmail()
        if not service:
            return jsonify({'message': 'Authentication failed'}), 400
        
        drafts = list_drafts(service)  # Call the list_drafts function to fetch drafts
        return jsonify(drafts), 200  # Send back the drafts as JSON
    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route('/send_draft/<draft_id>', methods=['POST'])
def send_draft_route(draft_id):
    try:
        # Retrieve the draft by ID
        draft = service.users().messages().get(userId='me', id=draft_id).execute()
        
        # Retrieve the draft content (like the original send_email function)
        msg = draft['payload']
        raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode('utf-8')
        
        # Send the draft message
        send_message = service.users().messages().send(userId='me', body={'raw': raw_message}).execute()
        print(f"Successfully sent draft email with ID {draft_id}")
        return jsonify({'status': 'success', 'message': 'Email sent successfully'}), 200
    except Exception as e:
        print(f"Error sending draft: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/edit_draft/<draft_id>', methods=['POST'])
def edit_draft_route(draft_id):
    try:
        # Retrieve the current draft by ID
        draft = service.users().messages().get(userId='me', id=draft_id).execute()
        new_body = request.json.get('body')  # Get the new body from the request
         
        # Modify the draft with new content
        msg = draft['payload']
        msg['body']['data'] = base64.urlsafe_b64encode(new_body.encode()).decode('utf-8')  # Update the message body
         
        # Update the draft with the new content
        updated_draft = service.users().messages().modify(userId='me', id=draft_id, body={'message': msg}).execute()
        return jsonify({'status': 'success', 'message': 'Draft updated successfully'}), 200
    except Exception as e:
        print(f"Error editing draft: {e}")
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)