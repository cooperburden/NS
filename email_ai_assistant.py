
import os
import pickle
import base64
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import openai  # Import OpenAI
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import mimetypes
from googleapiclient.errors import HttpError





# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 
          'https://www.googleapis.com/auth/gmail.send', 
          'https://www.googleapis.com/auth/gmail.modify']
CLIENT_SECRET_PATH = '/Users/cooperburden/Desktop/NS/client_secret_867110489635-uusid9grhf9opo60ld7bpui257belgds.apps.googleusercontent.com.json'

# Set OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

def authenticate_gmail():
    creds = None
    try:
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_PATH, SCOPES)
                creds = flow.run_local_server(port=8080)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        return build('gmail', 'v1', credentials=creds)
    except Exception as error:
        print(f'Authentication error: {error}')
        return None

def list_labels(service):
    try:
        results = service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])
        if not labels:
            print('No labels found.')
        else:
            print('Labels:')
            for label in labels:
                print(f"- {label['name']}")
    except Exception as error:
        print(f'Error fetching labels: {error}')

def list_messages(service):
    try:
        # Filter for unread emails in Inbox, excluding Promotions, Social, and Updates
        results = service.users().messages().list(
            userId='me', 
            labelIds=['INBOX'], 
            q="is:unread -category:promotions -category:social -category:updates"
        ).execute()
        messages = results.get('messages', [])
        if not messages:
            print('No new Primary inbox messages.')
            return []
        else:
            email_list = []
            for message in messages[:5]:
                msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
                headers = {h['name']: h['value'] for h in msg['payload']['headers']}
                subject = headers.get('Subject', 'No Subject')
                if 'parts' in msg['payload']:
                    for part in msg['payload']['parts']:
                        if part['mimeType'] == 'text/plain':
                            body = part['body']['data']
                            break
                    else:
                        body = msg['payload']['body'].get('data', '')
                else:
                    body = msg['payload']['body'].get('data', '')
                if body:
                    body = base64.urlsafe_b64decode(body).decode('utf-8')
                else:
                    body = msg['snippet']
                
                # Add 'headers' to the dictionary that is appended to the email list
                email_list.append({'id': message['id'], 'subject': subject, 'body': body, 'headers': headers})
                
                print(f"Subject: {subject}\nBody: {body}\n")
            return email_list
    except Exception as error:
        print(f'Error fetching messages: {error}')
        return []



# this is how the AI is trained and is able to respond intuitively
def generate_response(email_body):
    prompt = f"""
    You are an AI assistant for a business selling dual-language Book of Mormon scriptures (English on the left, another language on the right). 
    Available languages: Spanish, Portuguese, French, Tongan. Japanese is in progress. 
    Customers vote on our website for the next language. Shipping takes 5-7 business days. 
    The website has an FAQ with ordering and shipping details. 
    Respond politely and concisely to this email as if it’s a customer inquiry, sticking to our business context: {email_body}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message['content']
    except Exception as error:
        print(f'Error generating response: {error}')
        return "Sorry, I couldn’t generate a response at this time."


# this is how the application can correctly respond to emails, like responding to the 
# correct email and subject 
def send_email(service, message_id, response, sender_email):
    try:
        # Get the email details to retrieve the original email subject and recipient (From header)
        message = service.users().messages().get(userId='me', id=message_id).execute()
        headers = {h['name']: h['value'] for h in message['payload']['headers']}
        subject = headers.get('Subject', 'No Subject')
        to_email = headers.get('From', 'to@example.com')  # Use the "From" header or a default one
        
        # Create the email message with AI response
        msg = MIMEMultipart()
        msg['From'] = sender_email  # Sender's email
        msg['To'] = to_email  # Recipient email
        msg['Subject'] = "Re: " + subject  # Subject of the reply

        # Attach the AI response to the email body
        msg.attach(MIMEText(response, 'plain'))

        # Encode the message as base64
        raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode('utf-8')

        # Prepare the draft creation request
        create_message = {
            'message': {
                'raw': raw_message  # 'raw' field must be inside the 'message' key
            }
        }

        # Create the draft using the Gmail API
        draft = service.users().drafts().create(userId='me', body=create_message).execute()
        print(f'Successfully created draft email to {to_email}')
        
    except HttpError as error:
        print(f'Error creating draft: {error}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')



# this is a list view of all of the emails, and a snippet of the draft that was written.
def list_drafts(service):
    try:
        # Fetch drafts
        results = service.users().drafts().list(userId='me').execute()
        drafts = results.get('drafts', [])
        
        if not drafts:
            print('No drafts found.')
            return []
        else:
            draft_list = []
            for draft in drafts:
                # Get the full draft details using its ID
                draft_msg = service.users().drafts().get(userId='me', id=draft['id']).execute()
                
                # Extract subject and snippet (body preview) from the draft
                headers = draft_msg['message']['payload']['headers']
                subject = next((header['value'] for header in headers if header['name'] == 'Subject'), 'No Subject')
                snippet = draft_msg['message'].get('snippet', 'No preview available')
                
                draft_list.append({'id': draft['id'], 'subject': subject, 'body': snippet})
                
                # Print draft details (for debugging)
                print(f"Draft Subject: {subject}\nBody: {snippet}")
                
            return draft_list
    except Exception as error:
        print(f'Error fetching drafts: {error}')
        return []


# this is a function to actually see the message and the AI generated message as a draft

def get_draft_details(service, draft_id):
    try:
        # Retrieve the draft's full details
        draft = service.users().drafts().get(userId='me', id=draft_id).execute()
        
        # Extract the message details
        message = draft['message']
        headers = {h['name']: h['value'] for h in message['payload']['headers']}
        subject = headers.get('Subject', 'No Subject')
        
        # Decode the body if it's in base64 format
        if 'parts' in message['payload']:
            for part in message['payload']['parts']:
                if part['mimeType'] == 'text/plain':
                    body = part['body'].get('data', '')
                    body = base64.urlsafe_b64decode(body).decode('utf-8')
                    break
        else:
            body = message['payload']['body'].get('data', '')
            body = base64.urlsafe_b64decode(body).decode('utf-8')

        print(f"Subject: {subject}")
        print(f"Body: {body}")
        return {'subject': subject, 'body': body}
    
    except HttpError as error:
        print(f'Error retrieving draft details: {error}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        return None



# this is how you can actually edit the draft that AI created
def update_draft(service, draft_id, new_subject, new_body):
    try:
        # Retrieve the draft
        draft = service.users().drafts().get(userId='me', id=draft_id).execute()
        
        # Update the subject and body of the email
        message = draft['message']
        headers = {h['name']: h['value'] for h in message['payload']['headers']}
        headers['Subject'] = new_subject
        
        # Create a new MIME message
        msg = MIMEMultipart()
        msg['From'] = message['payload']['headers'][1]['value']  # Sender email
        msg['To'] = message['payload']['headers'][0]['value']  # Recipient email
        msg['Subject'] = new_subject
        msg.attach(MIMEText(new_body, 'plain'))  # New body text

        # Encode the updated message as base64
        raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode('utf-8')

        # Prepare the update request
        create_message = {
            'message': {
                'raw': raw_message  # 'raw' field must be inside the 'message' key
            }
        }

        # Update the draft with the new content
        updated_draft = service.users().drafts().update(userId='me', id=draft_id, body=create_message).execute()
        print(f"Draft updated successfully: {new_subject}")

    except HttpError as error:
        print(f'Error updating draft: {error}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')




# this is how you send the draft that you made edits to
def send_draft(service, draft_id):
    try:
        # Send the draft email
        send_message = service.users().drafts().send(userId='me', body={'id': draft_id}).execute()
        print(f"Draft sent successfully: {send_message['id']}")
    except HttpError as error:
        print(f'Error sending draft: {error}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')







def main():
    service = authenticate_gmail()
    if service:
        list_labels(service)
        emails = list_messages(service)
        if not emails:
            print("No unread emails to process.")
        else:
            email_responses = []
            for email in emails:
                print(f"Processing email with Subject: {email['subject']}")
                response = generate_response(email['body'])
                email_responses.append({
                    'id': email['id'],
                    'subject': email['subject'],
                    'body': email['body'],
                    'response': response,
                    'sender': 'me'  # Replace with actual sender from headers
                })
                print(f"AI Response for '{email['subject']}':\n{response}\n")
            # Save to a file or pass to app (for now, just print)
            return email_responses



                
if __name__ == '__main__':
    main()