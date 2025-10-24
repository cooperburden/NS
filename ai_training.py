import os
import base64
import json
import logging
import re
import pickle
import email
from email.header import decode_header
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from bs4 import BeautifulSoup

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Set the path for your credentials
CLIENT_SECRET_PATH = '/Users/cooperburden/Desktop/NS/client_secret_867110489635-uusid9grhf9opo60ld7bpui257belgds.apps.googleusercontent.com.json'

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

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
        logging.error(f'Authentication error: {type(error).__name__}: {error}')
        return None

def get_last_50_threads(service):
    """Retrieve the last 50 email threads from the user's Primary inbox, excluding Promotions, Social, and Updates"""
    try:
        query = "-category:promotions -category:social -category:updates"
        logging.info(f'Fetching threads with query: "{query}"')
        
        # Get the list of threads
        results = service.users().threads().list(
            userId='me',
            labelIds=['INBOX'],
            q=query
        ).execute()
        threads = results.get('threads', [])
        if not threads:
            print('No threads found in Primary inbox.')
            return []

        thread_data = []
        count = 0
        for thread in threads[:50]:  # Limit to the last 50 threads
            thread_id = thread['id']
            # Fetch full thread details
            thread_details = service.users().threads().get(userId='me', id=thread_id, format='full').execute()
            messages = thread_details.get('messages', [])
            
            # Process each message in the thread
            thread_messages = []
            for msg in messages:
                payload = msg['payload']
                headers = payload['headers']
                message_data = {
                    'id': msg['id'],
                    'snippet': msg.get('snippet', 'No preview available'),
                    'from': next((header['value'] for header in headers if header['name'] == 'From'), 'Unknown'),
                    'subject': decode_subject(next((header['value'] for header in headers if header['name'] == 'Subject'), '')),
                    'date': next((header['value'] for header in headers if header['name'] == 'Date'), 'Unknown'),
                    'body': get_email_body(payload),
                }
                thread_messages.append(message_data)
            
            # Add the thread with all its messages
            thread_data.append({
                'thread_id': thread_id,
                'messages': thread_messages
            })
            count += 1
            if count >= 50:
                break

        return thread_data

    except Exception as error:
        logging.error(f'An error occurred in get_last_50_threads: {type(error).__name__}: {error}')
        return []

def decode_subject(subject):
    """Decode the subject"""
    decoded, charset = decode_header(subject)[0]
    if isinstance(decoded, bytes):
        return decoded.decode(charset or 'utf-8', errors='ignore')
    return decoded

def get_email_body(payload):
    """Extract and clean email body from Gmail API payload."""
    try:
        if 'body' in payload and 'data' in payload['body']:
            body_data = payload['body']['data']
            body = base64.urlsafe_b64decode(body_data).decode('utf-8', errors='ignore')
            return clean_text(body)

        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                    body_data = part['body']['data']
                    body = base64.urlsafe_b64decode(body_data).decode('utf-8', errors='ignore')
                    return clean_text(body)
                elif part['mimeType'] == 'text/html' and 'data' in part['body']:
                    html_body = part['body']['data']
                    html_body = base64.urlsafe_b64decode(html_body).decode('utf-8', errors='ignore')
                    return clean_text(strip_html_tags(html_body))
        return None
    except Exception as error:
        logging.error(f'Error extracting body: {type(error).__name__}: {error}')
        return None

def strip_html_tags(text):
    """Remove HTML tags from text."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def clean_text(text):
    """Remove URLs, HTML, boilerplate, and extra formatting from text."""
    text = strip_html_tags(text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    boilerplate_patterns = [
        r'Unsubscribe:.*',
        r'Copyright.*',
        r'This email was intended for.*',
        r'You are receiving.*',
        r'Want to change how you receive these emails?.*',
        r'Learn why we included this:.*',
        r'Help:.*',
        r'©.*LinkedIn.*',
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
    text = re.sub(r'\r\n|\n+', '\n', text)
    text = re.sub(r'-{2,}', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def save_emails_as_json(thread_data, filename='emails.json'):
    """Save the fetched thread data to a JSON file"""
    with open(filename, 'w') as json_file:
        json.dump(thread_data, json_file, indent=4)
        print(f"Saved {len(thread_data)} threads to {filename}")

def main():
    """Main function to fetch the last 50 threads and save them"""
    service = authenticate_gmail()
    if service:
        thread_data = get_last_50_threads(service)
        if thread_data:
            save_emails_as_json(thread_data)

if __name__ == '__main__':
    main()