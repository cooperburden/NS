
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
from urllib.parse import urljoin, urlparse

from dotenv import load_dotenv
load_dotenv()

from playwright.sync_api import sync_playwright




# --- FAQ scraping and embedding setup ---
import os, time, json, re, hashlib, requests, numpy as np
from bs4 import BeautifulSoup
import openai

FAQ_URL = os.getenv("FAQ_URL", "https://example.com/faq")  # your live FAQ URL
FAQ_INDEX_PATH = "faq_index.json"
EMBED_MODEL = "text-embedding-3-small"  # cheaper + fast embeddings

def _clean_html_to_text(html: str) -> str:
    """Remove scripts, styles, and return clean readable text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 150):
    """Split text into overlapping chunks for embeddings."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0: start = 0
        if end == len(text): break
    return [c.strip() for c in chunks if c.strip()]

def _embed(texts: list[str]) -> list[list[float]]:
    """Create OpenAI embeddings for list of texts."""
    resp = openai.Embedding.create(model=EMBED_MODEL, input=texts)
    return [d["embedding"] for d in resp["data"]]

def _sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()

def _fetch_faq_html(url: str) -> tuple[str, dict]:
    """Download the FAQ page HTML + header metadata."""
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    meta = {"etag": r.headers.get("ETag"), "last_modified": r.headers.get("Last-Modified")}
    return r.text, meta

def build_or_refresh_faq_index(force: bool = False) -> dict:
    """
    Build or refresh the local FAQ embedding index.
    Returns a JSON with chunks + embeddings so it can be reused.
    """
    existing = None
    if os.path.exists(FAQ_INDEX_PATH):
        with open(FAQ_INDEX_PATH, "r") as f:
            existing = json.load(f)

    # Fetch latest HTML + check for changes
    html, meta = _fetch_faq_html(FAQ_URL)
    text = _clean_html_to_text(html)
    new_hash = _sha1(text)
    old_hash = existing.get("content_hash") if existing else None
    if existing and (not force) and (new_hash == old_hash):
        return existing  # nothing changed

    # Create new chunks + embeddings
    chunks = _chunk_text(text)
    embeddings = _embed(chunks)
    indexed = [{"id": _sha1(t)[:16], "text": t, "embedding": e} for t, e in zip(chunks, embeddings)]

    index = {
        "url": FAQ_URL,
        "updated_at": int(time.time()),
        "meta": meta,
        "content_hash": new_hash,
        "chunks": indexed,
    }
    with open(FAQ_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)
    return index

def _load_index_or_build() -> dict:
    """Load local index if it exists, otherwise build it."""
    if os.path.exists(FAQ_INDEX_PATH):
        with open(FAQ_INDEX_PATH, "r") as f:
            return json.load(f)
    return build_or_refresh_faq_index(force=True)

def _retrieve(email_text: str, k: int = 4) -> list[dict]:
    """Return top-k FAQ chunks by cosine similarity."""
    index = _load_index_or_build()
    if not index.get("chunks"):
        return []

    q_emb = _embed([email_text])[0]
    qv = np.array(q_emb, dtype=np.float32)

    scored = []
    for ch in index["chunks"]:
        v = np.array(ch["embedding"], dtype=np.float32)
        denom = (np.linalg.norm(qv) * np.linalg.norm(v)) or 1e-8
        sim = float(np.dot(qv, v) / denom)
        scored.append((sim, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]
# --- end FAQ scraping and embedding setup ---


def _fetch_rendered_html(url: str) -> str:
    """
    Use Playwright to fetch HTML *after* JavaScript runs.
    This is important for Shopify product pages where prices are rendered by JS.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=30000)
        page.wait_for_load_state("networkidle")
        html = page.content()
        browser.close()
        return html






# --- Whole-site crawling and indexing (Native Scripture) ---
SITE_START_URL = os.getenv("SITE_START_URL", "https://nativescripture.com")
SITE_INDEX_PATH = "site_index.json"
MAX_SITE_PAGES = 50  # safety cap so we don't crawl endlessly


def crawl_and_build_site_index(start_url: str = SITE_START_URL, max_pages: int = MAX_SITE_PAGES):
    """
    Crawl the Native Scripture site starting at start_url, collect text from each page,
    chunk + embed it, and save everything to site_index.json.
    """
    visited = set()
    to_visit = [start_url]
    index = {}

    # we'll reuse BeautifulSoup, requests, _clean_html_to_text, _chunk_text, _embed from above

    start_parsed = urlparse(start_url)

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            print(f"[crawl] Fetching {url}")

            # For product pages, use Playwright to render JS (so we see prices)
            if "/products/" in url:
                html = _fetch_rendered_html(url)
                content_type = "text/html"
            else:
                resp = requests.get(url, timeout=15)
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    continue
                html = resp.text

            text = _clean_html_to_text(html)
            # Skip pages with almost no content (e.g., empty redirects, etc.)
            if not text or len(text) < 200:
                continue

            # Chunk + embed this page
            chunks = _chunk_text(text)
            if not chunks:
                continue
            embeddings = _embed(chunks)

            page_chunks = []
            for t, e in zip(chunks, embeddings):
                page_chunks.append({
                    "id": _sha1(t)[:16],
                    "text": t,
                    "embedding": e,
                })

            index[url] = {
                "url": url,
                "chunks": page_chunks,
            }

            # Discover more internal links on this page
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].split("#")[0].strip()
                if not href:
                    continue
                full = urljoin(url, href)
                parsed = urlparse(full)

                # Stay on the same domain as the start URL
                if parsed.netloc != start_parsed.netloc:
                    continue

                if full not in visited and full not in to_visit:
                    to_visit.append(full)

        except Exception as e:
            print(f"[crawl] Error fetching {url}: {e}")
            continue

    # Save the site index to disk
    with open(SITE_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)

    print(f"[crawl] Finished. Indexed {len(index)} pages.")
    return index


def _load_site_index_or_build(max_age_days: int = 3) -> dict:
    """
    Load site_index.json if it exists and is recent.
    If it's older than `max_age_days`, rebuild the index automatically.
    """
    if os.path.exists(SITE_INDEX_PATH):
        mtime = os.path.getmtime(SITE_INDEX_PATH)
        age_days = (time.time() - mtime) / 86400
        if age_days < max_age_days:
            # File is still fresh
            with open(SITE_INDEX_PATH, "r") as f:
                return json.load(f)
        else:
            print(f"[site_index] Index is {age_days:.1f} days old — rebuilding...")
            return crawl_and_build_site_index()
    else:
        print("[site_index] No index found — building fresh...")
        return crawl_and_build_site_index()


def _retrieve_from_site(email_text: str, k: int = 4) -> list[dict]:
    """
    Retrieve the top-k most relevant content chunks across the entire site
    based on cosine similarity.
    """
    index = _load_site_index_or_build()
    if not index:
        return []

    # Flatten all chunks from all URLs
    all_chunks = []
    for url, page in index.items():
        for ch in page.get("chunks", []):
            all_chunks.append({
                "url": url,
                "text": ch["text"],
                "embedding": ch["embedding"],
            })

    if not all_chunks:
        return []

    # Embed the question (email)
    q_emb = _embed([email_text])[0]
    qv = np.array(q_emb, dtype=np.float32)

    scored = []
    for ch in all_chunks:
        v = np.array(ch["embedding"], dtype=np.float32)
        denom = (np.linalg.norm(qv) * np.linalg.norm(v)) or 1e-8
        sim = float(np.dot(qv, v) / denom)
        scored.append((sim, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[:k]]
    return top
# --- end whole-site crawling and indexing ---







# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 
          'https://www.googleapis.com/auth/gmail.send', 
          'https://www.googleapis.com/auth/gmail.modify']
CLIENT_SECRET_PATH = '/Users/cooperburden/Desktop/NS/client_secret_867110489635-uusid9grhf9opo60ld7bpui257belgds.apps.googleusercontent.com.json'

# Set OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')




def get_service():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    # Build the service
    service = build('gmail', 'v1', credentials=creds)
    return service







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
def generate_response(email_body: str) -> str:
    """
    Generate a customer reply grounded in:
    - The live FAQ page (faq_index.json)
    - The full Native Scripture site (site_index.json)

    It:
    - Ensures the FAQ index exists / is up to date
    - Retrieves top FAQ and site snippets relevant to the email
    - Crafts a concise, policy-safe reply using those snippets as context
    """
    # 1) Make sure the FAQ index exists and is up-to-date (cheap if unchanged)
    try:
        _ = build_or_refresh_faq_index(force=False)
    except Exception as e:
        print(f"[FAQ index] non-fatal: {e}")

    # 2) Retrieve the most relevant FAQ chunks
    try:
        faq_chunks = _retrieve(email_body, k=3)
    except Exception as e:
        print(f"[FAQ retrieve] non-fatal: {e}")
        faq_chunks = []

    # 3) Retrieve the most relevant site-wide chunks
    try:
        site_chunks = _retrieve_from_site(email_body, k=3)
    except Exception as e:
        print(f"[Site retrieve] non-fatal: {e}")
        site_chunks = []

    faq_context = "\n\n---\n".join([c["text"] for c in faq_chunks]) if faq_chunks else ""
    site_context = "\n\n---\n".join(
        [f"From {c['url']}:\n{c['text']}" for c in site_chunks]
    ) if site_chunks else ""

    combined_context = ""
    if faq_context:
        combined_context += "FAQ context:\n" + faq_context
    if site_context:
        if combined_context:
            combined_context += "\n\n====================\n\n"
        combined_context += "Website context:\n" + site_context

    # 4) Compose grounded prompt
    system_rules = (
        "You are the customer support AI for Native Scripture (dual-language Book of Mormon). "
        "You must base your answers on the provided FAQ and website context snippets. "
        "If the FAQ and website conflict, prefer the FAQ for policies and guarantees. "
        "If the answer is not present in the context, say you’re not sure and direct the customer to the website or FAQ page. "
        "Be concise (2–5 sentences), warm, and helpful. Do not invent policies, prices, or dates."
    )

    user_msg = (
        f"Customer email:\n{email_body}\n\n"
        f"Relevant context from FAQ and website (use if relevant):\n{combined_context}\n"
        "If the context conflicts with prior knowledge, ALWAYS prefer this context."
    )

    # 5) Call OpenAI
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_rules},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=220,
            temperature=0.2,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as error:
        print(f"Error generating response: {error}")
        return "Sorry, I couldn’t generate a response right now."




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



## this is a list view of all of the emails, and a snippet of the draft that was written.
def list_drafts(service):
   try:
       # List drafts in the user's Gmail account
       results = service.users().messages().list(userId='me', labelIds=['DRAFT']).execute()
       messages = results.get('messages', [])
      
       drafts = []
       for message in messages:
           msg = service.users().messages().get(userId='me', id=message['id']).execute()
           drafts.append({
               'id': msg['id'],
               'subject': msg['payload']['headers'][0]['value'],  # Extract subject from headers
               'body': msg['snippet']  # Get a snippet of the body
           })
      
       return drafts


   except Exception as e:
       print(f"Error fetching drafts: {e}")
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





def process_unread_emails_once():
    """
    Convenience function: authenticate, read unread emails, generate
    AI responses grounded in the FAQ, and save them as drafts.
    Run this by calling: python email_ai_assistant.py
    """
    service = authenticate_gmail()
    if not service:
        print("Authentication failed.")
        return

    emails = list_messages(service)
    if not emails:
        print("No unread emails to process.")
        return

    for email in emails:
        subject = email.get("subject", "No Subject")
        body = email.get("body", "")
        print(f"\nProcessing email: {subject}")

        response = generate_response(body)
        print("AI response:\n", response)

        # Create a draft reply (not send)
        send_email(service, email["id"], response, sender_email="me")
        print("Draft created for this email.")




if __name__ == "__main__":
    process_unread_emails_once()










