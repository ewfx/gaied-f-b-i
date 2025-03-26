import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import pipeline
import pytesseract
from PIL import Image
import email
import os
import re
from pdf2image import convert_from_path

class EmailProcessor:
    def __init__(self):
        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.classifier = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        self.ner_model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased")
        
        # Updated request types to match actual use cases
        self.request_types = [
            "Adjustment",
            "AU Transfer",
            "Closing Notice",
            "Commitment Change",
            "Fee Payment",
            "Money Movement-Inbound",
            "Money Movement-Outbound"
        ]
        
        # Additional attributes for financial notifications
        self.key_fields = {
            'facility_name': r'facility\s+(.*?)\s+have been',
            'global_balance': r'Global principal balance.*?USD\s+([\d,]+\.?\d*)',
            'share_amount': r'Your share.*?USD\s+([\d,]+\.?\d*)',
            'effective_date': r'Effective\s+(\d{1,2}-[A-Za-z]{3}-\d{4})',
            'aba_number': r'ABA.*?(\d{9})',
            'account_number': r'Account.*?(\d+)'
        }
        
        # Add priority configuration
        self.priority_indicators = {
            'urgent': 3,
            'asap': 3,
            'immediate': 3,
            'critical': 3,
            'important': 2,
            'priority': 2,
            'attention': 2,
            'review': 1,
            'update': 1
        }
        
        # Define request types and sub-types
        self.request_types = [
            "Loan Modification",
            "Payment Processing",
            "Document Request",
            "Account Update",
            "Information Inquiry"
        ]
        
        # Add email hash storage for duplicate detection
        self.processed_emails = set()
        
    def generate_email_hash(self, email_content, attributes):
        """Generate a unique hash for email based on content and key attributes"""
        key_elements = [
            email_content[:100],  # First 100 chars of content
            str(attributes.get('effective_date')),
            str(attributes.get('amount')),
            str(attributes.get('deal_name')),
            str(attributes.get('share_amount'))
        ]
        return hash(''.join(key_elements))

    def is_duplicate(self, email_content, attributes):
        """Check if this email has been processed before"""
        email_hash = self.generate_email_hash(email_content, attributes)
        if email_hash in self.processed_emails:
            return True
        self.processed_emails.add(email_hash)
        return False

    def extract_text_from_attachment(self, attachment_path):
        """Extract text from various attachment types (PDF, Images)"""
        file_ext = os.path.splitext(attachment_path)[1].lower()
        
        if file_ext in ['.pdf']:
            images = convert_from_path(attachment_path)
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image)
            return text
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff']:
            image = Image.open(attachment_path)
            return pytesseract.image_to_string(image)
        return ""

    def classify_email(self, email_content, attachments=[]):
        """Classify email into request type and sub-request type"""
        # Combine email content with extracted text from attachments
        full_text = email_content
        for attachment in attachments:
            attachment_text = self.extract_text_from_attachment(attachment)
            full_text += "\n" + attachment_text
        
        # Classify request type
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.classifier(**inputs)
        request_type = self.request_types[torch.argmax(outputs.logits)]
        
        return request_type

    def extract_key_attributes(self, text):
        """Extract key attributes from financial notifications"""
        attributes = {
            'deal_name': None,
            'amount': None,
            'effective_date': None,
            'transaction_type': None,
            'previous_balance': None,
            'new_balance': None,
            'share_amount': None,
            'payment_details': {
                'bank_name': None,
                'aba': None,
                'account': None
            }
        }
        
        # Extract deal/facility name
        deal_patterns = [
            r'Re:\s*(.*?)\s*(?:\d|$)',
            r'Deal NAME:\s*(.*?)\s*(?:\d|$)',
            r'Reference:\s*(.*?)\s*(?:\d|$)'
        ]
        for pattern in deal_patterns:
            match = re.search(pattern, text)
            if match:
                attributes['deal_name'] = match.group(1).strip()
                break

        # Extract amounts
        amount_matches = re.findall(r'USD\s*([\d,]+\.?\d*)', text)
        if amount_matches:
            attributes['amount'] = amount_matches[0]
            
        # Extract balances
        prev_balance = re.search(r'Previous.*balance.*USD\s*([\d,]+\.?\d*)', text)
        new_balance = re.search(r'New.*balance.*USD\s*([\d,]+\.?\d*)', text)
        if prev_balance:
            attributes['previous_balance'] = prev_balance.group(1)
        if new_balance:
            attributes['new_balance'] = new_balance.group(1)

        # Extract share amount
        share_match = re.search(r'Your share.*USD\s*([\d,]+\.?\d*)', text)
        if share_match:
            attributes['share_amount'] = share_match.group(1)

        # Extract effective date
        date_patterns = [
            r'Effective\s*(\d{1,2}-[A-Za-z]{3}-\d{4})',
            r'Date:\s*(\d{1,2}-[A-Za-z]{3}-\d{4})',
            r'Effective\s*(\d{2}/\d{2}/\d{4})'
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                attributes['effective_date'] = match.group(1)
                break

        # Extract payment details
        aba_match = re.search(r'ABA.*?[#:\s]*(\d{9})', text)
        account_match = re.search(r'Account.*?[#:\s]*([X\d]+)', text)
        bank_match = re.search(r'Bank Name:\s*(.*?)(?:\n|$)', text)
        
        if aba_match or account_match or bank_match:
            if aba_match:
                attributes['payment_details']['aba'] = aba_match.group(1)
            if account_match:
                attributes['payment_details']['account'] = account_match.group(1)
            if bank_match:
                attributes['payment_details']['bank_name'] = bank_match.group(1).strip()

        return attributes

    def classify_email(self, email_content, attachments=[]):
        """Classify email based on content patterns"""
        content_lower = email_content.lower()
        
        # Classification patterns
        patterns = {
            "Money Movement-Outbound": [
                r'elected to repay',
                r'will remit',
                r'payment.*outbound',
                r'repayment'
            ],
            "Money Movement-Inbound": [
                r'please fund',
                r'payment.*inbound',
                r'funding request'
            ],
            "Adjustment": [
                r'share.*adjustment',
                r'facility.*adjusted',
                r'commitment.*adjusted'
            ],
            "Commitment Change": [
                r'commitment.*change',
                r'facility.*amendment'
            ]
        }
        
        for request_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, content_lower):
                    return request_type
                    
        return "Information Inquiry"  # Default classification

    def determine_priority(self, email_content, attachments):
        """Determine email priority based on content and attachments"""
        priority_score = 0
        
        # Check content for priority keywords
        for keyword, weight in self.priority_indicators.items():
            if keyword in email_content.lower():
                priority_score += weight
        
        # Check attachments
        if len(attachments) > 2:
            priority_score += 1
            
        # Check for monetary amounts
        amounts = re.findall(r'\$[\d,]+\.?\d*', email_content)
        if amounts:
            max_amount = max([float(amt.replace('$','').replace(',','')) for amt in amounts])
            if max_amount > 1000000:
                priority_score += 2
            elif max_amount > 100000:
                priority_score += 1
        
        # Determine priority level
        if priority_score >= 3:
            return 'HIGH'
        elif priority_score >= 2:
            return 'MEDIUM'
        return 'LOW'

    def process_email(self, email_path, attachments_dir=None):
        """Main function to process email and its attachments"""
        with open(email_path, 'r') as f:
            msg = email.message_from_file(f)
        
        email_content = ""
        attachments = []
        
        # Process email body
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    email_content += part.get_payload()
        else:
            email_content = msg.get_payload()
            
        # Process attachments if directory provided
        if attachments_dir:
            for filename in os.listdir(attachments_dir):
                if filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tiff')):
                    attachments.append(os.path.join(attachments_dir, filename))
        
        # Classify email
        request_type = self.classify_email(email_content, attachments)
        
        # Extract attributes
        attributes = self.extract_key_attributes(email_content)
        
        # Check for duplicates
        is_duplicate = self.is_duplicate(email_content, attributes)
        
        # Continue with regular processing
        request_type = self.classify_email(email_content, attachments)
        priority = self.determine_priority(email_content, attachments)
        
        return {
            'request_type': request_type,
            'priority': priority,
            'attributes': attributes,
            'content': email_content,
            'processed_attachments': len(attachments),
            'is_duplicate': is_duplicate
        }

def get_resource_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_dir, 'code','src','resources', 'sample_emails')

def main():
    processor = EmailProcessor()
    email_dir = get_resource_path()
    
    print("Processing emails from directory...")
    for filename in os.listdir(email_dir):
        if filename.endswith('.eml'):
            print(f"\nProcessing: {filename}")
            result = processor.process_email(
                os.path.join(email_dir, filename)
            )
            
            print("-" * 50)
            print(f"From: {filename}")
            print(f"Request Type: {result['request_type']}")
            print(f"Priority Level: {result['priority']}")
            print(f"Extracted Attributes: {result['attributes']}")
            print(f"Is Duplicate: {result['is_duplicate']}")
            print(f"Attachments Processed: {result['processed_attachments']}")
            print("-" * 50)

if __name__ == "__main__":
    main()