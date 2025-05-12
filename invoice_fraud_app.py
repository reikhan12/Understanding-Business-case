import streamlit as st
import re
from datetime import datetime
import pytesseract
from PIL import Image, ImageChops
import io
import pandas as pd
import hashlib
import numpy as np
from pdf2image import convert_from_bytes
import fitz  
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import json



# --- OCR Field Extraction ---
def extract_invoice_data(text, contract_start=None, contract_end=None):
    try:
        lines = text.splitlines()
        invoice = {}

        def get_next_nonempty(index):
            for i in range(index + 1, len(lines)):
                if lines[i].strip():
                    return lines[i].strip()
            return ""

        for i, line in enumerate(lines):
            st.text(f"[DEBUG] Line {i}: {line}")
            line_clean = line.strip().lower()
            combined_text = line + " " + get_next_nonempty(i)

            if "invoice no" in line_clean or "invoice number" in line_clean:
                invoice['invoice_number'] = re.search(r'\d+', combined_text).group(0)

            elif "supplier" in line_clean or "seller" in line_clean:
                invoice['supplier_name'] = get_next_nonempty(i)

            elif "invoice date" in combined_text.lower() or "date of issue" in combined_text.lower() or "issue date" in combined_text.lower() or "date:" in combined_text.lower():
                date_match = None
                date_match = re.search(r'(\d{1,2}[./-]\d{1,2}[./-]\d{4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})', combined_text)
                if not date_match:
                    for j in range(i + 1, len(lines)):
                        future_line = lines[j].strip()
                        if any(keyword in future_line.lower() for keyword in ["tax id", "iban", "client", "total", "description", "qty", "summary", "items"]):
                            continue
                        date_match = re.search(r'(\d{1,2}[./-]\d{1,2}[./-]\d{4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})', future_line)
                        if date_match:
                            break
                if date_match:
                    invoice['invoice_date'] = date_match.group(1)

            elif "service period" in line_clean:
                date_match = re.search(r'(\d{1,2}[./-]\d{1,2}[./-]\d{4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})', combined_text.strip())
                if date_match:
                    invoice['service_date'] = date_match.group(1)

            elif "due date" in line_clean:
                date_match = re.search(r'(\d{1,2}[./-]\d{1,2}[./-]\d{4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})', combined_text.strip())
                if date_match:
                    invoice['due_date'] = date_match.group(1)

            elif "tax id" in line_clean and 'tax_id' not in invoice:
                invoice['tax_id'] = re.search(r'[\d-]+', combined_text).group(0)

            elif "bank account" in line_clean or "iban" in line_clean:
                bank_match = re.search(r'[A-Z]{2}\d{2}[A-Z0-9]{10,30}', combined_text)
                if bank_match:
                    invoice['bank_account'] = bank_match.group(0)

            elif "total amount" in line_clean or "total" in line_clean:
                total_match = re.search(r'\$?\s?(\d{1,3}(?:[.,]\d{2})?)', line)
                if not total_match:
                    for j in range(i+1, min(i+4, len(lines))):
                        future_line = lines[j]
                        total_match = re.search(r'\$?\s?(\d{1,3}(?:[.,]\d{2})?)', future_line)
                        if total_match:
                            break
                if total_match:
                    invoice['total_amount'] = float(total_match.group(1).replace(",", "."))

        invoice['line_items'] = []
        current_item = {}
        net_prices = []
        quantities = []
        collecting_prices = False

        for i, line in enumerate(lines):
            line = line.strip()
            if re.match(r'^\d+\.\s', line):
                if current_item:
                    invoice['line_items'].append(current_item)
                match = re.match(r'^\d+\.\s(.+?)\s(\d{1,3}[.,]\d{2})$', line)
                if match:
                    desc = match.group(1).strip()
                    qty = float(match.group(2).replace(",", "."))
                    current_item = {
                        'description': desc,
                        'quantity': qty,
                        'unit_price': None
                    }
                else:
                    desc = line.split('.', 1)[1].strip()
                    qty_match = re.search(r'(\d{1,3}[.,]\d{2})', desc)
                    qty = float(qty_match.group(1).replace(',', '.')) if qty_match else 1
                    desc = re.sub(r'(\d{1,3}[.,]\d{2})', '', desc).strip()
                    current_item = {
                        'description': desc,
                        'quantity': qty,
                        'unit_price': None
                    }
            elif current_item and not re.search(r'\d{1,3}[.,]\d{2}', line) and not any(kw in line.lower() for kw in ["summary", "vat", "net price", "gross", "client", "$"]):
                current_item['description'] += " " + line.strip()
            elif "net price" in line.lower():
                collecting_prices = True
            elif collecting_prices and re.match(r'^\d{1,3}[.,]\d{2}$', line):
                net_prices.append(float(line.replace(',', '.')))
            elif current_item and ("summary" in line.lower() or i == len(lines) - 1):
                invoice['line_items'].append(current_item)
                current_item = {}

        if current_item:
            invoice['line_items'].append(current_item)

        for idx, item in enumerate(invoice['line_items']):
            if idx < len(net_prices):
                item['unit_price'] = net_prices[idx]

        required_fields = ['invoice_number', 'supplier_name', 'invoice_date', 'tax_id', 'bank_account', 'total_amount']
        if all(field in invoice for field in required_fields):
            issues = []
            if contract_start and contract_end:
                invoice_date_obj = datetime.strptime(invoice['invoice_date'], "%m/%d/%Y")
                if not (contract_start <= invoice_date_obj.date() <= contract_end):
                    issues.append("Invoice date is outside the contract period.")
            for item in invoice['line_items']:
                if item['unit_price'] and item['unit_price'] > 100:
                    issues.append(f"High unit price detected: {item['unit_price']} for {item['description']}")

            if invoice['line_items']:
                df_items = pd.DataFrame(invoice['line_items'])
                model = IsolationForest(contamination=0.25, random_state=42)
                df_numeric = df_items[['unit_price', 'quantity']].dropna()
                if not df_numeric.empty:
                    model.fit(df_numeric)
                    df_items['anomaly_score'] = model.decision_function(df_numeric)
                    df_items['is_anomaly'] = model.predict(df_numeric)
                    anomalies = df_items[df_items['is_anomaly'] == -1]
                    if not anomalies.empty:
                        invoice['anomalous_items'] = anomalies.to_dict(orient='records')
                        issues.append("Potential fraud/anomalies detected via machine learning model.")

                        # --- Visualization ---
                        fig, ax = plt.subplots()
                        ax.scatter(df_items['unit_price'], df_items['quantity'], c=df_items['is_anomaly'], cmap='coolwarm', s=100)
                        ax.set_xlabel("Unit Price")
                        ax.set_ylabel("Quantity")
                        ax.set_title("Anomaly Detection on Line Items")
                        st.pyplot(fig)

            if issues:
                invoice['warnings'] = issues
            return invoice
        else:
            st.warning("Missing required fields: " + ", ".join([f for f in required_fields if f not in invoice]))
            return None
    except Exception as e:
        st.error(f"Error extracting invoice fields: {e}")
        return None






# --- Explanation Helper ---
def generate_anomaly_explanation(item):
    if item['unit_price'] > 100:
        return "This item has a high unit price which may be unusual for this type of product."
    elif item['quantity'] > 10:
        return "High quantity detected, might be bulk order or mis-entry."
    else:
        return "Anomaly detected based on unit price and quantity patterns."

# --- Streamlit App Entry Point ---
st.title("🧾 Invoice Fraud Detection App")

uploaded_file = st.file_uploader("Upload Invoice (PDF, Image, or Text)", type=["txt", "png", "jpg", "jpeg", "pdf"])
contract_start = st.date_input("Contract Start Date")
contract_end = st.date_input("Contract End Date")

invoice_text = ""
image = None

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_type = uploaded_file.name.lower()

    if file_type.endswith(".txt"):
        invoice_text = file_bytes.decode("utf-8")
    elif file_type.endswith(".pdf"):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            invoice_text = "\n".join([page.get_text() for page in doc])
            if not invoice_text.strip():
                images = convert_from_bytes(file_bytes)
                image = images[0]
                invoice_text = pytesseract.image_to_string(image)
        except:
            st.error("Could not process PDF.")
    else:
        image = Image.open(io.BytesIO(file_bytes))
        invoice_text = pytesseract.image_to_string(image)

    st.subheader("Extracted Text")
    st.text(invoice_text)

    if contract_start >= contract_end:
        st.warning("⚠️ Please ensure contract end date is after the start date.")

    invoice = extract_invoice_data(invoice_text, contract_start, contract_end)

    if invoice:
        st.subheader("Extracted Invoice Data")
        st.json(invoice)

        if 'anomalous_items' in invoice:
            st.subheader("Detected Anomalies")
            for idx, item in enumerate(invoice["anomalous_items"]):
                st.markdown(f"**Item {idx+1}: {item['description']}**")
                st.write(f"- Quantity: {item['quantity']}")
                st.write(f"- Unit Price: {item['unit_price']}")
                st.write(f"- Anomaly Score: {item['anomaly_score']:.4f}")
                explanation = generate_anomaly_explanation(item)
                st.info(f"📌 Explanation: {explanation}")

        if 'warnings' in invoice:
            st.subheader("Warnings")
            for warning in invoice['warnings']:
                st.warning(warning)

        # Optional: download button
        if st.button("📥 Download Extracted Data (JSON)"):
            st.download_button(
                label="Download JSON",
                data=json.dumps(invoice, indent=2),
                file_name="invoice_data.json",
                mime="application/json"
            )

    else:
        st.error("Failed to extract invoice fields. Please check format or OCR quality.")
else:
    st.info("Please upload an invoice to begin.")
