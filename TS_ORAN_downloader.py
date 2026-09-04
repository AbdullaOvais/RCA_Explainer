import os
import time
import requests

# CONFIG
START_ID = 1
END_ID = 10000
SAVE_DIR = "oran_specs_all"
REQUEST_DELAY = 0.5  # seconds

USE_BROWSER_COOKIES = True
cookies = None

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115 Safari/537.36"
}

if USE_BROWSER_COOKIES:
    try:
        import browser_cookie3
        print("🔍 Trying Brave cookies...")
        cookies = browser_cookie3.brave(domain_name="specifications.o-ran.org")
        print("✅ Brave cookies loaded.")
    except Exception:
        try:
            print("⚠️ Brave failed. Trying Firefox...")
            cookies = browser_cookie3.firefox(domain_name="specifications.o-ran.org")
            print("✅ Firefox cookies loaded.")
        except Exception:
            print("❌ No browser cookies found. Use manual mode.")
            USE_BROWSER_COOKIES = False

if not USE_BROWSER_COOKIES:
    cookie_string = "INSERT_YOUR_COOKIE_STRING_HERE"
    cookies = dict([cookie.split("=", 1) for cookie in cookie_string.split("; ")])

os.makedirs(SAVE_DIR, exist_ok=True)

def get_filename_from_headers(headers, doc_id, content_type):
    disp = headers.get("Content-Disposition", "")
    if "filename=" in disp:
        filename = disp.split("filename=")[-1].strip("\"'")
        return filename
    # fallback if no filename is given
    ext = ".docx" if "word" in content_type else ".pdf" if "pdf" in content_type else ".bin"
    return f"oran_doc_{doc_id}{ext}"

def download_document(doc_id):
    url = f"https://specifications.o-ran.org/download?id={doc_id}"
    try:
        r = requests.get(url, cookies=cookies, headers=headers, stream=True, allow_redirects=True, timeout=10)

        content_type = r.headers.get("Content-Type", "")
        if not any(x in content_type for x in ["application/vnd.openxmlformats-officedocument", "application/pdf"]):
            print(f"[{doc_id}] ❌ Not a DOCX or PDF (type: {content_type})")
            return

        filename = get_filename_from_headers(r.headers, doc_id, content_type)
        filepath = os.path.join(SAVE_DIR, filename)

        with open(filepath, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

        print(f"[{doc_id}] ✅ Downloaded: {filename}")
    except Exception as e:
        print(f"[{doc_id}] ⚠️ Error: {e}")

def main():
    print(f"🚀 Starting ORAN download from ID {START_ID} to {END_ID}")
    for doc_id in range(START_ID, END_ID + 1):
        download_document(doc_id)
        time.sleep(REQUEST_DELAY)
    print("🎉 All done.")

if __name__ == "__main__":
    main()
