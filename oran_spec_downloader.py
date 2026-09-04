import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin
import re
import json

class ORANDownloader:
    def __init__(self, download_dir="oran_specifications"):
        self.base_url = "https://specifications.o-ran.org"
        self.specs_url = f"{self.base_url}/specifications"
        self.download_dir = download_dir
        self.session = requests.Session()
        
        # Create download directory if it doesn't exist
        os.makedirs(self.download_dir, exist_ok=True)
        
    def get_specification_ids_selenium(self):
        """Use Selenium to scrape JavaScript-rendered page"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            print("Using Selenium to fetch page with JavaScript...")
            
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(self.specs_url)
            
            # Wait for page to load
            time.sleep(5)
            
            # Get page source after JavaScript execution
            page_source = driver.page_source
            driver.quit()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Find all download links
            spec_ids = []
            
            # Look for download links in various patterns
            for link in soup.find_all('a', href=True):
                href = link['href']
                match = re.search(r'download\?id=(\d+)', href)
                if match and match.group(1) not in spec_ids:
                    spec_ids.append(match.group(1))
            
            # Also check onclick events and data attributes
            for element in soup.find_all(True):
                # Check onclick
                if element.get('onclick'):
                    match = re.search(r'download\?id=(\d+)', element['onclick'])
                    if match and match.group(1) not in spec_ids:
                        spec_ids.append(match.group(1))
                
                # Check data-id or similar attributes
                for attr in element.attrs:
                    if 'id' in attr.lower() and str(element[attr]).isdigit():
                        if element[attr] not in spec_ids:
                            spec_ids.append(str(element[attr]))
            
            print(f"Found {len(spec_ids)} specifications with Selenium")
            return spec_ids
            
        except ImportError:
            print("Selenium not installed. Run: pip install selenium")
            return []
        except Exception as e:
            print(f"Error with Selenium: {e}")
            return []
    
    def download_specification(self, spec_id):
        """Download a single specification by ID"""
        download_url = f"{self.base_url}/download?id={spec_id}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': self.specs_url
        }
        
        try:
            print(f"Downloading specification ID {spec_id}...")
            response = self.session.get(download_url, headers=headers, stream=True, allow_redirects=True)
            response.raise_for_status()
            
            # Try to get filename from Content-Disposition header
            filename = None
            if 'Content-Disposition' in response.headers:
                content_disp = response.headers['Content-Disposition']
                match = re.search(r'filename[^;=\n]*=((["\']).*?\2|[^;\n]*)', content_disp)
                if match:
                    filename = match.group(1).strip('"\'')
            
            # If no filename found, create one
            if not filename:
                content_type = response.headers.get('Content-Type', '')
                if 'pdf' in content_type:
                    ext = '.pdf'
                elif 'zip' in content_type:
                    ext = '.zip'
                elif 'word' in content_type or 'document' in content_type:
                    ext = '.docx'
                else:
                    ext = '.bin'
                filename = f"oran_spec_{spec_id}{ext}"
            
            filepath = os.path.join(self.download_dir, filename)
            
            # Download with progress
            file_size = int(response.headers.get('Content-Length', 0))
            
            with open(filepath, 'wb') as f:
                if file_size:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress = (downloaded / file_size) * 100
                            print(f"  Progress: {progress:.1f}%", end='\r')
                    print()
                else:
                    f.write(response.content)
            
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✓ Saved: {filename} ({file_size_mb:.2f} MB)")
            return True
            
        except Exception as e:
            print(f"  ✗ Error downloading spec {spec_id}: {e}")
            return False
    
    def download_all_selenium(self, delay=2):
        """Download all specifications using Selenium for discovery"""
        spec_ids = self.get_specification_ids_selenium()
        
        if not spec_ids:
            print("\nNo specifications found with Selenium.")
            print("Please use the manual method with specific IDs.")
            return
        
        print(f"\nStarting download of {len(spec_ids)} specifications...")
        print(f"Download directory: {os.path.abspath(self.download_dir)}\n")
        
        success_count = 0
        for i, spec_id in enumerate(spec_ids, 1):
            print(f"[{i}/{len(spec_ids)}] ", end='')
            if self.download_specification(spec_id):
                success_count += 1
            
            if i < len(spec_ids):
                time.sleep(delay)
        
        print(f"\n{'='*60}")
        print(f"Download complete: {success_count}/{len(spec_ids)} successful")
        print(f"Files saved to: {os.path.abspath(self.download_dir)}")
    
    def download_by_ids(self, spec_ids, delay=2):
        """Download specific specifications by their IDs"""
        print(f"Starting download of {len(spec_ids)} specifications...")
        print(f"Download directory: {os.path.abspath(self.download_dir)}\n")
        
        success_count = 0
        for i, spec_id in enumerate(spec_ids, 1):
            print(f"[{i}/{len(spec_ids)}] ", end='')
            if self.download_specification(str(spec_id)):
                success_count += 1
            
            if i < len(spec_ids):
                time.sleep(delay)
        
        print(f"\n{'='*60}")
        print(f"Download complete: {success_count}/{len(spec_ids)} successful")
        print(f"Files saved to: {os.path.abspath(self.download_dir)}")
    
    def download_range(self, start_id, end_id, delay=2):
        """Download specifications in a range of IDs"""
        spec_ids = list(range(start_id, end_id + 1))
        self.download_by_ids(spec_ids, delay)


def main():
    downloader = ORANDownloader(download_dir="oran_specifications")
    
    print("="*60)
    print("O-RAN Specifications Downloader")
    print("="*60)
    print("\nChoose a method:")
    print("1. Try Selenium (requires: pip install selenium)")
    print("2. Manual download with specific IDs")
    print("3. Download a range of IDs")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        downloader.download_all_selenium(delay=2)
    
    elif choice == "2":
        print("\nEnter specification IDs (comma-separated):")
        print("Example: 927,928,929")
        ids_input = input("IDs: ").strip()
        
        try:
            spec_ids = [int(id.strip()) for id in ids_input.split(',')]
            downloader.download_by_ids(spec_ids, delay=2)
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")
    
    elif choice == "3":
        print("\nEnter ID range:")
        try:
            start_id = int(input("Start ID: ").strip())
            end_id = int(input("End ID: ").strip())
            downloader.download_range(start_id, end_id, delay=2)
        except ValueError:
            print("Invalid input. Please enter valid numbers.")
    
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Alternative: Uncomment to use directly in code
    # downloader.download_by_ids([927, 928, 929], delay=2)
    # downloader.download_range(900, 1000, delay=2)


if __name__ == "__main__":
    main()