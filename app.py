import os
import requests
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin


def download_pdf(k_number, base_url, output_dir):
    """
    Download a PDF from the FDA database using the K number.

    Args:
        k_number (str): The K number identifier (e.g., 'K241380')
        base_url (str): Base URL for FDA database
        output_dir (str): Directory to save the downloaded PDFs

    Returns:
        tuple: (k_number, success status, filepath or error message)
    """
    # Format the K number to ensure proper format (K followed by 6 digits)
    k_number = k_number.strip().upper()
    if not k_number.startswith("K"):
        k_number = f"K{k_number}"

    # Construct the URL using the pattern
    year = k_number[1:3]  # Extract the year (first 2 digits after K)
    url = urljoin(base_url, f"pdf{year}/{k_number}.pdf")

    output_path = os.path.join(output_dir, f"{k_number}.pdf")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses

        # Save the PDF
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return k_number, True, output_path
    except requests.exceptions.RequestException as e:
        return k_number, False, str(e)


def download_fda_pdfs(k_numbers, output_dir="fda_pdfs", max_workers=5):
    """
    Download multiple PDFs from the FDA database using K numbers.

    Args:
        k_numbers (list): List of K number identifiers
        output_dir (str): Directory to save the downloaded PDFs
        max_workers (int): Maximum number of concurrent downloads

    Returns:
        dict: Results of the download operations
    """
    base_url = "https://www.accessdata.fda.gov/cdrh_docs/"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    results = {"successful": [], "failed": []}

    # Use ThreadPoolExecutor for concurrent downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_k = {
            executor.submit(download_pdf, k, base_url, output_dir): k for k in k_numbers
        }

        for future in future_to_k:
            k_number, success, result = future.result()
            if success:
                results["successful"].append({"k_number": k_number, "filepath": result})
            else:
                results["failed"].append({"k_number": k_number, "error": result})

    # Print summary
    print(
        f"Download complete: {len(results['successful'])} successful, {len(results['failed'])} failed"
    )

    return results


# Example usage
if __name__ == "__main__":
    # List of K numbers to download
    k_numbers = [
        "K241380",
        # Add more K numbers here
    ]

    results = download_fda_pdfs(k_numbers)

    # Print results
    print("\nSuccessful downloads:")
    for item in results["successful"]:
        print(f"  - {item['k_number']}: {item['filepath']}")

    print("\nFailed downloads:")
    for item in results["failed"]:
        print(f"  - {item['k_number']}: {item['error']}")
