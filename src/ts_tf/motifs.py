import requests
import numpy as np
import csv

# Base URL for JASPAR API
BASE_URL = "http://jaspar.genereg.net/api/v1/matrix/"

def fetch_all_motifs(tax_group=None):
    """
    Retrieve all high-quality motifs from the JASPAR database, iterating through all pages.

    Args:
        tax_group (str, optional): Taxonomic group (e.g., "vertebrates").

    Returns:
        list: List of all high-quality motifs.
    """
    url = BASE_URL
    params = {
        "collection": "CORE",
        "quality": "high",
    }
    if tax_group:
        params["tax_group"] = tax_group

    all_motifs = []
    while url:
        print(f"Fetching motifs from: {url}")
        response = requests.get(url, params=params if url == BASE_URL else None)
        if response.status_code == 200:
            data = response.json()
            all_motifs.extend(data["results"])  # Add current page results to the list
            url = data["next"]  # Get the URL for the next page
        else:
            raise ValueError(f"Error fetching data: {response.status_code}")

    return all_motifs

def retrieve_pfm(matrix_url):
    """
    Retrieve the PFM for a specific motif using its matrix URL.

    Args:
        matrix_url (str): API URL for the motif matrix.

    Returns:
        list: PFM as a 2D list (rows: A, C, G, T; columns: motif positions).
    """
    response = requests.get(matrix_url)
    if response.status_code == 200:
        data = response.json()
        print(f"Retrieved PFM for {matrix_url}: {data.get('pfm', 'No PFM')}")
        return data["pfm"]
    else:
        raise ValueError(f"Error fetching PFM: {response.status_code}")

def compute_pwm(pfm, background_freq=None):
    """
    Compute the PWM from a given PFM.

    Args:
        pfm (list): PFM as a 2D list.
        background_freq (dict, optional): Background frequencies for A, C, G, T.
            Default is uniform distribution.

    Returns:
        np.array: PWM as a NumPy array.
    """
    if background_freq is None:
        background_freq = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}

    pfm = np.array(pfm)  # Convert to NumPy array
    pwm = []
    for i, base in enumerate(["A", "C", "G", "T"]):
        pwm.append(np.log2((pfm[i] + 1e-6) / background_freq[base]))  # Add pseudocount to avoid log(0)

    return np.array(pwm)

def save_to_csv(motifs, output_file):
    """
    Save motifs with their PFM and PWM to a CSV file.

    Args:
        motifs (list): List of motif dictionaries.
        output_file (str): Path to the output CSV file.
    """
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["Motif ID", "Name", "Collection", "Position", "A (PFM)", "C (PFM)", "G (PFM)", "T (PFM)",
                         "A (PWM)", "C (PWM)", "G (PWM)", "T (PWM)"])
        
        # Process each motif
        for motif in motifs:
            pfm = retrieve_pfm(motif["url"])  # Fetch PFM
            pwm = compute_pwm(pfm)           # Compute PWM
            pwm = pwm.tolist()               # Convert PWM to list
            
            # Write rows for each position in the motif
            for i in range(len(pfm[0])):
                writer.writerow([
                    motif["matrix_id"], motif["name"], motif["collection"],
                    i + 1, pfm[0][i], pfm[1][i], pfm[2][i], pfm[3][i],
                    pwm[0][i], pwm[1][i], pwm[2][i], pwm[3][i]
                ])