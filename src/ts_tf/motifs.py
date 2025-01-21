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
        list: Reformatted PFM as a 2D list (rows: A, C, G, T; columns: motif positions).
    """
    response = requests.get(matrix_url)
    if response.status_code == 200:
        data = response.json()
        if "pfm" in data and data["pfm"]:
            pfm_dict = data["pfm"]  # PFM as a dictionary
            pfm = [pfm_dict["A"], pfm_dict["C"], pfm_dict["G"], pfm_dict["T"]]
            print(f"Retrieved PFM for {matrix_url}: {pfm}")
            return pfm
        else:
            print(f"No PFM found for {matrix_url}")
            return None
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
    ppm = []
    for i in range(pfm.shape[0]):
        ppm.append(pfm[:][i] / np.sum(pfm[:][i], axis=0)) # Compute PPM for each position

    ppm = np.array(ppm)
    pwm = np.zeros(pfm.shape)
    for j in range(pfm.shape[0]):
        for k, base in enumerate(["A", "C", "G", "T"]):
            pwm[j][k] = np.log2((ppm[j][k] + 1e-3) / background_freq[base])  # Add pseudocount to avoid log(0)

    pwm = np.array(pwm)  # Transpose to match the format of the PFM

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


def save_metadata_to_csv(metadata_list, output_file):
    """
    Save motif metadata (including UniProt IDs) to a CSV file.

    Args:
        metadata_list (list): List of metadata dictionaries for motifs.
        output_file (str): Path to the output CSV file.
    """
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Write the header
        writer.writerow(["Matrix ID", "Gene Name", "UniProt ID", "Species", "Taxonomy ID"])
        
        for metadata in metadata_list:
            matrix_id = metadata.get("matrix_id", "Unknown")
            gene_name = metadata.get("name", "Unknown")
            uniprot_ids = ";".join(metadata.get("uniprot_ids", []))  # Join multiple UniProt IDs with ";"
            
            # Extract species information
            try:
                species_info = metadata.get("species", [{}])[0]
                species_name = species_info.get("name", "Unknown")
                tax_id = species_info.get("tax_id", "Unknown")
            except IndexError:
                print(f"Error extracting species information for {matrix_id}")
                species_name = "Unknown"
                tax_id = "Unknown"
            
            
            # Write the row
            writer.writerow([matrix_id, gene_name, uniprot_ids, species_name, tax_id])


def fetch_motif_metadata(motif_id):
    """
    Fetch metadata for a given motif from JASPAR.

    Args:
        motif_id (str): JASPAR motif ID (e.g., "MA0634.1").

    Returns:
        dict: Metadata for the motif, including species and associated genes.
    """
    url = f"https://jaspar.genereg.net/api/v1/matrix/{motif_id}/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"Error fetching metadata for motif {motif_id}: {response.status_code}")

def fetch_all_motif_metadata(motif_ids):
    """
    Retrieve metadata for all motifs.

    Args:
        motif_ids (list): List of JASPAR motif IDs.

    Returns:
        list: Metadata for all motifs.
    """
    n_successful = 0
    n_failed = 0

    metadata_list = []
    for motif_id in motif_ids:
        try:
            metadata = fetch_motif_metadata(motif_id)
            metadata_list.append(metadata)
            n_successful += 1
        except ValueError as e:
            n_failed += 1
            print(e)

    print(f"Successfully fetched metadata for {n_successful} motifs. Failed to fetch metadata for {n_failed} motifs.")
    return metadata_list