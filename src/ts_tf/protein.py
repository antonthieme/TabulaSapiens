import requests

def fetch_uniprot_sequence(uniprot_id):
    """
    Fetch the amino acid sequence for a given UniProt ID.

    Args:
        uniprot_id (str): UniProt ID (e.g., "Q92886").

    Returns:
        str: Amino acid sequence.
    """
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_data = response.text
        # Extract the sequence from the FASTA format
        sequence = "".join(fasta_data.split("\n")[1:])  # Skip the header
        return sequence
    else:
        print(f"Error fetching sequence for UniProt ID {uniprot_id}: {response.status_code}")
        return None