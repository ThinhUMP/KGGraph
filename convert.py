from bs4 import BeautifulSoup

def smi_to_text(smi_file_path, output_txt_path):
    with open(smi_file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Get all the <sync> tags
    syncs = soup.find_all('sync')

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for sync in syncs:
            text = sync.get_text(separator=' ', strip=True)
            if text:
                f.write(text + '\n')

# Example usage
smi_to_text('example.smi', 'output.txt')
