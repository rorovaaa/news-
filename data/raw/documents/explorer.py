

def list_documents():
    doc_dir = "data/raw/documents/"
    files = os.listdir(doc_dir)
    print(len(files))
    for file in files:
        print(f"  - {file}")
    return files

def file_checker():
    with open('data/raw/documents/py.txt', 'r', encoding='utf-8') as file:
        text = file.read()
        print(text)
    return text

def csv_check():
    df = pd.read_csv('data/raw/documents/prompts.csv')
    print(df.head)
    return df

if __name__ == "__main__":
    files = list_documents()
    text = file_checker()
    df = csv_check()
