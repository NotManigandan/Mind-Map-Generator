from werkzeug.utils import secure_filename
from main import load_text
from main import create_keywords_from_text
from main import get_mindmap

def file_upload(file):
    filename = secure_filename(file.filename)
    print(filename)
    file.save(f'static/files/{filename}')
    final_text = load_text(f'static/files/{filename}')    
    # print("Length of text: ", len(final_text))
    keywords2, topics2 = create_keywords_from_text(final_text, max_nodes=5, sentence_group=4)
    plt = get_mindmap(keywords2, topics2)
    return plt