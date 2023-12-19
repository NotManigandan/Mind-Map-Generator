from main import scrape_data, clean_text
from main import create_keywords_from_text
from main import get_mindmap

def web_scrape(web_link):
    scraped_data = scrape_data(web_link)
    final_text = clean_text(scraped_data)
    # print("Length of text: ", len(final_text))
    keywords2, topics2 = create_keywords_from_text(final_text, max_nodes=3, sentence_group=7)
    plt = get_mindmap(keywords2, topics2)
    return plt
