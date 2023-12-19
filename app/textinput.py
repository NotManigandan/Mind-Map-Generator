from main import create_keywords_from_text
from main import get_mindmap

def text_input(text_ip):
    final_text = text_ip
    # print("Length of text: ", len(final_text))
    keywords2, topics2 = create_keywords_from_text(final_text, max_nodes=5, sentence_group=4)
    # print(keywords2)
    plt = get_mindmap(keywords2, topics2)
    # plt.show()
    # plt.savefig("Graph.png", format="PNG")
    return plt


# text_input(eg)