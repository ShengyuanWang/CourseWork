import webbrowser
import argparse

# LINK_POS = "./links.txt"
# DOWNLOADS="./downloads.txt"
def parser_add_args(parser):
    parser.add_argument('--links', type=str, help='place of previews links')
    parser.add_argument('--store', type=str, help='place to store download links')

def read_txt(link_pos, output_pos):
    reader = open(link_pos)
    writer = open(output_pos, 'a')
    try:
        links = reader.readlines()
        for link in links:
            link_result = convert(link)
            writer.write(link_result+"\n")
    finally:
        reader.close()

def convert(preview_link):
    download_link = "https://drive.google.com/u/0/uc?id="
    p = preview_link[25:].strip().split("/")[2]
    download_link += p
    download_link += "&export=download"
    return download_link

def goToLinks(output_pos):
    reader = open(output_pos, 'r')
    lines = reader.readlines()
    chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
    for line in lines:
        webbrowser.get(chrome_path).open(line.strip("\n"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser_add_args(parser)
    args = parser.parse_args()
    LINK_POS=args.links
    DOWNLOADS=args.store
    read_txt(LINK_POS, DOWNLOADS)
    goToLinks(DOWNLOADS)