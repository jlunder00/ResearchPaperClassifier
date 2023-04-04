from bs4 import BeautifulSoup
import urllib.request
import json


def get_soup(url):

    fp = urllib.request.urlopen(url)
    page = fp.read().decode("utf8")
    fp.close()
    return BeautifulSoup(page)

def get_lists(soup):
    valid_lists = []
    lists = soup.findAll('ul')
    for l in lists:
        lists_to_add = l.findAll('li', class_=False)
        valid_lists.extend(lists_to_add)
    return valid_lists

def get_list_item_info(item, is_publisher):
    #Get info from an "a" tag
    a_tag = item.a
    if a_tag == None:
        return {'name': item.text,
                'url': '',
                'is_publisher': is_publisher}
    else:
        return {'name': a_tag.text,
                'url': a_tag.attrs['href'],
                'is_publisher': is_publisher}

def get_all_info(soup, is_publisher):
    info = []
    valid_lists_flattened = get_lists(soup)
    for item in valid_lists_flattened:
        info.append(get_list_item_info(item, is_publisher))
    return info

def save_info(info, fname):
    with open(fname, 'w') as fout:
        fout.write(json.dumps(info, indent=4))

def scrape_and_save_info(url, fname, is_publisher):
    soup = get_soup(url)
    info = get_all_info(soup, is_publisher)
    save_info(info, fname)

def main():
    scrape_and_save_info('https://beallslist.net/standalone-journals/', 'beals_standalone_journals.json', False)
    scrape_and_save_info('https://beallslist.net/', 'beals_publishers.json', True)

if __name__ == '__main__':
    main()






    

