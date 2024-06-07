from gnews import GNews
import datetime
from datetime import timedelta
import newspaper
from tqdm import tqdm
import numpy as np

def make_gnews_reader(start=(2024,1,1), end=(2024,4,1)):
    google_news = GNews(language='en', country='US', start_date=None, 
                        end_date=None, max_results=200)
    google_news.start_date = start
    google_news.end_date = end
    return google_news

def get_articles(school, gnews_reader):
    gnews_articles = gnews_reader.get_news(f'{school} football')
    return gnews_articles

def gnews_to_article(gnews_articles, gnews_reader):
    return [gnews_reader.get_full_article(article['url']) for article in tqdm(gnews_articles, total=len(gnews_articles),leave=False)]

def articles_to_text(articles, gnews):   
    text_list = []
    for article in tqdm(articles, total=len(articles)):
        try:
            text_list.append(article.text)
        except:
            continue
    return text_list

for abbr, university in [("psu", "Penn State"),
                         ("msu", "Michigan State"),
                         ("ucla", "UCLA"),
                         ("usc", "USC"),
                         ("iowa", "Iowa"),
                         ("wisconsin", "Wisconsin")]:
    print("-"*50)
    print(university)
    print("-"*50)
    articles_all = []
    for week in range(1,11):
        print(f"Week {week}")
        start = datetime.datetime(2024,2,1)
        gnews_reader = make_gnews_reader(start=start, end=start+timedelta(6*week))
        gnews_articles = get_articles(f"{university} football", gnews_reader)
        articles = gnews_to_article(gnews_articles, gnews_reader)
        articles_all.extend(articles)
    
    articles_out = []
    for a in articles_all:
        try:
            articles_out.append({"title":a.title, "text":a.text, "url":a.canonical_link, "meta_data":a.meta_data})
        except:
            pass
    
    np.save(f"data/{abbr}_articles_out.npy", articles_out)