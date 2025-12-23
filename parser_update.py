from bs4 import BeautifulSoup
import requests
import json
from datetime import datetime, timedelta
import re
import os

class NewsParser:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.recent_threshold = datetime.now() - timedelta(hours=24)
        self.save_folder = "data/raw/documents"
        
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
    
    def parse_date(self, date_str, source):
        try:
            if source == 1:
                timestamp_match = re.search(r'\d{10}', date_str or '')
                if timestamp_match:
                    return datetime.fromtimestamp(int(timestamp_match.group()))
            elif source == 2:
                if date_str:
                    return datetime.fromtimestamp(date_str)
            elif source == 3:
                date_str_lower = date_str.lower()
                if 'сегодня' in date_str_lower:
                    base_date = datetime.now()
                elif 'вчера' in date_str_lower:
                    base_date = datetime.now() - timedelta(days=1)
                else:
                    for fmt in ('%d.%m.%Y', '%Y-%m-%d'):
                        try:
                            return datetime.strptime(date_str.split()[0], fmt)
                        except:
                            continue
                    return None
                
                time_match = re.search(r'(\d{1,2}):(\d{2})', date_str)
                if time_match:
                    hour, minute = map(int, time_match.groups())
                    return base_date.replace(hour=hour, minute=minute, second=0)
        except Exception:
            pass
        return None
    
    def get_article_content(self, url):
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for element in soup.find_all(['script', 'style', 'nav', 'footer', 'aside']):
                    element.decompose()
                
                content = ''
                selectors = ['article', '.article__content', '.article-text', '.content', '.news-text', '.text', 'main']
                
                for selector in selectors:
                    elements = soup.select(selector)
                    if elements:
                        content = ' '.join([elem.get_text(strip=True) for elem in elements])
                        if len(content) > 200:
                            break
                
                if not content or len(content) < 200:
                    paragraphs = soup.find_all('p')
                    content = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
                
                return content
        except Exception:
            pass
        return None
    
    def search_news(self):
        zapros = input("Запрос: ")
        print("1 - РИА Новости\n2 - Lenta.ru\n3 - РБК")
        
        news_sources = {
            1: 'https://ria.ru/search/?query=' + zapros,
            2: 'https://lenta.ru/search/v2/process?from=0&size=50&sort=2&title_only=0&domain=1&query=' + zapros,
            3: 'https://www.rbc.ru/search/?query=' + zapros
        }
        
        try:
            a = int(input("Ресурс: "))
            if a not in news_sources:
                return None, None
            
            response = requests.get(news_sources[a], headers=self.headers, timeout=10)
            if response.status_code == 200:
                if a == 2:
                    return response.json(), a
                else:
                    return response.text, a
        except Exception:
            pass
        return None, None
    
    def parse_news(self, result, resource):
        all_news = []
        
        if resource == 1:
            soup = BeautifulSoup(result, 'html.parser')
            news_items = soup.find_all(['a', 'div'], class_=lambda x: x and any(cls in x for cls in ['list-item', 'search-item', 'cell']))
            
            for item in news_items:
                try:
                    title_elem = item.find(class_=lambda x: x and 'title' in x.lower())
                    title = title_elem.get_text(strip=True) if title_elem else None
                    link = item.get('href')
                    if link and not link.startswith('http'):
                        link = 'https://ria.ru' + link
                    
                    date_elem = item.find(class_=lambda x: x and any(word in x.lower() for word in ['date', 'time', 'timestamp']))
                    date_str = date_elem.get_text(strip=True) if date_elem else None
                    date_obj = self.parse_date(date_str, resource)
                    
                    if title and link:
                        all_news.append({'title': title, 'link': link, 'date': date_obj, 'date_str': date_str, 'source': 'РИА Новости'})
                except Exception:
                    continue
        
        elif resource == 2:
            if 'matches' in result:
                for item in result['matches']:
                    try:
                        title = item.get('title')
                        url = item.get('url')
                        pubdate = item.get('pubdate')
                        
                        if title and url:
                            date_obj = datetime.fromtimestamp(pubdate) if pubdate else None
                            if date_obj and date_obj >= self.recent_threshold:
                                all_news.append({
                                    'title': title,
                                    'link': 'https://lenta.ru' + url if url.startswith('/') else url,
                                    'date': date_obj,
                                    'date_str': date_obj.strftime('%d.%m.%Y %H:%M') if date_obj else None,
                                    'source': 'Lenta.ru'
                                })
                    except Exception:
                        continue
        
        elif resource == 3:
            soup = BeautifulSoup(result, 'html.parser')
            news_blocks = soup.find_all(class_=lambda x: x and 'search-item' in x)
            
            for block in news_blocks:
                try:
                    title_elem = block.find('a') or block.find(class_=lambda x: x and 'title' in x.lower())
                    title = title_elem.get_text(strip=True) if title_elem else None
                    link = title_elem.get('href') if title_elem else None
                    
                    date_elem = block.find(class_=lambda x: x and any(word in x.lower() for word in ['date', 'time']))
                    date_str = date_elem.get_text(strip=True) if date_elem else None
                    date_obj = self.parse_date(date_str, resource)
                    
                    if title and link:
                        all_news.append({
                            'title': title,
                            'link': link if link.startswith('http') else 'https://www.rbc.ru' + link,
                            'date': date_obj,
                            'date_str': date_str,
                            'source': 'РБК'
                        })
                except Exception:
                    continue
        
        recent_news = []
        for news in all_news:
            if news['date'] and news['date'] >= self.recent_threshold:
                recent_news.append(news)
            elif not news['date']:
                recent_news.append(news)
        
        return recent_news
    
    def display_news(self, news_list):
        if not news_list:
            print("Новости не найдены")
            return
        
        print(f"Найдено свежих новостей: {len(news_list)}")
        
        for i, news in enumerate(news_list, 1):
            date_info = news['date_str'] if news['date_str'] else 'Дата не указана'
            print(f"{i}. [{news['source']}] {date_info}")
            print(f"   {news['title']}")
        
        try:
            choices = input("Номера для скачивания (через пробел), 0 - выход: ")
            if choices == '0':
                return
            
            selected_indices = [int(idx) for idx in choices.split() if idx.isdigit() and 1 <= int(idx) <= len(news_list)]
            
            for idx in selected_indices:
                selected = news_list[idx-1]
                content = self.get_article_content(selected['link'])
                if content:
                    self.save_to_file(selected, content)
                    print(f"Сохранено: {idx}. {selected['title'][:50]}...")
                else:
                    print(f"Ошибка: {idx}. {selected['title'][:50]}...")
        except Exception:
            pass

    def save_to_file(self, news, content):
        try:
            filename = f"{self.save_folder}/{news['source']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Заголовок: {news['title']}\n")
                f.write(f"Источник: {news['source']}\n")
                f.write(f"Дата: {news['date_str']}\n")
                f.write(f"Ссылка: {news['link']}\n\n")
                f.write(f"{'='*50}\n\n")
                f.write(content)
        except Exception:
            pass

def main():
    parser = NewsParser()
    result, resource = parser.search_news()
    
    if result and resource:
        news_list = parser.parse_news(result, resource)
        parser.display_news(news_list)

if __name__ == "__main__":
    main()
