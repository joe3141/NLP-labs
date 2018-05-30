# -*- coding: utf-8 -*-
import scrapy
from newspaper import Article
from nyt.items import NytItem
from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors import LinkExtractor

# sudo /usr/share/elasticsearch/bin/elasticsearch start

class NewyorkCrawlerSpider(CrawlSpider):
    name = 'newyorkcrawler'
    idx = 0
    allowed_domains = ['www.nytimes.com']
    start_urls = ['https://www.nytimes.com/section/world/europe']
    rules = (Rule(LinkExtractor(allow=[r'\d{4}/\d{2}/\d{2}/[^/]+']), callback="parse_item", follow=True),)
    
    def parse_item(self, response):
        self.log("Scraping: " + response.url)

        item = NytItem()
        item['url'] = response.url
        a = Article(response.url)
        # According to the source, this doesn't download anything (i.e. opens a connection), if input_html is not None
        a.download(input_html=response.text) 
        a.parse()
        item['title'] = a.title
        item['authors'] = a.authors
        item['body'] = a.text
        # TODO: add tags

        f = open('articles/%d-%s'%(self.idx, a.title), 'w+', encoding='utf8')
        f.writelines(a.authors)
        f.write("\n" + response.url + "\n")
        f.write(a.text)
        f.close()
        self.idx+=1
        return item