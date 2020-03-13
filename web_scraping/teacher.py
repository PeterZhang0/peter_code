# coding:utf-8
from pyquery import PyQuery as pq
from lxml import etree
import requests
import json
import codecs


URL = 'http://www.itcast.cn/channel/teacher.shtml'

headers = {
 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
 'Upgrade-Insecure-Requests': '1',
 'Accept-Language': 'zh-CN,zh;q=0.9',
 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
 'Accept-Encoding': 'gzip, deflate'
}

def get_page(url):
    try:
        response = requests.get(url=url, headers=headers)
        if response.status_code == 200:
            # return response.text
            return response.content.decode('utf-8')
    except Exception as f:
        print(f)

# 解析html
def parse_html(html):
    content = pq(html)
    res = content(".tea_con div ul li")
    #print(res)
    result = []
    for item in res:
        name = item.xpath('.//div[@class="li_txt"]/h3')[0].text
        level = item.xpath('.//div[@class="li_txt"]/h4')[0].text
        comment = item.xpath('.//div[@class="li_txt"]/p')[0].text
        res = {
            'name': name,
            'level': level,
            'comment': comment
        }
        result.append(res)
    return result

# 写入文件
def write_to_file(content):
    with codecs.open("teacher_result1.txt", "a", encoding='utf-8') as f:
        f.write(json.dumps(content, ensure_ascii=False) + '\t\n')

def main():
    res = get_page(URL)
    result = parse_html(res)
    for item in result:
        write_to_file(item)

main()

# if __name__ == '__main__':
#     main()