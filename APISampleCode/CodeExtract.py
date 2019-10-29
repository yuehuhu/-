import json
import Database as db
import pymysql
import re
import sys
import os
from bs4 import BeautifulSoup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')  # the output dir


def main():
    htmls = list()
    qualified_names = list()
    # 查询数据库
    sql = "SELECT api1.qualified_name, api2.html\
    FROM se_database.jdk_all_api_entity as api1,\
    se_database.java_api_html_text as api2\
    where api1.id=api2.api_id and api2.html_type=3\
    and api2.html like '%<pre><code>%'"
    results = db.find_mysql(sql)

    save_file = []
    save_path = "SOSampleCode.json"

    # 将教程中的所需字段加载到内存
    for result in results:
        qualified_names.append(result['qualified_name'])
        htmls.append(result['html'])

    for i in range(len(htmls)):  # 找到所有<pre><code>……</pre></code>标签的内容
        html = htmls[i]
        qualified_name = qualified_names[i]
        soup_html = BeautifulSoup(html, 'html.parser')
        pre_codes = soup_html.findAll("pre")
        for pre_code in pre_codes:
            num = i
            API = qualified_name
            Code = pre_code.get_text()
            # 获取样例代码上下文各一段文本描述
            pre_Description = pre_code.find_previous_sibling().get_text()
            next_Description = pre_code.find_next_sibling().get_text()
            Description = pre_Description + ' ' + next_Description
            print("-------------------------------------------")
            print(Code)
            # 将全限定名，样例代码，文本描述保存
            json_save = {}
            json_save['NUM'] = num
            json_save['API'] = API
            json_save['Code'] = Code
            json_save['Description'] = Description
            save_file.append(json_save)

        # print(i)
        # i = i + 1
        # Code = pre_code.get_text()
        # print(Code)

    with open(OUTPUT_DIR + '/' + save_path, 'w', encoding='utf-8') as json_file:
        json.dump(save_file, json_file, indent=4)


if __name__ == '__main__':
    main()
