import pymysql

# 1. connect mysql
conn = pymysql.connect(host="127.0.0.1", port=3306, user='root', passwd='wsy2003518', charset='utf8')
cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
