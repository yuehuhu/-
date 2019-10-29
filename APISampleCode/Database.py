import pymysql


# 连接数据库
def conn_mysql():
    conn = pymysql.connect(host='10.141.221.89', port=3306, user='seuser', password='seuser', database='se_database',
                           charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    cur = conn.cursor()
    return conn, cur


# 插入数据库
def insert_mysql(sql):
    conn, cur = conn_mysql()
    cur.execute(sql)
    conn.commit()



# 查询数据库
def find_mysql(sql):
    conn, cur = conn_mysql()
    cur.execute(sql)
    result = cur.fetchall()
    return result
