import pymysql
from pymysql.cursors import DictCursor


connection = pymysql.connect(
    host='localhost',
    user='root',
    password='msql',
    db='usersdb',
    charset='utf8mb4',
    cursorclass=DictCursor
)

with connection.cursor() as cursor:

    sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"
    cursor.execute(sql, ('11111', '11112'))

with connection.cursor() as cursor:

    sql = "SELECT * FROM users;"
    cursor.execute(sql)
    result = cursor.fetchone()
    print(result)

connection.commit()
connection.close()