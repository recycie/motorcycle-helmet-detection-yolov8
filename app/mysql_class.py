from mysql.connector import pooling, Error

class Database:
    def __init__(self, h="localhost", u="root", p="", db="helmet", pool_name="mypool", pool_size=10):
        self.host = h
        self.username = u
        self.password = p
        self.database = db
        self.pool_name = pool_name
        self.pool_size = pool_size

        # Initialize connection pool
        self.pool = pooling.MySQLConnectionPool(
            pool_name=self.pool_name,
            pool_size=self.pool_size,
            pool_reset_session=True,
            host=self.host,
            user=self.username,
            password=self.password,
            database=self.database
        )

    def get_connection(self):
        try:
            return self.pool.get_connection()
        except Error as e:
            print(f'Error getting connection: {e}')
            return None

    def execute_query(self, query, params=(), fetch=False):
        connection = self.get_connection()
        if not connection:
            return None

        cursor = connection.cursor()
        try:
            cursor.execute(query, params)
            if fetch:
                result = cursor.fetchall()
                return result
            connection.commit()
            return True
        except Error as e:
            print(f'Error executing query: {e}')
            return None if fetch else False
        finally:
            cursor.close()
            connection.close()

    def select(self, sql_command, val=()):
        return self.execute_query(sql_command, val, fetch=True)

    def insert(self, sql_command, val):
        return self.execute_query(sql_command, val)