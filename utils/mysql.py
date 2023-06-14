import mysql.connector

class helmetSQL:
    def __init__(self, h = "Localhost", u = "root", p = "", db = "helmet") -> None:
        self.CONN = mysql.connector.connect(
            host=h,
            user=u,
            password=p,
            database=db
        )

        self.exc = self.CONN.cursor()
    
    def insert(self, name, helmet, image):
        sql_command = "INSERT INTO helmet (name, helmet, image) VALUES (%s, %s, %s)"
        val = (name, helmet, image)
        try:
          self.exc.execute(sql_command, val)
          self.CONN.commit()

          if self.exc.rowcount:
            return True
          return False
        except:
           return None
