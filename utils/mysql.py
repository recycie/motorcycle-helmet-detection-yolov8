import mysql.connector

class Database:
    def __init__(self, h = "Localhost", u = "root", p = "", db = "helmet"):
      self.host = h
      self.username = u
      self.password = p
      self.database = db
    
    def connection(self):
      self.con = mysql.connector.connect(
          host=self.host,
          user=self.username,
          password=self.password,
          database=self.database
      )
      self.dbcon = self.con.cursor()

    def select(self, sql_command, val=(), retries=3):
        for attempt in range(retries):
            try:
                self.connection()
                self.dbcon.execute(sql_command, val)
                result = self.dbcon.fetchall()
                return result
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt + 1 == retries:
                    print(f"All {retries} attempts failed.")
                    return None

    def insert(self, sql_command, val, retries=3):
        for attempt in range(retries):
            try:
                self.connection()
                self.dbcon.execute(sql_command, val)
                self.con.commit()

                if self.dbcon.rowcount:
                    return True
                return False
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: sql: {e}")
                if attempt + 1 == retries:
                    print(f"All {retries} attempts failed.")
                    return False

    def insert_detection(self, id, motocycle_id, monitor_id, driver, helmet, helmet_score, person_score):
      sql_command = "INSERT INTO `detection`(`id`, `motorcycle_id`, `monitor_id`, `driver`, `helmet`, `helmet_score`, `person_score`) VALUES (%s, %s, %s, %s, %s, %s, %s)"
      val = (id, motocycle_id, monitor_id, driver, helmet, helmet_score, person_score)
      self.insert(sql_command, val)
        
    def insert_bbox(self, detection_hash, helmet_x1, helmet_y1, helmet_x2, helmet_y2, person_x1, person_y1, person_x2, person_y2):
      sql_command = "INSERT INTO `bbox`(`detection_id`, `helmet_x1`, `helmet_y1`, `helmet_x2`, `helmet_y2`, `person_x1`, `person_y1`, `person_x2`, `person_y2`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
      val = (detection_hash, helmet_x1, helmet_y1, helmet_x2, helmet_y2, person_x1, person_y1, person_x2, person_y2)
      self.insert(sql_command, val)

    def insert_source(self, name, url):
        sql_command = "INSERT INTO monitor (name, url) VALUES (%s, %s)"
        val = (name, url)
        self.insert(sql_command, val)
        
    def update_source(self, name, url, id):
        sql_command = "UPDATE monitor set name = %s, url = %s, status = %s WHERE id = %s"
        val = (name, url, id)
        try:
          self.connection()
          self.dbcon.execute(sql_command, val)
          self.con.commit()

          if self.dbcon.rowcount:
            return True
          return False
        except:
           return False
