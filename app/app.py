from mysql_class import Database
from flask import render_template, jsonify, Flask, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from urllib.parse import urlparse
import enum
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import hashlib

DB_HOST = "localhost"
DB_USERNAME = "root"
DB_PASSWORD = ""
DB_NAME = "helmet"

NOREDIRECT = ['/login', '/monitor', '/users']
SESSION = {}

app = Flask(__name__, static_folder="assets")
app.config['VERSION'] = '1.0.0'
app.secret_key = 'keytest'
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class UserStatus(enum.Enum):
    active = '<span class="bg-green-100 text-green-800 text-xs font-medium me-2 px-2.5 py-0.5 rounded">Active</span>'
    locked = '<span class="bg-red-100 text-red-800 text-xs font-medium me-2 px-2.5 py-0.5 rounded">Locked</span>'
    closed = '<span class="bg-gray-100 text-gray-800 text-xs font-medium me-2 px-2.5 py-0.5 rounded">Closed</span>'
    suspended = '<span class="bg-red-100 text-red-800 text-xs font-medium me-2 px-2.5 py-0.5 rounded">Suspended</span>'

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(32), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(16), nullable=False, default='user')
    status = db.Column(db.Enum(UserStatus), nullable=False, default=UserStatus.active)
    modify_at = db.Column(db.DateTime, nullable=True, onupdate=func.now())
    created_at = db.Column(db.DateTime, nullable=False, default=func.now())

    @property
    def status_str(self):
        return self.status.value
    
    @status_str.setter
    def status_str(self, value):
        self.status = UserStatus(value)

def urlParse(url):
    parsed_url = urlparse(url)

    protocol = parsed_url.scheme
    host = parsed_url.netloc
    port = parsed_url.port

    print(parsed_url, url)

    if port is None:
        port = 80

    return f"{protocol}://{host}:{port}"

def hash_password_sha256(password: str) -> str:
    sha256 = hashlib.sha256()
    sha256.update(password.encode('utf-8'))
    return sha256.hexdigest()

def verify_password_sha256(stored_hash: str, password: str) -> bool:
    password_hash = hash_password_sha256(password)
    return password_hash == stored_hash

@login_manager.user_loader
def load_user(user_id):
    with app.app_context():
        session = sessionmaker(bind=db.engine)()
        return session.get(User, int(user_id))

def xRegister(username, password, role='user'):
    if not username or not password:
        return jsonify({'status': 'error', 'msg': 'Username and password are required.'}), 400

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({'status': 'error', 'msg': 'Username is already taken.'}), 400

    hashed_password = hash_password_sha256(password)

    new_user = User(username=username, password=hashed_password, role=role)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'status': 'success', 'msg': 'Added New User.'}), 201

@app.route('/api/v1/register', methods=['POST'])
@login_required
def register():
    data = request.get_json()
    
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'user')

    return xRegister(username, password, role)

@app.route('/api/v1/get_user', methods=['POST'])
@login_required
def get_user():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        user = User.query.get_or_404(user_id)
        return jsonify({
            'id': user.id,
            'username': user.username,
            'role': user.role,
            'status': user.status.name
        })
    except SQLAlchemyError as e:
        return jsonify({'status': 'error', 'msg': str(e)}), 500

@app.route('/api/v1/update_user', methods=['POST'])
@login_required
def update_user():
    try:
        data = request.get_json()

        user = User.query.get_or_404(data['userId'])
        if len(data.get('password')) != 0 and len(data.get('password')) >= 6:
            user.password = hash_password_sha256(data['password'])
        user.role = data.get('role', user.role)
        user.status = data.get('status', user.status)

        db.session.commit()
        return jsonify({'status': 'success', 'msg': 'Updated Successfully'}), 200
    except SQLAlchemyError as e:
        return jsonify({'status': 'error', 'msg': str(e)}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        user = User.query.filter_by(username=username).first()

        if user:
            if user.status != UserStatus.active:
                return jsonify({'status': 'error', 'msg': 'Your account is not active. Please contact support.'})

            if verify_password_sha256(user.password, password):
                login_user(user)
                return jsonify({'status': 'success', 'msg': 'Login successfully'})

        return jsonify({'status': 'error', 'msg': 'Username or Password Invalid.'})
    return render_template("index.html", page='login', sources={}, serverselect={})

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

def validationSource():
    global serverselect
    if serverselect:
        serverSQL = DATABASE.select('SELECT * FROM source WHERE id = %s AND name = %s', (serverselect[0], serverselect[1]))
        if serverSQL and len(serverSQL) != 0:
            serverselect = serverSQL[0]
        else:
            serverselect = {}

def loadUserList(page):
    user_list = []
    if page == 'users' and current_user.role == 'admin':
        users = User.query.all()
        user_list = [
            {
                'id': user.id,
                'username': user.username,
                'password': user.password,
                'role': user.role,
                'accountstatus': user.status
            }
            for user in users
        ]
    return user_list

@app.route("/dashboard")
@app.route("/live")
@app.route("/monitor")
@app.route("/reports")
@app.route("/users")
@app.route("/")
@login_required
def index():
    global SESSION

    page = (request.path).replace('/', '')

    sourceSQL = DATABASE.select("SELECT * FROM `source`")
    validationSource()

    # Redirect to monitor page if not select server
    if len(serverselect) == 0 and request.path not in NOREDIRECT:
        page = 'monitor'

    if current_user.role == 'user' and page == 'users':
        page = 'deny'
    
    if current_user.username and serverselect:
        SESSION[current_user.username] = urlParse(serverselect[3])
    
    user_list = loadUserList(page)

    return render_template("index.html", page=page, sources=sourceSQL, serverselect=serverselect, userlist=user_list, session=SESSION)

@app.route('/js/<page>')
def js(page):
    return render_template(f'js/{page}')

@app.route('/api/v1/select-server', methods=['POST'])
def selectserver():
    global serverselect, session

    status_result = {
            'status': 'error',
            'msg': 'ข้อมูลไม่ถูกต้อง'
        }
    if request.is_json:
        requestData = request.get_json()
        result = DATABASE.select('SELECT * FROM source WHERE id = %s AND name = %s', (requestData['serverId'], requestData['serverName']))
        if len(result) != 0 :
            serverselect = result[0]
            SESSION[current_user.username] = urlParse(serverselect[2])
            return jsonify({
                'status': 'success',
                'msg': 'ok'
            })
        
        status_result = {
            'status': 'error',
            'msg': 'ไม่พบ Server นี้ในฐานข้อมูล'
        }

    return jsonify(status_result)

@app.route('/api/v1/source/create', methods=['POST'])
def createSource():
    result = request.json
    data = {
        'status': 'error',
        'msg': 'เกิดข้อผิดพลาด'
    }
    
    if result['name'] and result['endpoint'] and result['key']:
        results = DATABASE.insert("INSERT INTO `source`(`name`, `endpoint`, `apikey`) VALUES (%s,%s,%s)", (result['name'], result['endpoint'], result['key']))
        if results:
            data = {
                'status': 'success',
                'msg': 'ok'
            }
        else:
            data = {
                'status': 'error',
                'msg': 'เกิดข้อผิดพลาด'
            }

    return jsonify(data)

@app.route('/api/v1/source/delete', methods=['POST'])
def deleteSource():
    data = {
        'status': 'error',
        'msg': 'เกิดข้อผิดพลาด'
    }
    
    if request.is_json:
        print(request.get_json()['id'])
        results = DATABASE.insert('DELETE FROM `source` WHERE id = %s', (request.get_json()['id'],))
        if results:
            data = {
                'status': 'success',
                'msg': 'ลบข้อมูลแล้ว'
            }
        else:
            data = {
                'status': 'error',
                'msg': 'เกิดข้อผิดพลาด'
            }

    return jsonify(data)

if __name__ == "__main__":
    serverselect = {}

    DATABASE = Database(DB_HOST, DB_USERNAME, DB_PASSWORD, DB_NAME)

    app.run(port=5001, debug=True)
