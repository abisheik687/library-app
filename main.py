import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import relationship
from flask_cors import CORS
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here' # Replace with a real secret key
CORS(app)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///library.db'  # Using SQLite for simplicity
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # The name of the login route function

@login_manager.user_loader
def load_user(user_id):
    # user_id is the primary key of the user object
    # This function is called every time a page is loaded that requires a logged-in user
    # It should return the user object or None if the user is not found
    return User.query.get(int(user_id))

# --- Routes ---

@app.route('/')
@login_required # Protect this route
def index():
    # This route will now require the user to be logged in
    # You can render a dashboard or main page here
    return render_template('index.html') # Or a new dashboard.html

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        # If the user is already logged in, redirect to the index page
        return redirect(url_for('index'))

    if request.method == 'POST':
        username_or_email = request.form.get('username')
        password = request.form.get('password')

        # Try to find user by username or email
        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()

        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Logged in successfully.', 'success')
            next_page = request.args.get('next')
            # Redirect based on role
            if user.role == 'admin':
                return redirect(next_page or url_for('admin_dashboard'))
            elif user.role == 'staff':
                return redirect(next_page or url_for('staff_dashboard'))
            else:
                return redirect(next_page or url_for('index')) # Default user dashboard/index

        flash('Invalid username/email or password.', 'error')

    return render_template('login.html')

@app.route('/logout')
@login_required # Protect this route
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index')) # Redirect to index or login page

# --- Database Models ---

class Book(db.Model):

    __tablename__ = 'books'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False) # Book title
    author = db.Column(db.String(255), nullable=False) # Book author
    isbn = db.Column(db.String(20), unique=True, nullable=False) # International Standard Book Number
    category = db.Column(db.String(100)) # Intelligent categorization (can be populated by AI)
    is_borrowed = db.Column(db.Boolean, default=False)  # Borrowed status
    borrowed_by_id = db.Column(db.Integer, db.ForeignKey('users.id'))  # User who borrowed the book
    borrowed_by = relationship("User", back_populates="borrowed_books_rel")
    borrow_date = db.Column(db.DateTime)
    return_date = db.Column(db.DateTime) # Expected return date

class User(UserMixin, db.Model):

    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False) # Added username field
    name = db.Column(db.String(255)) # User's name (optional)
    email = db.Column(db.String(255), unique=True) # Email (optional)
    password_hash = db.Column(db.String(255), nullable=False) # Store hashed passwords
    role = db.Column(db.String(50), default='user') # 'user', 'staff', 'admin'
    borrowed_books_rel = relationship("Book", back_populates="borrowed_by")
    borrowing_history_rel = relationship("BorrowingHistory", back_populates="user")
    attendance_rel = relationship("Attendance", back_populates="staff") # For staff role

    # Required by Flask-Login UserMixin
    def get_id(self):
        return str(self.id)

    def is_active(self):
        # You can add logic here to check if a user account is active
        return True

    def is_authenticated(self):
        # This should return True if the user is authenticated
        return True # Assuming all users in the DB are authenticated

    def is_anonymous(self):
        # This should return True if this is a generic anonymous user
        return False

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self):
        return self.role == 'admin'

    @property
    def is_staff(self):
        return self.role == 'staff'


# Removed separate Staff and Admin models, consolidating into User
# class Staff(db.Model):
#     __tablename__ = 'staff'
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(255), nullable=False)
#     password_hash = db.Column(db.String(255), nullable=False)
#     role = db.Column(db.String(100)) # e.g., 'Librarian', 'Assistant'
#     attendance_rel = relationship("Attendance", back_populates="staff")

#     def set_password(self, password):
#         self.password_hash = generate_password_hash(password)

#     def check_password(self, password):
#         return check_password_hash(self.password_hash, password)

# class Admin(db.Model):
#     __tablename__ = 'admins'
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(255), unique=True, nullable=False)
#     password_hash = db.Column(db.String(255), nullable=False)

#     def set_password(self, password):
#         self.password_hash = generate_password_hash(password)

#     def check_password(self, password):
#         return check_password_hash(self.password_hash, password)


class BorrowingHistory(db.Model):
    __tablename__ = 'borrowing_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False) # User who borrowed
    book_id = db.Column(db.Integer, db.ForeignKey('books.id'), nullable=False)
    borrow_date = db.Column(db.DateTime, nullable=False)
    return_date = db.Column(db.DateTime)
    user = relationship("User", back_populates="borrowing_history_rel") # Relationship to User
    book = relationship("Book")

class Attendance(db.Model):

    __tablename__ = 'attendance'
    id = db.Column(db.Integer, primary_key=True)
    staff_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False) # Link to User table
    date = db.Column(db.Date, nullable=False)
    status = db.Column(db.String(50)) # e.g., 'Present', 'Absent' # Attendance status
    staff = relationship("User", back_populates="attendance_rel") # Relationship to User


# --- Class Definitions ---

class Library():

    # --- Book Management ---
    def add_book(self, title, author, isbn):
        book = Book(title=title, author=author, isbn=isbn)

        db.session.add(book)
        db.session.commit()
        return book
    
    def get_book_by_id(self, book_id):
        return db.session.get(Book, book_id)

    # --- User Management ---
    def add_user(self, username, password, name=None, email=None, role='user'):
        user = User(username=username, name=name, email=email, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return user

    def get_user_by_id(self, user_id):
        return db.session.get(User, user_id)

    def get_user_by_username(self, username):
        return User.query.filter_by(username=username).first()

    def get_user_by_email(self, email):
        return User.query.filter_by(email=email).first()

    # Removed separate staff and admin getter methods
    # def get_staff_by_id(self, staff_id):
    #     return db.session.get(Staff, staff_id)

    # def get_staff_by_name(self, name):
    #     return Staff.query.filter_by(name=name).first()

    # def get_admin_by_id(self, admin_id):
    #     return db.session.get(Admin, admin_id)

    # def get_admin_by_username(self, username):
    #     return Admin.query.filter_by(username=username).first()


    # --- AI Features ---
    def recommend_books(self, user):
        """
        AI-powered book recommendation based on borrowing history.
        This is a collaborative filtering approach (item-based).
        Can be extended with content-based features (genres, keywords).
        """
        if not user.borrowing_history_rel:
            return Book.query.limit(10).all()  # Recommend popular books if no history

        borrowed_book_ids = [history.book_id for history in user.borrowing_history_rel if history.book_id is not None]
        all_books = Book.query.all()
        book_index = {book.id: i for i, book in enumerate(all_books)}

        # Create a user-book matrix (simplified: just if borrowed)
        all_users = User.query.all()
        user_index = {u.id: i for i, u in enumerate(all_users)}
        user_book_matrix = np.zeros((len(all_users), len(all_books)))
        for u in all_users:
            for history in u.borrowing_history_rel:
                if history.book_id in book_index:
                    user_book_matrix[user_index[u.id], book_index[history.book_id]] = 1
        
        # Calculate item similarity (cosine similarity)
        item_similarity = cosine_similarity(user_book_matrix.T)

        recommended_scores = defaultdict(float)
        for borrowed_book_id in borrowed_book_ids:
            if borrowed_book_id in book_index:
                borrowed_book_index = book_index[borrowed_book_id]
                similar_books_indices = np.argsort(item_similarity[borrowed_book_index])[::-1] # Sort descending
                for i in similar_books_indices:
                    book_id = all_books[i].id
                    if book_id not in borrowed_book_ids and not all_books[i].is_borrowed:
                        recommended_scores[book_id] += item_similarity[borrowed_book_index, i]

        sorted_recommendations = sorted(recommended_scores.items(), key=lambda item: item[1], reverse=True)[:10]  # Get top 10
        recommended_book_ids = [book_id for book_id, score in sorted_recommendations]

        return Book.query.filter(Book.id.in_(recommended_book_ids)).all()

    def predict_overdue(self):
        """
        Predicts books likely to be overdue.
        Simple implementation: checks books that are past their return date.
        Can be enhanced with ML models based on user history and book type.  
        """
        return Book.query.filter(Book.is_borrowed == True, Book.return_date < datetime.now().date(), Book.return_date is not None).all()

    def get_borrowing_history_analysis(self):
        """
        Analyzes borrowing history for trends and popular books.
        """
        history = BorrowingHistory.query.all()
        borrow_counts = defaultdict(int)
        book_titles = {book.id: book.title for book in Book.query.all()}
        for item in history:
            borrow_counts[item.book_id] += 1
        analysis_results = [{"book_id": book_id, "title": book_titles.get(book_id, 'Unknown'), "borrow_count": count} for book_id, count in borrow_counts.items()]
        return sorted(analysis_results, key=lambda item: item["borrow_count"], reverse=True)
# Initialize the library
library = Library()
# Add some initial data (for testing)

with app.app_context():
    db.create_all() # Create database tables if they don't exist

    # Add initial admin
    if not User.query.filter_by(username="admin").first():
        library.add_user(username="admin", password="admin", role="admin")

    # Add initial user (Alice)
    if not User.query.filter_by(email="alice@example.com").first():
        library.add_user(username="alice", name="Alice", email="alice@example.com", password="password", role="user")

    # Add some initial books and borrowing history for recommendations
    if not Book.query.first(): # Add books only if none exist
        library.add_book("The Hitchhiker's Guide to the Galaxy", "Douglas Adams", "978-0345391803")
        library.add_book("Pride and Prejudice", "Jane Austen", "978-0141439518")

    # Add 4 more staff members
    if not User.query.filter_by(username="staff1").first():
        library.add_user(username="staff1", name="Staff1", password="password", role="staff")
    if not User.query.filter_by(username="staff2").first():
        library.add_user(username="staff2", name="Staff2", password="password", role="staff")
    if not User.query.filter_by(username="staff3").first():
        library.add_user(username="staff3", name="Staff3", password="password", role="staff")
    if not User.query.filter_by(username="staff4").first():
        library.add_user(username="staff4", name="Staff4", password="password", role="staff")

    # Add 4 more users (5 total including Alice)
    if not User.query.filter_by(username="user1").first():
        library.add_user(username="user1", name="User1", email="user1@example.com", password="password", role="user")
    if not User.query.filter_by(username="user2").first():
        library.add_user(username="user2", name="User2", email="user2@example.com", password="password", role="user")
    if not User.query.filter_by(username="user3").first():
        library.add_user(username="user3", name="User3", email="user3@example.com", password="password", role="user")
    if not User.query.filter_by(username="user4").first():
        library.add_user(username="user4", name="User4", email="user4@example.com", password="password", role="user")

    db.session.commit() # Commit all changes at once

# --- API Routes ---

@app.route("/api/books", methods=["GET"])
def get_books():
    book_list = [{"id": book.id, "title": book.title, "author": book.author, "isbn": book.isbn, "is_borrowed": book.is_borrowed} for book in Book.query.all()]
    return jsonify(book_list)

@app.route("/api/users", methods=["GET"])
@login_required
def get_users():
    # Only admin and staff should be able to list all users
    if not current_user.is_admin and not current_user.is_staff:
        return jsonify({"message": "Unauthorized"}), 403
    user_list = [{"id": user.id, "username": user.username, "name": user.name, "email": user.email, "role": user.role} for user in User.query.all()]
    return jsonify(user_list)

@app.route("/api/staff", methods=["GET"])
@login_required
def get_staff():
    # Only admin should be able to list all staff
    if not current_user.is_admin:
         return jsonify({"message": "Unauthorized"}), 403
    staff_list = [{"id": user.id, "username": user.username, "name": user.name, "role": user.role} for user in User.query.filter_by(role='staff').all()]
    return jsonify(staff_list)

@app.route("/api/recommendations/<int:user_id>", methods=["GET"])
@login_required
def get_book_recommendations(user_id):
    # Users can only get recommendations for themselves, admin/staff can get for any user
    if not current_user.is_admin and not current_user.is_staff and current_user.id != user_id:
         return jsonify({"message": "Unauthorized"}), 403

    user = library.get_user_by_id(user_id)
    if user:
        recommendations = library.recommend_books(user)
        recommended_books = [{"id": book.id, "title": book.title, "author": book.author} for book in recommendations] # Corrected to use book.id
        return jsonify(recommended_books)
    return jsonify({"message": "User not found"}), 404

# NOTE: The existing /api/login route is redundant with the new /login route
# and should ideally be removed or refactored if a separate API login is needed.
# For now, I will comment it out to avoid confusion and potential issues.
# @app.route("/api/login", methods=["POST"])
# def api_login():
#     data = request.json
#     username = data.get("username")
#     password = data.get("password")

#     # Check for admin
#     for admin in Admin.query.all():
#         if admin.username == username and admin.password == password: # In production, use hashed password comparison
#             return jsonify({"role": "admin", "id": admin.id, "username": admin.username})

#     # Check for staff
#     for staff_member in Staff.query.all():
#         if staff_member.name == username and staff_member.password == password: # In production, use hashed password comparison
#             return jsonify({"role": "staff", "id": staff_member.id, "name": staff_member.name})

#     # Check for user
#     for user in User.query.all():
#         if user.name == username and user.password == password: # In production, use hashed password comparison
#             return jsonify({"role": "user", "id": user.id, "name": user.name})

#     return jsonify({"message": "Invalid credentials"}), 401

@app.route("/api/user/status/<int:user_id>", methods=["GET"])
@login_required # Protect this route
def get_user_status(user_id):
    # Users can only get their own status, admin/staff can get for any user
    if not current_user.is_admin and not current_user.is_staff and current_user.id != user_id:
         return jsonify({"message": "Unauthorized"}), 403

    user = library.get_user_by_id(user_id)
    if user:
        borrowed_books = [{"id": book.id, "title": book.title, "borrow_date": book.borrow_date.strftime('%Y-%m-%d') if book.borrow_date else 'N/A', "return_date": book.return_date.strftime('%Y-%m-%d') if book.return_date else 'N/A'} for book in user.borrowed_books_rel if book.borrowed_by_id == user_id]
        return jsonify({"user_id": user.id, "name": user.name, "borrowed_books": borrowed_books})
    return jsonify({"message": "User not found"}), 404

@app.route("/api/admin/dashboard", methods=["GET"])
@login_required # Protect this route
def admin_dashboard():
    # Ensure only admins can access this route
    if not current_user.is_admin:
        return jsonify({"message": "Unauthorized"}), 403

    total_books = Book.query.count() # Corrected to use Book.query.count()
    borrowed_books_count = Book.query.filter_by(is_borrowed=True).count() # Corrected
    total_users = User.query.count()
    total_staff = User.query.filter_by(role='staff').count() # Corrected
    return jsonify({ # Basic admin dashboard data
        "total_books": total_books,
        "borrowed_books_count": borrowed_books_count,
        "total_users": total_users,
        "total_staff": total_staff
    })

@app.route("/api/staff/dashboard", methods=["GET"]) # Removed staff_id from route
@login_required # Protect this route
def staff_dashboard():
    # Ensure only staff and admin can access this route
    if not current_user.is_admin and not current_user.is_staff:
        return jsonify({"message": "Unauthorized"}), 403

    # Staff dashboard could show lists of users and books
    user_list = [{"id": user.id, "username": user.username, "name": user.name, "email": user.email, "role": user.role} for user in User.query.all()]
    book_list = [{"id": book.id, "title": book.title, "author": book.author, "is_borrowed": book.is_borrowed} for book in Book.query.all()]
    return jsonify({"users": user_list, "books": book_list})

@app.route("/api/admin/analytics", methods=["GET"])
@login_required # Protect this route
def admin_analytics():
    # Ensure only admins can access this route
    if not current_user.is_admin:
        return jsonify({"message": "Unauthorized"}), 403

    # Real-time analytics and reporting
    borrowing_trends = library.get_borrowing_history_analysis()
    overdue_books = library.predict_overdue()
    # Format data for API response
    borrowing_trends_data = borrowing_trends
    overdue_books_data = [{"id": book.id, "title": book.title, "borrowed_by": book.borrowed_by.name if book.borrowed_by else 'N/A', "return_date": book.return_date.strftime('%Y-%m-%d') if book.return_date else 'N/A'} for book in overdue_books]

    return jsonify({
        "borrowing_trends": borrowing_trends_data,
        "overdue_books": overdue_books_data
    })
    
@app.route("/api/staff/attendance", methods=["POST"])
@login_required # Protect this route
def staff_attendance():
    # Ensure only staff and admin can record attendance
    if not current_user.is_admin and not current_user.is_staff:
        return jsonify({"message": "Unauthorized"}), 403

    data = request.json
    staff_id = data.get("staff_id") # This should ideally be the logged-in staff's ID
    status = data.get("status") # e.g., 'Present'

    # For simplicity, assuming the staff_id is passed in the request body.
    # In a real application, you'd likely use current_user.id
    staff_member = library.get_user_by_id(staff_id) # Use get_user_by_id
    if staff_member and staff_member.role == 'staff': # Check if the user is staff
        attendance_record = Attendance(staff_id=staff_id, date=datetime.now().date(), status=status)
        db.session.add(attendance_record)
        db.session.commit()
        return jsonify({"message": "Attendance recorded successfully"})

    return jsonify({"message": "Staff member not found or not a staff role"}), 404

@app.route("/api/borrow", methods=["POST"])
@login_required # Protect this route
def borrow_book():
    # Ensure only staff and admin can borrow books for users
    if not current_user.is_admin and not current_user.is_staff:
        return jsonify({"message": "Unauthorized"}), 403

    data = request.json
    user_id = data.get("user_id")
    book_id = data.get("book_id")
    user = library.get_user_by_id(user_id)
    book = library.get_book_by_id(book_id)
    if user and book and not book.is_borrowed:
        book.is_borrowed = True
        book.borrowed_by_id = user_id
        book.borrow_date = datetime.now()

        book.return_date = book.borrow_date + timedelta(days=14) # Example: 14 days loan period
        borrowing_record = BorrowingHistory(user_id=user_id, book_id=book_id, borrow_date=book.borrow_date)
        db.session.add(borrowing_record)
        db.session.commit()

        return jsonify({"message": "Book borrowed successfully"})
    return jsonify({"message": "Failed to borrow book"}), 400
# Add more API endpoints for borrowing, returning, staff management, admin dashboards, etc.

@app.route("/api/return", methods=["POST"])
@login_required # Protect this route
def return_book():
    # Ensure only staff and admin can return books
    if not current_user.is_admin and not current_user.is_staff:
        return jsonify({"message": "Unauthorized"}), 403

    data = request.json
    book_id = data.get("book_id")

    book = library.get_book_by_id(book_id)
    if book and book.is_borrowed:
        # Find the active borrowing record
        borrowing_record = BorrowingHistory.query.filter_by(book_id=book_id, return_date=None).first() # Removed user_id check here as it's not needed for finding the active record

        book.is_borrowed = False
        book.borrowed_by_id = None
        book.borrow_date = None # Reset borrow date
        book.return_date = None # Reset return date

        if borrowing_record:
            borrowing_record.return_date = datetime.now()
        db.session.commit()
        return jsonify({"message": "Book returned successfully"})
    return jsonify({"message": "Failed to return book"}), 400

# --- Book Management API Endpoints ---

@app.route("/api/book/<int:book_id>", methods=["GET"])
def get_book(book_id):
    book = library.get_book_by_id(book_id)
    if book:
        return jsonify({"id": book.id, "title": book.title, "author": book.author, "isbn": book.isbn, "category": book.category, "is_borrowed": book.is_borrowed, "borrowed_by_id": book.borrowed_by_id})
    return jsonify({"message": "Book not found"}), 404

@app.route("/api/book", methods=["POST"])
@login_required
def add_book():
    # Only admin and staff can add books
    if not current_user.is_admin and not current_user.is_staff:
        return jsonify({"message": "Unauthorized"}), 403

    data = request.json
    title = data.get("title")
    author = data.get("author")
    isbn = data.get("isbn")

    if not title or not author or not isbn:
        return jsonify({"message": "Missing required fields (title, author, isbn)"}), 400

    # Check if book with the same ISBN already exists
    existing_book = Book.query.filter_by(isbn=isbn).first()
    if existing_book:
        return jsonify({"message": "Book with this ISBN already exists"}), 409 # Conflict

    book = library.add_book(title, author, isbn)
    return jsonify({"message": "Book added successfully", "book_id": book.id}), 201 # Created

@app.route("/api/book/<int:book_id>", methods=["PUT"])
@login_required
def update_book(book_id):
    # Only admin and staff can update books
    if not current_user.is_admin and not current_user.is_staff:
        return jsonify({"message": "Unauthorized"}), 403

    book = library.get_book_by_id(book_id)
    if not book:
        return jsonify({"message": "Book not found"}), 404

    data = request.json
    book.title = data.get("title", book.title)
    book.author = data.get("author", book.author)
    book.isbn = data.get("isbn", book.isbn)
    book.category = data.get("category", book.category)
    # Note: is_borrowed, borrowed_by_id, borrow_date, return_date should be managed by borrow/return routes

    db.session.commit()
    return jsonify({"message": "Book updated successfully"})

@app.route("/api/book/<int:book_id>", methods=["DELETE"])
@login_required
def delete_book(book_id):
    # Only admin and staff can delete books
    if not current_user.is_admin and not current_user.is_staff:
        return jsonify({"message": "Unauthorized"}), 403

    book = library.get_book_by_id(book_id)
    if not book:
        return jsonify({"message": "Book not found"}), 404

    db.session.delete(book)
    db.session.commit()
    return jsonify({"message": "Book deleted successfully"})

# --- User Management API Endpoints ---

@app.route("/api/user/<int:user_id>", methods=["GET"])
@login_required
def get_user(user_id):
    # Users can only view their own profile, admin/staff can view any user
    if not current_user.is_admin and not current_user.is_staff and current_user.id != user_id:
         return jsonify({"message": "Unauthorized"}), 403

    user = library.get_user_by_id(user_id)
    if user:
        return jsonify({"id": user.id, "username": user.username, "name": user.name, "email": user.email, "role": user.role})
    return jsonify({"message": "User not found"}), 404

@app.route("/api/user", methods=["POST"])
@login_required
def add_user():
    # Only admin can add users (including staff and other admins)
    if not current_user.is_admin:
        return jsonify({"message": "Unauthorized"}), 403

    data = request.json
    username = data.get("username")
    password = data.get("password")
    name = data.get("name")
    email = data.get("email")
    role = data.get("role", "user") # Default role is 'user'

    if not username or not password:
        return jsonify({"message": "Missing required fields (username, password)"}), 400

    # Check if user with the same username or email already exists
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({"message": "User with this username already exists"}), 409 # Conflict
    if email and User.query.filter_by(email=email).first():
         return jsonify({"message": "User with this email already exists"}), 409 # Conflict

    user = library.add_user(username=username, password=password, name=name, email=email, role=role)
    return jsonify({"message": "User added successfully", "user_id": user.id}), 201 # Created

@app.route("/api/user/<int:user_id>", methods=["PUT"])
@login_required
def update_user(user_id):
    # Admin can update any user, users can update their own profile (except role)
    if not current_user.is_admin and current_user.id != user_id:
         return jsonify({"message": "Unauthorized"}), 403

    user = library.get_user_by_id(user_id)
    if not user:
        return jsonify({"message": "User not found"}), 404

    data = request.json
    user.name = data.get("name", user.name)
    user.email = data.get("email", user.email)
    # Allow admin to change role
    if current_user.is_admin:
        user.role = data.get("role", user.role)
    # Allow changing password
    new_password = data.get("password")
    if new_password:
        user.set_password(new_password)

    db.session.commit()
    return jsonify({"message": "User updated successfully"})

@app.route("/api/user/<int:user_id>", methods=["DELETE"])
@login_required
def delete_user(user_id):
    # Only admin can delete users
    if not current_user.is_admin:
        return jsonify({"message": "Unauthorized"}), 403

    user = library.get_user_by_id(user_id)
    if not user:
        return jsonify({"message": "User not found"}), 404

    # Prevent admin from deleting themselves
    if current_user.id == user_id:
        return jsonify({"message": "Cannot delete your own admin account"}), 400

    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": "User deleted successfully"})


def main():
    app.run(port=int(os.environ.get('PORT', 80)))

if __name__ == "__main__":
    main()
