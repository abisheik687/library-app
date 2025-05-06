import os
from flask import Flask, request, jsonify, render_template
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import relationship
from flask_cors import CORS
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
app = Flask(__name__)
# In-memory data stores (replace with database in a real application)
books = []
users = []
staff = []
admin_users = []
CORS(app)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///library.db'  # Using SQLite for simplicity
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

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

class User(db.Model):

    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False) # User's name
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False) # Store hashed passwords in production
    borrowed_books_rel = relationship("Book", back_populates="borrowed_by")
    borrowing_history_rel = relationship("BorrowingHistory", back_populates="user")

class Staff(db.Model):

    __tablename__ = 'staff'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(100)) # e.g., 'Librarian', 'Assistant'
    attendance_rel = relationship("Attendance", back_populates="staff")

class Admin(db.Model):

    __tablename__ = 'admins'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False) # Store hashed passwords in production


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
    staff_id = db.Column(db.Integer, db.ForeignKey('staff.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    status = db.Column(db.String(50)) # e.g., 'Present', 'Absent' # Attendance status
    staff = relationship("Staff", back_populates="attendance_rel")

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
    def add_user(self, name, email, password="password"):
        user = User(name=name, email=email, password=password)

        db.session.add(user)
        db.session.commit()
        return user
    
    def get_user_by_id(self, user_id):
        return db.session.get(User, user_id)

    # --- Staff Management ---
    def add_staff(self, name, role, password="password"):
        staff_member = Staff(name=name, role=role, password=password)

        db.session.add(staff_member)
        db.session.commit()
        return staff_member
    
    def get_staff_by_id(self, staff_id):
        return db.session.get(Staff, staff_id)

    # --- Admin Management ---
    def add_admin(self, username, password="password"):
        admin = Admin(username=username, password=password)

        db.session.add(admin)
        db.session.commit()
        return admin 
    def get_admin_by_id(self, admin_id):
        return db.session.get(Admin, admin_id)


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
        user_book_matrix = np.zeros((len(User.query.all()), len(all_books)))
        for u in User.query.all():
            for history in u.borrowing_history_rel:
                if history.book_id in book_index:
                    user_book_matrix[u.id - 1, book_index[history.book_id]] = 1
        
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
    # --- UI Routes ---
@app.route('/')
def index():
    return render_template('index.html')

# Initialize the library
library = Library()
# Add some initial data (for testing)

with app.app_context():
    db.create_all() # Create database tables if they don't exist
    if not Admin.query.filter_by(username="admin").first():
        library.add_admin("admin", "admin")
    if not User.query.filter_by(email="alice@example.com").first():
        library.add_user("Alice", "alice@example.com")
        # Add some initial books and borrowing history for recommendations
        if not Book.query.first(): # Add books only if none exist
            library.add_book("The Hitchhiker's Guide to the Galaxy", "Douglas Adams", "978-0345391803")
            library.add_book("Pride and Prejudice", "Jane Austen", "978-0141439518")
    if not Staff.query.filter_by(name="Bob").first():
        library.add_staff("Bob", "Librarian", password="password")
# --- Flask Routes ---

@app.route("/api/books", methods=["GET"])
def get_books():
    book_list = [{"id": book.id, "title": book.title, "author": book.author, "isbn": book.isbn, "is_borrowed": book.is_borrowed} for book in Book.query.all()]
    return jsonify(book_list)

@app.route("/api/users", methods=["GET"])
def get_users():
    user_list = [{"id": user.id, "name": user.name, "email": user.email} for user in User.query.all()]
    return jsonify(user_list)

@app.route("/api/staff", methods=["GET"])
def get_staff():
    staff_list = [{"id": staff_member.id, "name": staff_member.name, "role": staff_member.role} for staff_member in Staff.query.all()]
    return jsonify(staff_list)

@app.route("/api/recommendations/<int:user_id>", methods=["GET"])
def get_book_recommendations(user_id):
    user = db.session.get(User, user_id)
    if user:
        recommendations = library.recommend_books(user)
        recommended_books = [{"id": book.book_id, "title": book.title, "author": book.author} for book in recommendations]
        return jsonify(recommended_books)
    return jsonify({"message": "User not found"}), 404

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    # Check for admin
    for admin in Admin.query.all():
        if admin.username == username and admin.password == password: # In production, use hashed password comparison
            return jsonify({"role": "admin", "id": admin.id, "username": admin.username})

    # Check for staff
    for staff_member in Staff.query.all():
        if staff_member.name == username and staff_member.password == password: # In production, use hashed password comparison
            return jsonify({"role": "staff", "id": staff_member.id, "name": staff_member.name})

    # Check for user
    for user in User.query.all():
        if user.name == username and user.password == password: # In production, use hashed password comparison
            return jsonify({"role": "user", "id": user.id, "name": user.name})

    return jsonify({"message": "Invalid credentials"}), 401

@app.route("/api/user/status/<int:user_id>", methods=["GET"])
def get_user_status(user_id):
    user = library.get_user_by_id(user_id)
    if user:
        borrowed_books = [{"id": book.id, "title": book.title, "borrow_date": book.borrow_date.strftime('%Y-%m-%d') if book.borrow_date else 'N/A', "return_date": book.return_date.strftime('%Y-%m-%d') if book.return_date else 'N/A'} for book in user.borrowed_books_rel if book.borrowed_by_id == user_id]
        return jsonify({"user_id": user.id, "name": user.name, "borrowed_books": borrowed_books})
    return jsonify({"message": "User not found"}), 404

@app.route("/api/admin/dashboard", methods=["GET"])
def admin_dashboard():
    total_books = len(library.books)
    borrowed_books_count = len([book for book in library.books if book.is_borrowed])
    total_users = User.query.count()
    total_staff = Staff.query.count()
    return jsonify({ # Basic admin dashboard data
        "total_books": total_books,
        "borrowed_books_count": borrowed_books_count,
        "total_users": total_users,
        "total_staff": total_staff
    })

@app.route("/api/staff/dashboard/<int:staff_id>", methods=["GET"])
def staff_dashboard(staff_id):
    # Staff dashboard could show lists of users and books
    user_list = [{"id": user.id, "name": user.name, "email": user.email} for user in User.query.all()]
    book_list = [{"id": book.id, "title": book.title, "author": book.author, "is_borrowed": book.is_borrowed} for book in Book.query.all()]
    return jsonify({"users": user_list, "books": book_list})

@app.route("/api/admin/analytics", methods=["GET"])
def admin_analytics():
    # Real-time analytics and reporting
    borrowing_trends = library.get_borrowing_history_analysis()
    overdue_books = library.predict_overdue()
    # Format data for API response
    borrowing_trends_data = borrowing_trends
    overdue_books_data = [{"id": book.id, "title": book.title, "borrowed_by": book.borrowed_by.name if book.borrowed_by else 'N/A', "return_date": book.return_date.strftime('%Y-%m-%d') if book.return_date else 'N/A'} for book in overdue_books]

    return jsonify({
        "borrowing_trends": borrowing_trends_data,
        "overdue_books": overdue_books_data # This line caused the issue
    })
    
@app.route("/api/staff/attendance", methods=["POST"])
def staff_attendance():
    data = request.json
    staff_id = data.get("staff_id")
    status = data.get("status") # e.g., 'Present'

    staff_member = library.get_staff_by_id(staff_id)
    if staff_member:
        attendance_record = Attendance(staff_id=staff_id, date=datetime.now().date(), status=status)
        db.session.add(attendance_record)
        db.session.commit()
        return jsonify({"message": "Attendance recorded successfully"})

    return jsonify({"message": "Staff member not found"}), 404

@app.route("/api/borrow", methods=["POST"])
def borrow_book():
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
def return_book():
    data = request.json
    book_id = data.get("book_id")

    book = library.get_book_by_id(book_id)
    if book and book.is_borrowed:
        book.is_borrowed = False
        book.borrowed_by_id = None
        # Update borrowing history with return date
        borrowing_record = BorrowingHistory.query.filter_by(book_id=book_id, user_id=book.borrowed_by_id, return_date=None).first()
        if borrowing_record:
            borrowing_record.return_date = datetime.now()
        db.session.commit()
        return jsonify({"message": "Book returned successfully"})
    return jsonify({"message": "Failed to return book"}), 400



def main():
    app.run(port=int(os.environ.get('PORT', 80)))

if __name__ == "__main__":
    main()
