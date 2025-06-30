from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
import json

app = Flask(__name__)

# SQL Server Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = (
    'mssql+pyodbc://DESKTOP-S22JEMV\\SQLEXPRESS/SmartPromoDb_v2024'
    '?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

def get_database_structure():
    """Get database structure"""
    with app.app_context():
        try:
            # Get all tables
            query = text("""
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_TYPE = 'BASE TABLE' 
                ORDER BY TABLE_NAME
            """)
            result = db.session.execute(query)
            tables = [row[0] for row in result.fetchall()]
            
            structure = {}
            for table in tables:
                try:
                    query = text(f"""
                        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_NAME = '{table}'
                        ORDER BY ORDINAL_POSITION
                    """)
                    result = db.session.execute(query)
                    columns = []
                    for row in result.fetchall():
                        columns.append(f"{row[0]} ({row[1]}) {'NULL' if row[2] == 'YES' else 'NOT NULL'}")
                    structure[table] = columns
                except Exception as e:
                    structure[table] = [f"Error: {e}"]
            
            return structure
        except Exception as e:
            return {"error": str(e)}

if __name__ == '__main__':
    structure = get_database_structure()
    print(json.dumps(structure, indent=2))
